"""Pure classical Digital Image Processing (PCD) motion detection pipeline.

Pipeline stages (100% classical OpenCV + NumPy, zero ML/DL):
1. BGR -> Grayscale conversion + Gaussian Blur (noise reduction).
2. Frame Differencing (`cv2.absdiff`):
   - Consecutive mode (Green Light): compares current frame against previous frame.
   - Reference mode (Red Light): compares current frame against locked reference frame.
3. Binary Thresholding (`cv2.threshold`) to isolate real pixel changes from light flicker.
4. Morphological Operations:
   - Opening (`cv2.MORPH_OPEN`) to suppress tiny isolated noise blobs.
   - Dilation (`cv2.dilate`) to merge nearby fragmented motion regions.
5. Contour Analysis & Filtering (`cv2.findContours`, `cv2.contourArea`, `cv2.boundingRect`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import DEFAULT_DIFFICULTY, DIFFICULTY_PRESETS, DifficultyConfig

BoundingBox = Tuple[int, int, int, int]


def _ensure_odd_kernel(size: Tuple[int, int]) -> Tuple[int, int]:
    """Ensure kernel dimensions are positive odd integers for Gaussian/Morph ops."""
    kw = max(1, int(size[0]))
    kh = max(1, int(size[1]))
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1
    return (kw, kh)


@dataclass
class MotionResult:
    """Structured output of a single PCD motion detection pass."""

    motion_score: float
    bounding_boxes: List[BoundingBox]
    contours: List[np.ndarray]
    max_contour_area: float
    is_motion_detected: bool
    reference_ready: bool
    gray_blur: np.ndarray
    frame_diff: np.ndarray
    thresh: np.ndarray
    morphed: np.ndarray
    mode: str = "consecutive"
    contour_areas: List[float] = field(default_factory=list)


class MotionDetector:
    """Classical PCD motion detector using Frame Differencing & Morphology."""

    def __init__(self, config: Optional[DifficultyConfig] = None) -> None:
        self._config: DifficultyConfig = config or DIFFICULTY_PRESETS[DEFAULT_DIFFICULTY]
        self._prev_gray_blur: Optional[np.ndarray] = None
        self._reference_gray_blur: Optional[np.ndarray] = None
        self._build_kernels()

    @property
    def config(self) -> DifficultyConfig:
        """Return active PCD configuration."""
        return self._config

    @property
    def has_reference(self) -> bool:
        """Whether a locked reference frame is available for Red Light comparison."""
        return self._reference_gray_blur is not None

    def update_config(self, config: DifficultyConfig) -> None:
        """Update PCD parameters dynamically (e.g., when switching difficulty)."""
        self._config = config
        self._build_kernels()

    def _build_kernels(self) -> None:
        self._blur_kernel = _ensure_odd_kernel(self._config.blur_kernel_size)
        open_k = _ensure_odd_kernel(self._config.morph_open_kernel_size)
        dilate_k = _ensure_odd_kernel(self._config.morph_dilate_kernel_size)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_k)
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_k)

    def reset(self) -> None:
        """Clear both previous and reference frames."""
        self._prev_gray_blur = None
        self._reference_gray_blur = None

    def reset_previous(self) -> None:
        """Clear consecutive previous frame (useful across state transitions)."""
        self._prev_gray_blur = None

    def reset_reference(self) -> None:
        """Clear Red Light reference frame (e.g., at start of grace period)."""
        self._reference_gray_blur = None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Stage 1: Convert BGR frame to Grayscale and apply Gaussian Blur."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame.copy()
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        return cv2.GaussianBlur(gray, self._blur_kernel, 0)

    def capture_reference_frame(self, frame: np.ndarray) -> np.ndarray:
        """Lock the preprocessed reference frame used during Red Light."""
        blurred = self.preprocess(frame)
        self._reference_gray_blur = blurred.copy()
        return self._reference_gray_blur

    def _empty_result(
        self,
        gray_blur: np.ndarray,
        mode: str,
        reference_ready: bool,
    ) -> MotionResult:
        zeros = np.zeros_like(gray_blur, dtype=np.uint8)
        return MotionResult(
            motion_score=0.0,
            bounding_boxes=[],
            contours=[],
            max_contour_area=0.0,
            is_motion_detected=False,
            reference_ready=reference_ready,
            gray_blur=gray_blur,
            frame_diff=zeros,
            thresh=zeros.copy(),
            morphed=zeros.copy(),
            mode=mode,
            contour_areas=[],
        )

    def _analyze_difference(
        self,
        current_blur: np.ndarray,
        baseline_blur: np.ndarray,
        mode: str,
        reference_ready: bool,
    ) -> MotionResult:
        """Run Stages 2-5 on two preprocessed grayscale frames."""
        # If resolution changed dynamically, match baseline shape safely
        if current_blur.shape != baseline_blur.shape:
            baseline_blur = cv2.resize(
                baseline_blur,
                (current_blur.shape[1], current_blur.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        # Stage 2: Frame Differencing
        frame_diff = cv2.absdiff(baseline_blur, current_blur)

        # Stage 3: Binary Thresholding
        _, thresh = cv2.threshold(
            frame_diff,
            self._config.threshold_value,
            255,
            cv2.THRESH_BINARY,
        )

        # Stage 4: Morphological Operations (Opening + Dilation)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._open_kernel)
        morphed = cv2.dilate(
            opened,
            self._dilate_kernel,
            iterations=self._config.morph_dilate_iterations,
        )

        # Stage 5: Contour Analysis & Area Filtering
        contours, _ = cv2.findContours(
            morphed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        filtered_items: List[Tuple[float, BoundingBox, np.ndarray]] = []
        min_area = float(self._config.min_contour_area)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                filtered_items.append((area, (int(x), int(y), int(w), int(h)), contour))

        # Sort largest motion region first
        filtered_items.sort(key=lambda item: item[0], reverse=True)

        contour_areas = [item[0] for item in filtered_items]
        bounding_boxes = [item[1] for item in filtered_items]
        filtered_contours = [item[2] for item in filtered_items]

        motion_score = float(sum(contour_areas))
        max_area = contour_areas[0] if contour_areas else 0.0

        return MotionResult(
            motion_score=motion_score,
            bounding_boxes=bounding_boxes,
            contours=filtered_contours,
            max_contour_area=max_area,
            is_motion_detected=len(bounding_boxes) > 0,
            reference_ready=reference_ready,
            gray_blur=current_blur,
            frame_diff=frame_diff,
            thresh=thresh,
            morphed=morphed,
            mode=mode,
            contour_areas=contour_areas,
        )

    def detect_consecutive(self, frame: np.ndarray) -> MotionResult:
        """Detect motion by comparing consecutive frames (used during Green Light)."""
        current_blur = self.preprocess(frame)
        if self._prev_gray_blur is None:
            self._prev_gray_blur = current_blur
            return self._empty_result(
                gray_blur=current_blur,
                mode="consecutive",
                reference_ready=False,
            )

        result = self._analyze_difference(
            current_blur=current_blur,
            baseline_blur=self._prev_gray_blur,
            mode="consecutive",
            reference_ready=True,
        )
        self._prev_gray_blur = current_blur
        return result

    def detect_against_reference(self, frame: np.ndarray) -> MotionResult:
        """Detect motion by comparing against the locked Red Light reference frame."""
        current_blur = self.preprocess(frame)
        if self._reference_gray_blur is None:
            return self._empty_result(
                gray_blur=current_blur,
                mode="reference",
                reference_ready=False,
            )

        return self._analyze_difference(
            current_blur=current_blur,
            baseline_blur=self._reference_gray_blur,
            mode="reference",
            reference_ready=True,
        )

    def detect(self, frame: np.ndarray, use_reference: bool = False) -> MotionResult:
        """Unified entry point to run consecutive or reference-based detection."""
        if use_reference:
            return self.detect_against_reference(frame)
        return self.detect_consecutive(frame)
