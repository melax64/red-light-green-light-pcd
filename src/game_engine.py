"""Game state machine, timer, progress logic, audio fallback, and evidence capture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    ASSETS_SOUNDS_DIR,
    CAPTURES_DIR,
    COLOR_RED,
    COLOR_WHITE,
    DEFAULT_DIFFICULTY,
    DIFFICULTY_PRESETS,
    DifficultyConfig,
    GAME_DURATION_SECONDS,
    PROGRESS_MAX,
    RED_GRACE_PERIOD_SECONDS,
    SOUND_FILES,
    WARNING_DURATION_SECONDS,
)
from src.motion_detector import BoundingBox, MotionDetector, MotionResult


class GameState(Enum):
    """Finite state machine states for Red Light, Green Light."""

    MENU = "MENU"
    GREEN_LIGHT = "GREEN_LIGHT"
    WARNING = "WARNING"
    RED_LIGHT = "RED_LIGHT"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class SoundManager:
    """Optional pygame-based sound effect player with 100% safe fallback."""

    def __init__(self, sounds_dir: Path = ASSETS_SOUNDS_DIR) -> None:
        self._sounds_dir = Path(sounds_dir)
        self._enabled = False
        self._sounds: Dict[str, object] = {}
        self._pygame = None
        self._init_mixer()

    @property
    def enabled(self) -> bool:
        """Whether pygame audio initialized successfully."""
        return self._enabled

    def _init_mixer(self) -> None:
        try:
            import pygame  # type: ignore

            pygame.mixer.init()
            self._pygame = pygame
            self._enabled = True
            self._load_sounds()
        except Exception:
            self._enabled = False
            self._sounds = {}

    def _load_sounds(self) -> None:
        if not self._enabled or self._pygame is None:
            return
        for key, filename in SOUND_FILES.items():
            path = self._sounds_dir / filename
            if path.is_file():
                try:
                    self._sounds[key] = self._pygame.mixer.Sound(str(path))
                except Exception:
                    continue

    def play(self, event_name: str) -> None:
        """Play a sound effect if available; silently no-op otherwise."""
        if not self._enabled:
            return
        sound = self._sounds.get(event_name)
        if sound is not None:
            try:
                sound.play()  # type: ignore[attr-defined]
            except Exception:
                pass

    def close(self) -> None:
        """Safely shut down pygame mixer if active."""
        if self._enabled and self._pygame is not None:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass
            self._enabled = False


@dataclass
class GameSnapshot:
    """Immutable view of the game state returned each frame for UI rendering."""

    state: GameState
    difficulty: DifficultyConfig
    progress: float
    time_remaining: float
    state_time_remaining: float
    in_grace_period: bool
    grace_remaining: float
    motion_result: MotionResult
    red_motion_limit: float
    green_target_motion_score: float
    game_over_reason: str
    last_capture_path: Optional[Path]
    elimination_boxes: List[BoundingBox]
    elimination_score: float


class GameEngine:
    """Core state machine coordinating timing, progress, PCD motion, and captures."""

    def __init__(
        self,
        difficulty_name: str = DEFAULT_DIFFICULTY,
        captures_dir: Path = CAPTURES_DIR,
        sound_manager: Optional[SoundManager] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if difficulty_name not in DIFFICULTY_PRESETS:
            difficulty_name = DEFAULT_DIFFICULTY

        self._difficulty: DifficultyConfig = DIFFICULTY_PRESETS[difficulty_name]
        self._detector = MotionDetector(self._difficulty)
        self._captures_dir = Path(captures_dir)
        self._sound = sound_manager if sound_manager is not None else SoundManager()
        self._rng = rng if rng is not None else random.Random()

        self._state: GameState = GameState.MENU
        self._progress: float = 0.0
        self._time_remaining: float = GAME_DURATION_SECONDS
        self._state_duration: float = 0.0
        self._state_start_time: float = 0.0
        self._last_update_time: Optional[float] = None
        self._game_over_reason: str = ""
        self._last_capture_path: Optional[Path] = None
        self._elimination_boxes: List[BoundingBox] = []
        self._elimination_score: float = 0.0

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def difficulty(self) -> DifficultyConfig:
        return self._difficulty

    @property
    def detector(self) -> MotionDetector:
        return self._detector

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def time_remaining(self) -> float:
        return self._time_remaining

    @property
    def last_capture_path(self) -> Optional[Path]:
        return self._last_capture_path

    def set_difficulty(self, difficulty_name: str) -> None:
        """Switch difficulty preset and propagate PCD thresholds immediately."""
        if difficulty_name not in DIFFICULTY_PRESETS:
            return
        self._difficulty = DIFFICULTY_PRESETS[difficulty_name]
        self._detector.update_config(self._difficulty)

    def start_game(self, now: Optional[float] = None) -> None:
        """Start or restart a new match from GREEN_LIGHT."""
        current_time = time.perf_counter() if now is None else now
        self._progress = 0.0
        self._time_remaining = GAME_DURATION_SECONDS
        self._game_over_reason = ""
        self._last_capture_path = None
        self._elimination_boxes = []
        self._elimination_score = 0.0
        self._detector.reset()
        self._last_update_time = current_time
        self._enter_state(GameState.GREEN_LIGHT, current_time)

    def _enter_state(self, new_state: GameState, now: float) -> None:
        self._state = new_state
        self._state_start_time = now

        if new_state == GameState.GREEN_LIGHT:
            low, high = self._difficulty.green_duration_range
            self._state_duration = self._rng.uniform(low, high)
            self._detector.reset_previous()
            self._sound.play("green_light")
        elif new_state == GameState.WARNING:
            self._state_duration = WARNING_DURATION_SECONDS
            self._sound.play("warning")
        elif new_state == GameState.RED_LIGHT:
            low, high = self._difficulty.red_duration_range
            self._state_duration = self._rng.uniform(low, high)
            self._detector.reset_reference()
            self._sound.play("red_light")
        elif new_state == GameState.WIN:
            self._state_duration = 0.0
            self._sound.play("win")
        elif new_state == GameState.GAME_OVER:
            self._state_duration = 0.0
            self._sound.play("eliminated")

    def _save_evidence_screenshot(
        self,
        frame: np.ndarray,
        motion_result: MotionResult,
        reason: str,
    ) -> Optional[Path]:
        """Save annotated elimination evidence frame to `captures/`."""
        try:
            self._captures_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"eliminated_{timestamp}.jpg"
            output_path = self._captures_dir / filename

            annotated = frame.copy()
            for x, y, w, h in motion_result.bounding_boxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_RED, 3)
                cv2.putText(
                    annotated,
                    "MOVED",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_RED,
                    2,
                    cv2.LINE_AA,
                )

            # Stamp evidence metadata banner at top
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 56), (18, 18, 28), -1)
            banner_text = (
                f"ELIMINATED | {reason} | "
                f"Score: {int(motion_result.motion_score)} "
                f"(Limit: {int(self._difficulty.red_motion_limit)}) | "
                f"{self._difficulty.name}"
            )
            cv2.putText(
                annotated,
                banner_text,
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )

            if cv2.imwrite(str(output_path), annotated):
                return output_path
        except Exception:
            return None
        return None

    def _trigger_game_over(
        self,
        frame: np.ndarray,
        motion_result: MotionResult,
        reason: str,
        now: float,
    ) -> None:
        self._game_over_reason = reason
        self._elimination_boxes = list(motion_result.bounding_boxes)
        self._elimination_score = motion_result.motion_score
        self._last_capture_path = self._save_evidence_screenshot(
            frame=frame,
            motion_result=motion_result,
            reason=reason,
        )
        self._enter_state(GameState.GAME_OVER, now)

    def _accumulate_progress(self, motion_score: float, dt: float) -> None:
        """Convert Green Light motion intensity into smooth 0-100% progress."""
        min_score = self._difficulty.green_min_motion_score
        target_score = max(min_score + 1.0, self._difficulty.green_target_motion_score)
        if motion_score <= min_score or dt <= 0.0:
            return

        normalized = min(1.0, (motion_score - min_score) / (target_score - min_score))
        gain = normalized * self._difficulty.max_progress_per_sec * dt
        self._progress = min(PROGRESS_MAX, self._progress + gain)

    def update(self, frame: np.ndarray, now: Optional[float] = None) -> GameSnapshot:
        """Advance state machine by one frame and return a UI snapshot."""
        current_time = time.perf_counter() if now is None else now
        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = max(0.0, min(0.25, current_time - self._last_update_time))
        self._last_update_time = current_time

        in_grace_period = False
        grace_remaining = 0.0

        # Non-active states: keep PCD pipeline alive for preview & debug mode
        if self._state in (GameState.MENU, GameState.WIN, GameState.GAME_OVER):
            motion_result = self._detector.detect_consecutive(frame)
            return self._build_snapshot(
                motion_result=motion_result,
                state_elapsed=0.0,
                in_grace_period=False,
                grace_remaining=0.0,
            )

        # Active gameplay states: tick countdown timer
        self._time_remaining = max(0.0, self._time_remaining - dt)
        state_elapsed = max(0.0, current_time - self._state_start_time)

        if self._state == GameState.GREEN_LIGHT:
            motion_result = self._detector.detect_consecutive(frame)
            if self._time_remaining <= 0.0:
                self._trigger_game_over(frame, motion_result, "TIME UP", current_time)
            else:
                self._accumulate_progress(motion_result.motion_score, dt)
                if self._progress >= PROGRESS_MAX:
                    self._progress = PROGRESS_MAX
                    self._enter_state(GameState.WIN, current_time)
                elif state_elapsed >= self._state_duration:
                    self._enter_state(GameState.WARNING, current_time)

        elif self._state == GameState.WARNING:
            motion_result = self._detector.detect_consecutive(frame)
            if self._time_remaining <= 0.0:
                self._trigger_game_over(frame, motion_result, "TIME UP", current_time)
            elif state_elapsed >= self._state_duration:
                self._enter_state(GameState.RED_LIGHT, current_time)

        elif self._state == GameState.RED_LIGHT:
            if state_elapsed < RED_GRACE_PERIOD_SECONDS:
                in_grace_period = True
                grace_remaining = max(0.0, RED_GRACE_PERIOD_SECONDS - state_elapsed)
                # Keep detector output valid while player freezes
                motion_result = self._detector.detect_against_reference(frame)
                if self._time_remaining <= 0.0:
                    self._trigger_game_over(frame, motion_result, "TIME UP", current_time)
            else:
                if not self._detector.has_reference:
                    self._detector.capture_reference_frame(frame)
                motion_result = self._detector.detect_against_reference(frame)

                if self._time_remaining <= 0.0:
                    self._trigger_game_over(frame, motion_result, "TIME UP", current_time)
                elif (
                    motion_result.is_motion_detected
                    and motion_result.motion_score >= self._difficulty.red_motion_limit
                ):
                    self._trigger_game_over(
                        frame,
                        motion_result,
                        "MOVED ON RED LIGHT",
                        current_time,
                    )
                elif state_elapsed >= self._state_duration:
                    self._enter_state(GameState.GREEN_LIGHT, current_time)
        else:
            motion_result = self._detector.detect_consecutive(frame)

        updated_elapsed = max(0.0, current_time - self._state_start_time)
        if self._state == GameState.RED_LIGHT and updated_elapsed < RED_GRACE_PERIOD_SECONDS:
            in_grace_period = True
            grace_remaining = max(0.0, RED_GRACE_PERIOD_SECONDS - updated_elapsed)

        return self._build_snapshot(
            motion_result=motion_result,
            state_elapsed=updated_elapsed,
            in_grace_period=in_grace_period,
            grace_remaining=grace_remaining,
        )

    def _build_snapshot(
        self,
        motion_result: MotionResult,
        state_elapsed: float,
        in_grace_period: bool,
        grace_remaining: float,
    ) -> GameSnapshot:
        state_remaining = max(0.0, self._state_duration - state_elapsed)
        return GameSnapshot(
            state=self._state,
            difficulty=self._difficulty,
            progress=self._progress,
            time_remaining=self._time_remaining,
            state_time_remaining=state_remaining,
            in_grace_period=in_grace_period,
            grace_remaining=grace_remaining,
            motion_result=motion_result,
            red_motion_limit=self._difficulty.red_motion_limit,
            green_target_motion_score=self._difficulty.green_target_motion_score,
            game_over_reason=self._game_over_reason,
            last_capture_path=self._last_capture_path,
            elimination_boxes=list(self._elimination_boxes),
            elimination_score=self._elimination_score,
        )

    def close(self) -> None:
        """Release resources owned by the engine."""
        self._sound.close()
