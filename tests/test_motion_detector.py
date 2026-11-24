"""Unit tests for the pure PCD MotionDetector, GameEngine, and UIRenderer."""

from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np

from src.config import (
    DIFFICULTY_PRESETS,
    RED_GRACE_PERIOD_SECONDS,
    WARNING_DURATION_SECONDS,
)
from src.game_engine import GameEngine, GameState, SoundManager
from src.motion_detector import MotionDetector
from src.ui_renderer import UIRenderer


def make_synthetic_frame(
    width: int = 640,
    height: int = 480,
    bg_value: int = 50,
    rect: tuple[int, int, int, int] | None = None,
    rect_value: int = 230,
) -> np.ndarray:
    """Create a deterministic BGR synthetic image with an optional bright rectangle."""
    frame = np.full((height, width, 3), bg_value, dtype=np.uint8)
    if rect is not None:
        x, y, w, h = rect
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (rect_value, rect_value, rect_value),
            -1,
        )
    return frame


class DummySoundManager(SoundManager):
    """Silent sound manager for deterministic headless testing."""

    def __init__(self) -> None:
        self._enabled = False
        self._sounds = {}
        self._pygame = None

    def play(self, event_name: str) -> None:  # noqa: ARG002
        return


class TestMotionDetector(unittest.TestCase):
    """Verify classical PCD pipeline behavior on synthetic NumPy images."""

    def setUp(self) -> None:
        self.config = DIFFICULTY_PRESETS["Medium"]
        self.detector = MotionDetector(self.config)

    def test_static_frames_no_motion_consecutive(self) -> None:
        frame1 = make_synthetic_frame(bg_value=60)
        frame2 = make_synthetic_frame(bg_value=60)

        res1 = self.detector.detect_consecutive(frame1)
        res2 = self.detector.detect_consecutive(frame2)

        self.assertFalse(res1.is_motion_detected)
        self.assertEqual(res1.motion_score, 0.0)
        self.assertEqual(res1.bounding_boxes, [])

        self.assertFalse(res2.is_motion_detected)
        self.assertEqual(res2.motion_score, 0.0)
        self.assertEqual(res2.bounding_boxes, [])

    def test_static_frames_no_motion_reference(self) -> None:
        ref_frame = make_synthetic_frame(bg_value=80, rect=(100, 100, 80, 80), rect_value=180)
        same_frame = ref_frame.copy()

        self.detector.capture_reference_frame(ref_frame)
        result = self.detector.detect_against_reference(same_frame)

        self.assertTrue(result.reference_ready)
        self.assertFalse(result.is_motion_detected)
        self.assertEqual(result.motion_score, 0.0)
        self.assertEqual(result.bounding_boxes, [])

    def test_motion_detected_with_accurate_bounding_box_in_reference_mode(self) -> None:
        target_x, target_y, target_w, target_h = 200, 140, 120, 100
        ref_frame = make_synthetic_frame(bg_value=40)
        moved_frame = make_synthetic_frame(
            bg_value=40,
            rect=(target_x, target_y, target_w, target_h),
            rect_value=240,
        )

        self.detector.capture_reference_frame(ref_frame)
        result = self.detector.detect_against_reference(moved_frame)

        self.assertTrue(result.is_motion_detected)
        self.assertGreater(result.motion_score, float(self.config.min_contour_area))
        self.assertEqual(len(result.bounding_boxes), 1)

        bx, by, bw, bh = result.bounding_boxes[0]
        # Bounding box should cover the synthetic object (allowing dilation padding)
        self.assertLessEqual(bx, target_x + 5)
        self.assertLessEqual(by, target_y + 5)
        self.assertGreaterEqual(bx + bw, target_x + target_w - 5)
        self.assertGreaterEqual(by + bh, target_y + target_h - 5)

        # Center of detected box should closely match synthetic object center
        detected_cx = bx + bw / 2.0
        detected_cy = by + bh / 2.0
        expected_cx = target_x + target_w / 2.0
        expected_cy = target_y + target_h / 2.0
        self.assertAlmostEqual(detected_cx, expected_cx, delta=15.0)
        self.assertAlmostEqual(detected_cy, expected_cy, delta=15.0)

    def test_consecutive_mode_detects_motion_and_returns_all_debug_stages(self) -> None:
        frame_a = make_synthetic_frame(bg_value=50)
        frame_b = make_synthetic_frame(bg_value=50, rect=(120, 120, 110, 90), rect_value=235)
        frame_c = frame_b.copy()

        _ = self.detector.detect_consecutive(frame_a)
        res_move = self.detector.detect_consecutive(frame_b)
        res_stop = self.detector.detect_consecutive(frame_c)

        self.assertTrue(res_move.is_motion_detected)
        self.assertGreater(res_move.motion_score, 0.0)
        self.assertGreaterEqual(len(res_move.bounding_boxes), 1)

        # Verify all intermediate PCD stage outputs exist with 2D uint8 shapes
        for stage_img in (
            res_move.gray_blur,
            res_move.frame_diff,
            res_move.thresh,
            res_move.morphed,
        ):
            self.assertEqual(stage_img.shape, (480, 640))
            self.assertEqual(stage_img.dtype, np.uint8)

        # When object stops moving, consecutive frame difference drops back to 0
        self.assertFalse(res_stop.is_motion_detected)
        self.assertEqual(res_stop.motion_score, 0.0)

    def test_small_light_fluctuation_and_tiny_noise_are_filtered_out(self) -> None:
        ref_frame = make_synthetic_frame(bg_value=60)
        self.detector.capture_reference_frame(ref_frame)

        # Slight lighting change (+6 intensity, well below threshold_value=25)
        slight_light_change = make_synthetic_frame(bg_value=66)
        res_light = self.detector.detect_against_reference(slight_light_change)
        self.assertFalse(res_light.is_motion_detected)
        self.assertEqual(res_light.motion_score, 0.0)

        # Tiny 4x4 noise speckle should be suppressed by GaussianBlur + Opening + min_contour_area
        tiny_noise = make_synthetic_frame(bg_value=60, rect=(300, 200, 4, 4), rect_value=255)
        res_noise = self.detector.detect_against_reference(tiny_noise)
        self.assertFalse(res_noise.is_motion_detected)
        self.assertEqual(res_noise.bounding_boxes, [])

    def test_difficulty_presets_sensitivity_ordering(self) -> None:
        easy = DIFFICULTY_PRESETS["Easy"]
        med = DIFFICULTY_PRESETS["Medium"]
        hard = DIFFICULTY_PRESETS["Hard"]

        self.assertGreater(easy.threshold_value, med.threshold_value)
        self.assertGreater(med.threshold_value, hard.threshold_value)
        self.assertGreater(easy.min_contour_area, med.min_contour_area)
        self.assertGreater(med.min_contour_area, hard.min_contour_area)
        self.assertGreater(easy.red_motion_limit, med.red_motion_limit)
        self.assertGreater(med.red_motion_limit, hard.red_motion_limit)


class TestGameEngineAndRenderer(unittest.TestCase):
    """Integration tests for state transitions, grace period, captures, and UI rendering."""

    def test_red_light_grace_period_elimination_and_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            captures_dir = Path(tmpdir)
            engine = GameEngine(
                difficulty_name="Medium",
                captures_dir=captures_dir,
                sound_manager=DummySoundManager(),
                rng=random.Random(42),
            )
            self.assertEqual(engine.state, GameState.MENU)

            still_frame = make_synthetic_frame(bg_value=45)
            move_frame_a = make_synthetic_frame(bg_value=45, rect=(80, 80, 180, 180), rect_value=240)
            move_frame_b = make_synthetic_frame(bg_value=45, rect=(320, 120, 180, 180), rect_value=240)

            t = 0.0
            engine.start_game(now=t)
            self.assertEqual(engine.state, GameState.GREEN_LIGHT)

            # Move during GREEN_LIGHT to accumulate progress until WARNING triggers
            for step in range(40):
                t += 0.2
                frame = move_frame_a if step % 2 == 0 else move_frame_b
                snap = engine.update(frame, now=t)
                if snap.state == GameState.WARNING:
                    break

            self.assertEqual(engine.state, GameState.WARNING)
            self.assertGreater(engine.progress, 0.0)

            # Advance past 1.0s WARNING into RED_LIGHT
            t += WARNING_DURATION_SECONDS + 0.05
            snap = engine.update(still_frame, now=t)
            self.assertEqual(engine.state, GameState.RED_LIGHT)
            self.assertTrue(snap.in_grace_period)

            # Movement during 0.5s grace period must NOT eliminate the player
            t += RED_GRACE_PERIOD_SECONDS * 0.5
            snap = engine.update(move_frame_a, now=t)
            self.assertEqual(engine.state, GameState.RED_LIGHT)
            self.assertTrue(snap.in_grace_period)

            # After grace period ends, reference frame locks on still_frame
            t += RED_GRACE_PERIOD_SECONDS
            snap = engine.update(still_frame, now=t)
            self.assertEqual(engine.state, GameState.RED_LIGHT)
            self.assertFalse(snap.in_grace_period)
            self.assertTrue(snap.motion_result.reference_ready)

            # Moving significantly after lock triggers ELIMINATED + saves screenshot
            t += 0.1
            snap = engine.update(move_frame_b, now=t)
            self.assertEqual(engine.state, GameState.GAME_OVER)
            self.assertEqual(snap.game_over_reason, "MOVED ON RED LIGHT")
            self.assertIsNotNone(snap.last_capture_path)
            self.assertTrue(snap.last_capture_path.is_file())
            self.assertGreaterEqual(len(snap.elimination_boxes), 1)

            # Verify both Normal HUD and 4-panel Debug Mode render cleanly
            renderer = UIRenderer()
            normal_out = renderer.render(move_frame_b, snap, debug_mode=False)
            debug_out = renderer.render(move_frame_b, snap, debug_mode=True)
            self.assertEqual(normal_out.shape, move_frame_b.shape)
            self.assertEqual(debug_out.shape, move_frame_b.shape)


if __name__ == "__main__":
    unittest.main()
