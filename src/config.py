"""Configuration constants, colors, game timing, and PCD difficulty presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ASSETS_SOUNDS_DIR: Path = BASE_DIR / "assets" / "sounds"
CAPTURES_DIR: Path = BASE_DIR / "captures"

# ---------------------------------------------------------------------------
# Window & Camera Settings
# ---------------------------------------------------------------------------
WINDOW_NAME: str = "Red Light, Green Light - Pure PCD"
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 960
FRAME_HEIGHT: int = 540
TARGET_FPS: int = 30

# ---------------------------------------------------------------------------
# Game Timing & Rules
# ---------------------------------------------------------------------------
GAME_DURATION_SECONDS: float = 45.0
WARNING_DURATION_SECONDS: float = 1.0
RED_GRACE_PERIOD_SECONDS: float = 0.5
PROGRESS_MAX: float = 100.0

# ---------------------------------------------------------------------------
# Colors (OpenCV BGR Format)
# ---------------------------------------------------------------------------
COLOR_WHITE: Tuple[int, int, int] = (245, 247, 250)
COLOR_BLACK: Tuple[int, int, int] = (12, 14, 18)
COLOR_DARK_BG: Tuple[int, int, int] = (24, 27, 34)
COLOR_PANEL_BG: Tuple[int, int, int] = (32, 36, 46)
COLOR_BORDER_SUBTLE: Tuple[int, int, int] = (70, 78, 92)
COLOR_MUTED: Tuple[int, int, int] = (160, 168, 180)

COLOR_GREEN: Tuple[int, int, int] = (72, 214, 96)
COLOR_GREEN_DARK: Tuple[int, int, int] = (28, 84, 38)
COLOR_YELLOW: Tuple[int, int, int] = (40, 206, 255)
COLOR_YELLOW_DARK: Tuple[int, int, int] = (22, 86, 110)
COLOR_RED: Tuple[int, int, int] = (68, 68, 245)
COLOR_RED_DARK: Tuple[int, int, int] = (32, 32, 98)
COLOR_CYAN: Tuple[int, int, int] = (240, 205, 65)
COLOR_ORANGE: Tuple[int, int, int] = (45, 145, 255)

# ---------------------------------------------------------------------------
# Sound File Mapping (optional assets inside assets/sounds/)
# ---------------------------------------------------------------------------
SOUND_FILES: Dict[str, str] = {
    "green_light": "green_light.wav",
    "warning": "warning.wav",
    "red_light": "red_light.wav",
    "eliminated": "eliminated.wav",
    "win": "win.wav",
}


# ---------------------------------------------------------------------------
# PCD Difficulty Presets (Pure Classical Digital Image Processing)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DifficultyConfig:
    """PCD pipeline and gameplay parameters for a difficulty level."""

    name: str
    blur_kernel_size: Tuple[int, int]
    threshold_value: int
    morph_open_kernel_size: Tuple[int, int]
    morph_dilate_kernel_size: Tuple[int, int]
    morph_dilate_iterations: int
    min_contour_area: int
    red_motion_limit: float
    green_min_motion_score: float
    green_target_motion_score: float
    max_progress_per_sec: float
    green_duration_range: Tuple[float, float]
    red_duration_range: Tuple[float, float]


DIFFICULTY_PRESETS: Dict[str, DifficultyConfig] = {
    "Easy": DifficultyConfig(
        name="Easy",
        blur_kernel_size=(21, 21),
        threshold_value=35,
        morph_open_kernel_size=(7, 7),
        morph_dilate_kernel_size=(11, 11),
        morph_dilate_iterations=1,
        min_contour_area=1500,
        red_motion_limit=4000.0,
        green_min_motion_score=700.0,
        green_target_motion_score=16000.0,
        max_progress_per_sec=11.5,
        green_duration_range=(3.2, 5.5),
        red_duration_range=(2.0, 3.5),
    ),
    "Medium": DifficultyConfig(
        name="Medium",
        blur_kernel_size=(15, 15),
        threshold_value=25,
        morph_open_kernel_size=(5, 5),
        morph_dilate_kernel_size=(9, 9),
        morph_dilate_iterations=1,
        min_contour_area=800,
        red_motion_limit=2000.0,
        green_min_motion_score=900.0,
        green_target_motion_score=20000.0,
        max_progress_per_sec=10.0,
        green_duration_range=(2.5, 4.8),
        red_duration_range=(2.2, 4.2),
    ),
    "Hard": DifficultyConfig(
        name="Hard",
        blur_kernel_size=(11, 11),
        threshold_value=18,
        morph_open_kernel_size=(3, 3),
        morph_dilate_kernel_size=(7, 7),
        morph_dilate_iterations=1,
        min_contour_area=400,
        red_motion_limit=900.0,
        green_min_motion_score=1100.0,
        green_target_motion_score=24000.0,
        max_progress_per_sec=8.8,
        green_duration_range=(2.0, 4.0),
        red_duration_range=(2.6, 5.0),
    ),
}

DEFAULT_DIFFICULTY: str = "Medium"

DIFFICULTY_KEYS: Dict[int, str] = {
    ord("1"): "Easy",
    ord("2"): "Medium",
    ord("3"): "Hard",
}
