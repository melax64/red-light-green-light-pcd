"""Main entry point for Red Light, Green Light (Pure PCD Webcam Game)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

MAX_CONSECUTIVE_READ_FAILURES = 30

import cv2

# Allow running both `python -m src.main` and `python src/main.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CAMERA_INDEX,
    DEFAULT_DIFFICULTY,
    DIFFICULTY_KEYS,
    DIFFICULTY_PRESETS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_NAME,
)
from src.game_engine import GameEngine  # noqa: E402
from src.ui_renderer import UIRenderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse optional CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Red Light, Green Light - Pure Digital Image Processing (No ML)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Webcam device index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--difficulty",
        choices=list(DIFFICULTY_PRESETS.keys()),
        default=DEFAULT_DIFFICULTY,
        help=f"Initial difficulty preset (default: {DEFAULT_DIFFICULTY})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start directly in 4-panel PCD Debug Mode",
    )
    return parser.parse_args()


def _candidate_backends() -> list[int]:
    """Backend order: on Windows prefer DirectShow (more stable than MSMF)."""
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Try several capture backends and return the first that delivers real frames."""
    cap = cv2.VideoCapture()
    for backend in _candidate_backends():
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Warm-up: some webcams need a few reads before delivering frames
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Webcam terbuka dengan backend: {cap.getBackendName()}")
                return cap
            time.sleep(0.1)

        cap.release()
    return cap


def main() -> int:
    """Run the webcam capture, PCD processing, and interactive game loop."""
    args = parse_args()
    cap = open_camera(args.camera)

    if not cap.isOpened():
        print(
            f"[ERROR] Tidak dapat membuka webcam pada indeks {args.camera}. "
            "Pastikan kamera terhubung dan tidak sedang digunakan aplikasi lain.",
            file=sys.stderr,
        )
        return 1

    engine = GameEngine(difficulty_name=args.difficulty)
    renderer = UIRenderer()
    debug_mode = bool(args.debug)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    read_failures = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Tolerate transient glitches instead of quitting on the first bad frame
                read_failures += 1
                if read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    print(
                        "[WARNING] Gagal membaca frame dari webcam berulang kali. "
                        "Pastikan kamera tidak dipakai aplikasi lain (Zoom, Teams, browser, dll).",
                        file=sys.stderr,
                    )
                    break
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                continue
            read_failures = 0

            # Mirror frame horizontally for intuitive player movement
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(
                    frame,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

            snapshot = engine.update(frame)
            display_frame = renderer.render(frame, snapshot, debug_mode=debug_mode)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            if key == 32:  # SPACE -> Start / Restart
                engine.start_game()
            elif key in (ord("d"), ord("D")):  # Toggle 4-panel PCD Debug Mode
                debug_mode = not debug_mode
            elif key in DIFFICULTY_KEYS:  # 1 / 2 / 3 -> Easy / Medium / Hard
                engine.set_difficulty(DIFFICULTY_KEYS[key])

            # Allow closing cleanly via the window X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
