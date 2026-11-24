"""Modern OpenCV HUD renderer and 4-panel PCD Debug Mode visualizer."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.config import (
    COLOR_BLACK,
    COLOR_BORDER_SUBTLE,
    COLOR_CYAN,
    COLOR_DARK_BG,
    COLOR_GREEN,
    COLOR_GREEN_DARK,
    COLOR_MUTED,
    COLOR_ORANGE,
    COLOR_PANEL_BG,
    COLOR_RED,
    COLOR_RED_DARK,
    COLOR_WHITE,
    COLOR_YELLOW,
    COLOR_YELLOW_DARK,
)
from src.game_engine import GameSnapshot, GameState
from src.motion_detector import BoundingBox


class UIRenderer:
    """Renders HUD overlays and real-time 4-panel PCD debug splits onto frames."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, snapshot: GameSnapshot, debug_mode: bool = False) -> np.ndarray:
        """Render either the standard HUD view or the 4-panel PCD Debug view."""
        hud_frame = self.render_game_frame(frame, snapshot)
        if not debug_mode:
            return hud_frame
        return self.render_debug_view(hud_frame, snapshot)

    def render_game_frame(self, frame: np.ndarray, snapshot: GameSnapshot) -> np.ndarray:
        """Draw bounding boxes, traffic light, timer, motion meter, progress bar, and overlays."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw motion bounding boxes based on active state
        if snapshot.state == GameState.RED_LIGHT and not snapshot.in_grace_period:
            self._draw_motion_boxes(
                canvas,
                snapshot.motion_result.bounding_boxes,
                color=COLOR_RED,
                label="MOTION!",
                thickness=3,
            )
        elif snapshot.state == GameState.GAME_OVER and snapshot.elimination_boxes:
            self._draw_motion_boxes(
                canvas,
                snapshot.elimination_boxes,
                color=COLOR_RED,
                label="DETECTED",
                thickness=3,
            )
        elif snapshot.state == GameState.GREEN_LIGHT:
            self._draw_motion_boxes(
                canvas,
                snapshot.motion_result.bounding_boxes,
                color=COLOR_GREEN,
                label="MOVE",
                thickness=2,
            )
        elif snapshot.state == GameState.WARNING:
            self._draw_motion_boxes(
                canvas,
                snapshot.motion_result.bounding_boxes,
                color=COLOR_YELLOW,
                label="SLOW DOWN",
                thickness=2,
            )

        # 2. State-colored outer border glow
        border_color = self._get_state_color(snapshot.state)
        border_thickness = 8 if snapshot.state in (GameState.RED_LIGHT, GameState.GAME_OVER) else 5
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border_color, border_thickness)

        # 3. Top HUD Bar
        self._draw_top_hud(canvas, snapshot)

        # 4. Grace Period banner on RED_LIGHT
        if snapshot.state == GameState.RED_LIGHT and snapshot.in_grace_period:
            self._draw_center_pill(
                canvas,
                f"FREEZE! LOCKING FRAME ({snapshot.grace_remaining:.1f}s)",
                y_center=108,
                bg_color=COLOR_ORANGE,
                text_color=COLOR_BLACK,
            )

        # 5. Bottom Progress Bar & Controls Footer
        self._draw_bottom_hud(canvas, snapshot)

        # 6. Fullscreen state overlays (MENU / WIN / GAME_OVER)
        if snapshot.state == GameState.MENU:
            self._draw_menu_overlay(canvas, snapshot)
        elif snapshot.state == GameState.WIN:
            self._draw_end_overlay(
                canvas,
                title="YOU WIN!",
                subtitle=f"Finished with {snapshot.time_remaining:.1f}s remaining | Difficulty: {snapshot.difficulty.name}",
                accent_color=COLOR_GREEN,
                extra_line="Great control! Press SPACE to play again.",
            )
        elif snapshot.state == GameState.GAME_OVER:
            capture_note = (
                f"Evidence saved: captures/{snapshot.last_capture_path.name}"
                if snapshot.last_capture_path is not None
                else "Press SPACE to try again."
            )
            detail = (
                f"Reason: {snapshot.game_over_reason} | "
                f"Motion Score: {int(snapshot.elimination_score)} / {int(snapshot.red_motion_limit)}"
                if snapshot.game_over_reason == "MOVED ON RED LIGHT"
                else f"Reason: {snapshot.game_over_reason} | Progress: {snapshot.progress:.1f}%"
            )
            self._draw_end_overlay(
                canvas,
                title="ELIMINATED",
                subtitle=detail,
                accent_color=COLOR_RED,
                extra_line=f"{capture_note}  |  Press SPACE to Restart",
            )

        return canvas

    def render_debug_view(self, hud_frame: np.ndarray, snapshot: GameSnapshot) -> np.ndarray:
        """Build 2x2 real-time PCD Debug split-view."""
        h, w = hud_frame.shape[:2]
        half_w = w // 2
        half_h = h // 2
        mr = snapshot.motion_result

        # Convert single-channel PCD stages to 3-channel BGR for clean concatenation
        panel_1 = cv2.resize(hud_frame, (half_w, half_h), interpolation=cv2.INTER_AREA)

        gray_bgr = cv2.cvtColor(mr.gray_blur, cv2.COLOR_GRAY2BGR)
        panel_2 = cv2.resize(gray_bgr, (half_w, half_h), interpolation=cv2.INTER_AREA)

        diff_bgr = cv2.cvtColor(mr.frame_diff, cv2.COLOR_GRAY2BGR)
        panel_3 = cv2.resize(diff_bgr, (half_w, half_h), interpolation=cv2.INTER_AREA)

        morph_bgr = cv2.cvtColor(mr.morphed, cv2.COLOR_GRAY2BGR)
        panel_4 = cv2.resize(morph_bgr, (half_w, half_h), interpolation=cv2.INTER_AREA)

        # Label banners on each quadrant
        diff_mode_txt = "Ref Frame" if mr.mode == "reference" else "Consecutive"
        cfg = snapshot.difficulty

        self._draw_debug_label(
            panel_1,
            "1. Game Frame + HUD + BBox",
            f"State: {snapshot.state.value}",
            COLOR_GREEN,
        )
        self._draw_debug_label(
            panel_2,
            "2. Grayscale + Gaussian Blur",
            f"Kernel: {cfg.blur_kernel_size[0]}x{cfg.blur_kernel_size[1]}",
            COLOR_CYAN,
        )
        self._draw_debug_label(
            panel_3,
            "3. Frame Differencing (absdiff)",
            f"Mode: {diff_mode_txt}",
            COLOR_YELLOW,
        )
        self._draw_debug_label(
            panel_4,
            "4. Threshold + Morphology",
            f"Thr:{cfg.threshold_value} | Contours:{len(mr.bounding_boxes)} | Score:{int(mr.motion_score)}",
            COLOR_RED if snapshot.state == GameState.RED_LIGHT else COLOR_ORANGE,
        )

        top_row = np.hstack([panel_1, panel_2])
        bottom_row = np.hstack([panel_3, panel_4])
        grid = np.vstack([top_row, bottom_row])

        # Draw crisp crosshairs dividing the 4 panels
        gh, gw = grid.shape[:2]
        cv2.line(grid, (gw // 2, 0), (gw // 2, gh), COLOR_WHITE, 2)
        cv2.line(grid, (0, gh // 2), (gw, gh // 2), COLOR_WHITE, 2)
        return grid

    # ------------------------------------------------------------------
    # Internal Drawing Helpers
    # ------------------------------------------------------------------
    def _get_state_color(self, state: GameState) -> Tuple[int, int, int]:
        if state == GameState.GREEN_LIGHT:
            return COLOR_GREEN
        if state == GameState.WARNING:
            return COLOR_YELLOW
        if state in (GameState.RED_LIGHT, GameState.GAME_OVER):
            return COLOR_RED
        if state == GameState.WIN:
            return COLOR_CYAN
        return COLOR_BORDER_SUBTLE

    def _draw_alpha_rect(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.72,
    ) -> None:
        h, w = canvas.shape[:2]
        x1 = max(0, min(w, pt1[0]))
        y1 = max(0, min(h, pt1[1]))
        x2 = max(0, min(w, pt2[0]))
        y2 = max(0, min(h, pt2[1]))
        if x2 <= x1 or y2 <= y1:
            return
        roi = canvas[y1:y2, x1:x2]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    def _draw_motion_boxes(
        self,
        canvas: np.ndarray,
        boxes: List[BoundingBox],
        color: Tuple[int, int, int],
        label: str,
        thickness: int = 2,
    ) -> None:
        for idx, (x, y, bw, bh) in enumerate(boxes):
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, thickness)
            # Corner accents for modern HUD feel
            corner = min(18, bw // 3, bh // 3)
            if corner >= 5:
                t = thickness + 1
                cv2.line(canvas, (x, y), (x + corner, y), COLOR_WHITE, t)
                cv2.line(canvas, (x, y), (x, y + corner), COLOR_WHITE, t)
                cv2.line(canvas, (x + bw, y), (x + bw - corner, y), COLOR_WHITE, t)
                cv2.line(canvas, (x + bw, y), (x + bw, y + corner), COLOR_WHITE, t)
                cv2.line(canvas, (x, y + bh), (x + corner, y + bh), COLOR_WHITE, t)
                cv2.line(canvas, (x, y + bh), (x, y + bh - corner), COLOR_WHITE, t)
                cv2.line(canvas, (x + bw, y + bh), (x + bw - corner, y + bh), COLOR_WHITE, t)
                cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner), COLOR_WHITE, t)

            if idx < 3:
                tag = f"{label}"
                (tw, th), _ = cv2.getTextSize(tag, self.FONT, 0.5, 1)
                tag_y = max(th + 8, y - 6)
                cv2.rectangle(
                    canvas,
                    (x, tag_y - th - 6),
                    (x + tw + 10, tag_y + 2),
                    color,
                    -1,
                )
                cv2.putText(
                    canvas,
                    tag,
                    (x + 5, tag_y - 2),
                    self.FONT,
                    0.5,
                    COLOR_BLACK,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_top_hud(self, canvas: np.ndarray, snapshot: GameSnapshot) -> None:
        _, w = canvas.shape[:2]
        bar_h = 74
        self._draw_alpha_rect(canvas, (0, 0), (w, bar_h), COLOR_DARK_BG, alpha=0.80)
        cv2.line(canvas, (0, bar_h), (w, bar_h), COLOR_BORDER_SUBTLE, 1)

        # --- Left: Traffic Light Bulbs + State Label ---
        bulb_y = 37
        bulb_r = 14
        bulbs = [
            (36, GameState.RED_LIGHT, COLOR_RED, COLOR_RED_DARK),
            (74, GameState.WARNING, COLOR_YELLOW, COLOR_YELLOW_DARK),
            (112, GameState.GREEN_LIGHT, COLOR_GREEN, COLOR_GREEN_DARK),
        ]
        for bx, target_state, active_col, dim_col in bulbs:
            is_active = snapshot.state == target_state or (
                target_state == GameState.RED_LIGHT and snapshot.state == GameState.GAME_OVER
            )
            if is_active:
                cv2.circle(canvas, (bx, bulb_y), bulb_r + 4, active_col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (bx, bulb_y), bulb_r, active_col, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (bx, bulb_y), bulb_r, dim_col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (bx, bulb_y), bulb_r, COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

        state_titles = {
            GameState.MENU: ("READY", "Press SPACE to start"),
            GameState.GREEN_LIGHT: ("GREEN LIGHT", "MOVE NOW!"),
            GameState.WARNING: ("WARNING", "PREPARE TO FREEZE!"),
            GameState.RED_LIGHT: (
                "RED LIGHT",
                "LOCKING..." if snapshot.in_grace_period else "DO NOT MOVE!",
            ),
            GameState.WIN: ("VICTORY", "Goal reached!"),
            GameState.GAME_OVER: ("ELIMINATED", snapshot.game_over_reason or "Game Over"),
        }
        title_txt, sub_txt = state_titles[snapshot.state]
        state_col = self._get_state_color(snapshot.state)
        cv2.putText(canvas, title_txt, (142, 34), self.FONT, 0.80, state_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, sub_txt, (142, 57), self.FONT, 0.48, COLOR_MUTED, 1, cv2.LINE_AA)

        # --- Center: Timer + Difficulty Pill ---
        timer_x = w // 2 - 75
        timer_color = COLOR_RED if snapshot.time_remaining <= 10.0 else COLOR_WHITE
        cv2.putText(
            canvas,
            f"TIME {snapshot.time_remaining:04.1f}s",
            (timer_x, 34),
            self.FONT,
            0.78,
            timer_color,
            2,
            cv2.LINE_AA,
        )
        diff_label = f"MODE: {snapshot.difficulty.name.upper()}"
        cv2.putText(
            canvas,
            diff_label,
            (timer_x + 12, 57),
            self.FONT,
            0.48,
            COLOR_CYAN,
            1,
            cv2.LINE_AA,
        )

        # --- Right: Live Motion Meter ---
        meter_w = 220
        meter_h = 16
        meter_x = w - meter_w - 22
        meter_y = 38

        score = snapshot.motion_result.motion_score
        if snapshot.state == GameState.RED_LIGHT:
            meter_max = max(1.0, snapshot.red_motion_limit * 1.5)
            limit_ratio = snapshot.red_motion_limit / meter_max
            fill_color = COLOR_RED if score >= snapshot.red_motion_limit * 0.7 else COLOR_YELLOW
            meter_title = f"MOTION: {int(score)} / {int(snapshot.red_motion_limit)}"
        else:
            meter_max = max(1.0, snapshot.green_target_motion_score)
            limit_ratio = None
            fill_color = COLOR_GREEN
            meter_title = f"MOTION: {int(score)}"

        ratio = min(1.0, max(0.0, score / meter_max))
        cv2.putText(
            canvas,
            meter_title,
            (meter_x, 27),
            self.FONT,
            0.48,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h),
            COLOR_PANEL_BG,
            -1,
        )
        fill_w = int(meter_w * ratio)
        if fill_w > 0:
            cv2.rectangle(
                canvas,
                (meter_x, meter_y),
                (meter_x + fill_w, meter_y + meter_h),
                fill_color,
                -1,
            )
        if limit_ratio is not None:
            lim_x = meter_x + int(meter_w * limit_ratio)
            cv2.line(canvas, (lim_x, meter_y - 3), (lim_x, meter_y + meter_h + 3), COLOR_RED, 2)
        cv2.rectangle(
            canvas,
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h),
            COLOR_BORDER_SUBTLE,
            1,
        )

    def _draw_bottom_hud(self, canvas: np.ndarray, snapshot: GameSnapshot) -> None:
        h, w = canvas.shape[:2]
        bar_h = 74
        top_y = h - bar_h
        self._draw_alpha_rect(canvas, (0, top_y), (w, h), COLOR_DARK_BG, alpha=0.82)
        cv2.line(canvas, (0, top_y), (w, top_y), COLOR_BORDER_SUBTLE, 1)

        # Progress bar
        pad_x = 24
        pb_x1 = pad_x
        pb_x2 = w - pad_x
        pb_y1 = top_y + 14
        pb_y2 = top_y + 38
        pb_w = pb_x2 - pb_x1

        cv2.rectangle(canvas, (pb_x1, pb_y1), (pb_x2, pb_y2), COLOR_PANEL_BG, -1)
        progress_ratio = min(1.0, max(0.0, snapshot.progress / 100.0))
        fill_w = int(pb_w * progress_ratio)
        bar_color = COLOR_CYAN if snapshot.state == GameState.WIN else COLOR_GREEN
        if fill_w > 0:
            cv2.rectangle(canvas, (pb_x1, pb_y1), (pb_x1 + fill_w, pb_y2), bar_color, -1)
        cv2.rectangle(canvas, (pb_x1, pb_y1), (pb_x2, pb_y2), COLOR_WHITE, 1)

        prog_text = f"PROGRESS: {snapshot.progress:5.1f}%"
        (tw, _), _ = cv2.getTextSize(prog_text, self.FONT, 0.56, 2)
        cv2.putText(
            canvas,
            prog_text,
            ((w - tw) // 2, pb_y2 - 5),
            self.FONT,
            0.56,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        # Keyboard controls legend
        controls_txt = "[SPACE] Start/Restart   |   [D] Debug PCD 4-Panel   |   [1/2/3] Easy/Medium/Hard   |   [Q/ESC] Quit"
        (cw, _), _ = cv2.getTextSize(controls_txt, self.FONT, 0.45, 1)
        cv2.putText(
            canvas,
            controls_txt,
            (max(12, (w - cw) // 2), h - 13),
            self.FONT,
            0.45,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

    def _draw_center_pill(
        self,
        canvas: np.ndarray,
        text: str,
        y_center: int,
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
    ) -> None:
        _, w = canvas.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, self.FONT, 0.65, 2)
        pad_x, pad_y = 20, 10
        x1 = (w - tw) // 2 - pad_x
        y1 = y_center - th // 2 - pad_y
        x2 = (w + tw) // 2 + pad_x
        y2 = y_center + th // 2 + pad_y
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_WHITE, 1)
        cv2.putText(
            canvas,
            text,
            ((w - tw) // 2, y_center + th // 2 - 2),
            self.FONT,
            0.65,
            text_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_menu_overlay(self, canvas: np.ndarray, snapshot: GameSnapshot) -> None:
        h, w = canvas.shape[:2]
        card_w, card_h = 620, 270
        x1 = (w - card_w) // 2
        y1 = (h - card_h) // 2
        x2 = x1 + card_w
        y2 = y1 + card_h

        self._draw_alpha_rect(canvas, (x1, y1), (x2, y2), COLOR_DARK_BG, alpha=0.88)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_CYAN, 2)

        self._draw_centered_text(canvas, "RED LIGHT, GREEN LIGHT", y1 + 50, 1.05, COLOR_WHITE, 3)
        self._draw_centered_text(
            canvas,
            "Pure Digital Image Processing (No Machine Learning)",
            y1 + 82,
            0.55,
            COLOR_CYAN,
            1,
        )
        self._draw_centered_text(
            canvas,
            "GREEN LIGHT: Keep moving until Progress reaches 100% (45s)",
            y1 + 124,
            0.52,
            COLOR_GREEN,
            1,
        )
        self._draw_centered_text(
            canvas,
            "RED LIGHT: Freeze completely after the 0.5s grace period!",
            y1 + 152,
            0.52,
            COLOR_RED,
            1,
        )

        # Difficulty badges
        diff_y = y1 + 195
        options = [("1", "Easy"), ("2", "Medium"), ("3", "Hard")]
        start_x = w // 2 - 210
        for idx, (key_lbl, name) in enumerate(options):
            bx1 = start_x + idx * 145
            bx2 = bx1 + 130
            selected = snapshot.difficulty.name == name
            bg = COLOR_CYAN if selected else COLOR_PANEL_BG
            fg = COLOR_BLACK if selected else COLOR_WHITE
            cv2.rectangle(canvas, (bx1, diff_y - 22), (bx2, diff_y + 10), bg, -1)
            cv2.rectangle(canvas, (bx1, diff_y - 22), (bx2, diff_y + 10), COLOR_WHITE, 1)
            lbl = f"[{key_lbl}] {name}"
            (lw, _), _ = cv2.getTextSize(lbl, self.FONT, 0.50, 1)
            cv2.putText(
                canvas,
                lbl,
                (bx1 + (130 - lw) // 2, diff_y - 1),
                self.FONT,
                0.50,
                fg,
                2 if selected else 1,
                cv2.LINE_AA,
            )

        self._draw_centered_text(
            canvas,
            "Press [SPACE] to Start  |  [D] Toggle Debug PCD",
            y2 - 24,
            0.60,
            COLOR_YELLOW,
            2,
        )

    def _draw_end_overlay(
        self,
        canvas: np.ndarray,
        title: str,
        subtitle: str,
        accent_color: Tuple[int, int, int],
        extra_line: str,
    ) -> None:
        h, w = canvas.shape[:2]
        card_w, card_h = 660, 175
        x1 = (w - card_w) // 2
        y1 = (h - card_h) // 2
        x2 = x1 + card_w
        y2 = y1 + card_h

        self._draw_alpha_rect(canvas, (x1, y1), (x2, y2), COLOR_DARK_BG, alpha=0.84)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), accent_color, 3)

        self._draw_centered_text(canvas, title, y1 + 58, 1.35, accent_color, 3)
        self._draw_centered_text(canvas, subtitle, y1 + 104, 0.56, COLOR_WHITE, 1)
        self._draw_centered_text(canvas, extra_line, y1 + 142, 0.50, COLOR_YELLOW, 1)

    def _draw_centered_text(
        self,
        canvas: np.ndarray,
        text: str,
        y: int,
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        _, w = canvas.shape[:2]
        (tw, _), _ = cv2.getTextSize(text, self.FONT, scale, thickness)
        cv2.putText(
            canvas,
            text,
            ((w - tw) // 2, y),
            self.FONT,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_debug_label(
        self,
        panel: np.ndarray,
        title: str,
        meta: str,
        accent_color: Tuple[int, int, int],
    ) -> None:
        _, pw = panel.shape[:2]
        self._draw_alpha_rect(panel, (0, 0), (pw, 30), COLOR_BLACK, alpha=0.75)
        cv2.rectangle(panel, (0, 0), (5, 30), accent_color, -1)
        cv2.putText(panel, title, (12, 20), self.FONT, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)
        (mw, _), _ = cv2.getTextSize(meta, self.FONT, 0.44, 1)
        cv2.putText(
            panel,
            meta,
            (max(12, pw - mw - 10), 20),
            self.FONT,
            0.44,
            accent_color,
            1,
            cv2.LINE_AA,
        )
