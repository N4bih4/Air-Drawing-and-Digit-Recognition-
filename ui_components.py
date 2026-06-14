"""
Professional dark-theme UI: toolbar, prediction panel, toasts, hit testing.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Theme (BGR) ─────────────────────────────────────────────────────────────
C_BLACK = (0, 0, 0)
C_PANEL = (22, 22, 28)
C_PURPLE = (235, 95, 215)
C_PURPLE_DIM = (160, 60, 140)
C_BORDER = (210, 75, 190)
C_TEXT = (245, 245, 250)
C_TEXT_DIM = (175, 175, 185)
C_GREEN = (90, 230, 90)
C_BLUE_DOT = (255, 180, 80)
C_YELLOW = (0, 230, 255)
C_RED_KEY = (90, 90, 255)
C_YELLOW_KEY = (0, 230, 255)
C_ACCENT = (255, 180, 60)

PALETTE = [
    (0, 0, 255),
    (0, 140, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 200),
    (180, 105, 255),
    (255, 255, 255),
    (0, 255, 128),
    (42, 82, 165),
    (255, 128, 0),
]

SWATCH_COLS = 6
SWATCH_R = 13
SWATCH_HIT_EXTRA = 18

TOOL_KEYS = ["pen", "eraser", "clear", "save", "predict", "undo"]


def pen_color_from_index(index: int):
    return PALETTE[index]


@dataclass
class Toast:
    message: str
    color: Tuple[int, int, int]
    until: float
    kind: str = "info"


@dataclass
class PredictionDisplay:
    digit: int = -1
    confidence: float = 0.0
    preview_28: Optional[np.ndarray] = None
    visible: bool = False
    is_digit: bool = True
    note: str = ""


@dataclass
class Layout:
    w: int = 1280
    h: int = 720
    header_h: int = 158
    footer_h: int = 50
    canvas_margin: int = 6
    canvas_top: int = 160
    canvas_bottom: int = 670
    p_colors: Tuple[int, int, int, int] = (10, 10, 400, 148)
    p_tools: Tuple[int, int, int, int] = (410, 10, 720, 148)
    p_brush: Tuple[int, int, int, int] = (730, 10, 980, 148)
    p_predict: Tuple[int, int, int, int] = (990, 10, 1270, 148)


def get_work_area_size() -> Tuple[int, int]:
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0)
        return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) - 48
        except Exception:
            return 1280, 720


def build_layout(screen_w: int, screen_h: int) -> Layout:
    lay = Layout()
    lay.w = max(1024, screen_w)
    lay.h = max(600, screen_h)
    lay.footer_h = max(40, int(lay.h * 0.062))
    lay.header_h = max(128, int(lay.h * 0.152))
    lay.canvas_margin = max(4, int(lay.w * 0.005))
    lay.canvas_top = lay.header_h + 2
    lay.canvas_bottom = lay.h - lay.footer_h - 2

    margin = max(10, int(lay.w * 0.012))
    gap = max(8, int(lay.w * 0.006))
    py1, py2 = 8, lay.header_h - 6
    usable = lay.w - 2 * margin - 3 * gap

    w_colors = int(usable * 0.32)
    w_tools = int(usable * 0.28)
    w_brush = int(usable * 0.22)
    w_pred = usable - w_colors - w_tools - w_brush

    x = margin
    lay.p_colors = (x, py1, x + w_colors, py2)
    x += w_colors + gap
    lay.p_tools = (x, py1, x + w_tools, py2)
    x += w_tools + gap
    lay.p_brush = (x, py1, x + w_brush, py2)
    x += w_brush + gap
    lay.p_predict = (x, py1, lay.w - margin, py2)
    return lay


class UIRenderer:
    """Draws toolbar, status bar, prediction panel, toasts."""

    def __init__(self, layout: Layout):
        self.L = layout
        self.toasts: List[Toast] = []
        self.prediction = PredictionDisplay()
        self.history_digits: List[str] = []

    def add_toast(self, message: str, color=C_GREEN, duration: float = 2.0, kind: str = "info"):
        self.toasts.append(Toast(message, color, time.time() + duration, kind))

    def set_prediction(self, digit: int, confidence: float, preview_28: np.ndarray, is_digit: bool = True, note: str = ""):
        self.prediction = PredictionDisplay(digit, confidence, preview_28, True, is_digit, note)
        if is_digit and digit >= 0:
            self.history_digits.append(str(digit))
            self.history_digits = self.history_digits[-8:]

    def set_rejected(self, preview_28: np.ndarray, note: str):
        self.prediction = PredictionDisplay(-1, 0.0, preview_28, True, False, note)

    def clear_prediction_display(self):
        self.prediction.visible = False

    # ── drawing helpers ──
    @staticmethod
    def _rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness < 0:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            for cx, cy in [
                (x1 + radius, y1 + radius),
                (x2 - radius, y1 + radius),
                (x1 + radius, y2 - radius),
                (x2 - radius, y2 - radius),
            ]:
                cv2.circle(img, (cx, cy), radius, color, -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def _label(img, text, pos, scale=0.45, color=C_TEXT, thickness=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _panel(self, img, rect, title=None):
        x1, y1, x2, y2 = rect
        self._rounded_rect(img, (x1, y1), (x2, y2), C_PANEL, -1, 6)
        cv2.rectangle(img, (x1, y1), (x2, y2), C_PURPLE_DIM, 1)
        if title:
            self._label(img, title, (x1 + 10, y1 + 20), 0.48, C_PURPLE)

    def _content_band(self, rect, title_offset=34):
        x1, y1, x2, y2 = rect
        return y1 + title_offset, y2 - 8

    def _swatch_centers(self):
        x1, y1, x2, y2 = self.L.p_colors
        pad = max(12, int((x2 - x1) * 0.05))
        inner_w = x2 - x1 - 2 * pad
        top, bottom = self._content_band(self.L.p_colors)
        mid_y = (top + bottom) // 2
        step = inner_w / (SWATCH_COLS - 1)
        row1 = [(int(x1 + pad + i * step), mid_y - 20) for i in range(SWATCH_COLS)]
        row2 = [(int(x1 + pad + i * step), mid_y + 20) for i in range(SWATCH_COLS)]
        return row1 + row2

    def _tool_rects(self):
        x1, y1, x2, y2 = self.L.p_tools
        pad = max(4, int((x2 - x1) * 0.02))
        n = len(TOOL_KEYS)
        inner_w = x2 - x1 - 2 * pad
        btn_w = inner_w // n
        top, bottom = self._content_band(self.L.p_tools, title_offset=8)
        return [
            (x1 + pad + i * btn_w, top, x1 + pad + i * btn_w + btn_w - 2, bottom)
            for i in range(n)
        ]

    def _brush_mid_y(self):
        top, bottom = self._content_band(self.L.p_brush)
        return (top + bottom) // 2

    def _draw_rainbow(self, img, cx, cy, r):
        hues = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        for i, col in enumerate(hues):
            cv2.ellipse(img, (cx, cy), (r, r), 0, int(i * 72), int((i + 1) * 72), col, -1, cv2.LINE_AA)

    def _draw_test_overlay(self, img, test_mode, layout: Layout):
        L = layout
        cx = L.w // 2
        cy = L.canvas_top + 36
        lines = ["TEST MODE"]
        if test_mode.target_digit >= 0:
            lines.append(f"Draw digit: {test_mode.target_digit}")
        else:
            lines.append("Press 0-9 to pick target")
        if test_mode.total > 0:
            lines.append(
                f"Accuracy: {test_mode.accuracy_pct:.1f}% ({test_mode.correct}/{test_mode.total})"
            )
        if test_mode.last_result:
            pred = test_mode.last_predicted
            if test_mode.last_result == "CORRECT":
                lines.append(f"Last: CORRECT ({pred})")
            else:
                lines.append(f"Last: WRONG (got {pred})")
        max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0][0] for l in lines)
        box_w = max_w + 36
        box_h = 16 + len(lines) * 26
        x1, y1 = cx - box_w // 2, cy - 12
        x2, y2 = cx + box_w // 2, y1 + box_h
        self._rounded_rect(img, (x1, y1), (x2, y2), (30, 28, 38), -1, 10)
        cv2.rectangle(img, (x1, y1), (x2, y2), C_PURPLE, 2)
        for i, line in enumerate(lines):
            col = C_GREEN if i == 0 else C_TEXT
            if "CORRECT" in line:
                col = C_GREEN
            elif "WRONG" in line:
                col = C_RED_KEY
            elif line.startswith("Draw digit"):
                col = C_YELLOW
            self._label(img, line, (cx - max_w // 2, y1 + 28 + i * 26), 0.52, col, 2)

    def _icon_pen(self, img, cx, cy, col):
        cv2.line(img, (cx - 6, cy + 6), (cx + 8, cy - 8), col, 2, cv2.LINE_AA)

    def draw_frame(
        self,
        img: np.ndarray,
        pen_color,
        brush_thickness: int,
        tool_mode: str,
        color_index: int,
        status_text: str,
        shapes_on: bool = True,
        test_mode=None,
    ):
        L = self.L
        img[: L.header_h] = C_BLACK
        img[L.h - L.footer_h :] = C_BLACK
        cv2.line(img, (0, L.header_h), (L.w, L.header_h), C_BORDER, 1)
        cv2.rectangle(
            img,
            (L.canvas_margin, L.canvas_top),
            (L.w - L.canvas_margin, L.canvas_bottom),
            C_BORDER,
            2,
        )

        # COLORS
        self._panel(img, L.p_colors, "COLORS")
        for i, (cx, cy) in enumerate(self._swatch_centers()):
            if i == 11:
                self._draw_rainbow(img, cx, cy, SWATCH_R)
            else:
                cv2.circle(img, (cx, cy), SWATCH_R, PALETTE[i], -1, cv2.LINE_AA)
                if i == 8:
                    cv2.circle(img, (cx, cy), SWATCH_R, C_TEXT_DIM, 1, cv2.LINE_AA)
            ring = (0, 0, 0) if i == 8 else (255, 255, 255)
            if i == color_index:
                cv2.circle(img, (cx, cy), SWATCH_R + 4, ring, 2, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), SWATCH_R + 6, C_PURPLE, 2, cv2.LINE_AA)

        # TOOLS
        self._panel(img, L.p_tools, "TOOLS")
        icons = {
            "pen": self._icon_pen,
            "eraser": lambda i, cx, cy, c: cv2.rectangle(i, (cx - 7, cy - 5), (cx + 7, cy + 5), c, -1),
            "clear": lambda i, cx, cy, c: cv2.line(i, (cx - 6, cy + 4), (cx + 6, cy - 4), c, 2),
            "save": lambda i, cx, cy, c: cv2.rectangle(i, (cx - 6, cy - 6), (cx + 6, cy + 6), c, 2),
            "predict": lambda i, cx, cy, c: self._label(i, "AI", (cx - 10, cy + 4), 0.4, c, 1),
            "undo": lambda i, cx, cy, c: cv2.polylines(i, [np.array([[cx + 6, cy - 4], [cx - 2, cy], [cx + 6, cy + 4]], np.int32)], False, c, 2),
        }
        labels = ["PEN", "ERASE", "CLR", "SAVE", "PRED", "UNDO"]
        for key, label, (bx1, by1, bx2, by2) in zip(TOOL_KEYS, labels, self._tool_rects()):
            active = tool_mode == key if key in ("pen", "eraser") else False
            fill = (48, 32, 52) if active else (30, 30, 36)
            self._rounded_rect(img, (bx1, by1), (bx2, by2), fill, -1, 4)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), C_PURPLE if active else C_PURPLE_DIM, 2 if active else 1)
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            icons[key](img, cx, cy - 6, C_TEXT)
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
            self._label(img, label, (cx - tw // 2, by2 - 8), 0.32, C_TEXT_DIM)

        # BRUSH + preview
        bx1, by1, bx2, by2 = L.p_brush
        self._panel(img, L.p_brush, "BRUSH")
        mid_y = self._brush_mid_y()
        btn_off = min(44, (bx2 - bx1) // 6)
        for cx, sym in [(bx1 + btn_off, "-"), (bx2 - btn_off, "+")]:
            cv2.circle(img, (cx, mid_y), 18, (38, 34, 44), -1, cv2.LINE_AA)
            self._label(img, sym, (cx - 5, mid_y + 6), 0.65, C_TEXT, 2)
        center_cx = (bx1 + bx2) // 2
        self._label(img, str(brush_thickness), (center_cx - 12, mid_y + 6), 0.55, C_TEXT, 2)
        # Brush preview circle
        prev_x = bx2 - 28
        prev_y = by1 + 24
        cv2.circle(img, (prev_x, prev_y), max(4, min(18, brush_thickness // 2 + 2)), pen_color, -1, cv2.LINE_AA)
        cv2.circle(img, (prev_x, prev_y), max(5, min(20, brush_thickness // 2 + 4)), C_PURPLE_DIM, 1, cv2.LINE_AA)
        track_y = by2 - max(18, int((by2 - by1) * 0.16))
        tx1, tx2 = bx1 + 14, bx2 - 14
        cv2.line(img, (tx1, track_y), (tx2, track_y), (55, 50, 62), 4, cv2.LINE_AA)
        t = (brush_thickness - 2) / 98.0
        cv2.circle(img, (int(tx1 + t * (tx2 - tx1)), track_y), 8, C_PURPLE, -1, cv2.LINE_AA)

        # PREDICTION PANEL
        px1, py1, px2, py2 = L.p_predict
        self._panel(img, L.p_predict, "AI PREDICTION")
        if self.prediction.visible and self.prediction.is_digit and self.prediction.digit >= 0:
            self._label(img, f"Digit: {self.prediction.digit}", (px1 + 10, py1 + 48), 0.55, C_GREEN, 2)
            self._label(
                img,
                f"Conf: {self.prediction.confidence:.1f}%",
                (px1 + 10, py1 + 72),
                0.45,
                C_TEXT,
            )
            if self.prediction.preview_28 is not None:
                prev = cv2.resize(self.prediction.preview_28, (56, 56), interpolation=cv2.INTER_NEAREST)
                prev_bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
                px, py = px2 - 66, py1 + 38
                img[py : py + 56, px : px + 56] = prev_bgr
                cv2.rectangle(img, (px - 1, py - 1), (px + 57, py + 57), C_PURPLE, 1)
        elif self.prediction.visible and not self.prediction.is_digit:
            self._label(img, "Not a digit", (px1 + 10, py1 + 48), 0.5, C_YELLOW, 2)
            note = self.prediction.note or "Draw 0-9 only"
            self._label(img, note[:22], (px1 + 10, py1 + 72), 0.38, C_TEXT_DIM)
            if self.prediction.preview_28 is not None:
                prev = cv2.resize(self.prediction.preview_28, (56, 56), interpolation=cv2.INTER_NEAREST)
                prev_bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
                px, py = px2 - 66, py1 + 38
                img[py : py + 56, px : px + 56] = prev_bgr
                cv2.rectangle(img, (px - 1, py - 1), (px + 57, py + 57), C_YELLOW, 1)
        else:
            self._label(img, "Draw digit", (px1 + 10, py1 + 48), 0.42, C_TEXT_DIM)
            self._label(img, "Press PRED", (px1 + 10, py1 + 68), 0.42, C_TEXT_DIM)
        if self.history_digits:
            hist = " ".join(self.history_digits[-6:])
            self._label(img, f"Hist: {hist}", (px1 + 10, py2 - 12), 0.38, C_TEXT_DIM)

        # Footer
        foot_y = L.h - L.footer_h
        cv2.line(img, (0, foot_y), (L.w, foot_y), C_PURPLE_DIM, 1)
        dot_y = foot_y + L.footer_h // 2
        dot_col = C_GREEN if status_text in ("READY", "TRACKING") else C_YELLOW
        cv2.circle(img, (20, dot_y), 5, dot_col, -1, cv2.LINE_AA)
        text_y = foot_y + L.footer_h - 12
        self._label(img, f"STATUS: {status_text}", (34, text_y), 0.48, C_TEXT_DIM)
        hint = "T:Test | S:Save | P:Predict | U:Undo | Q:Quit"
        if test_mode and test_mode.enabled:
            hint = "0-9:Target | T:Exit Test | P:Check | U:Undo | Q:Quit"
        tw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        self._label(img, hint, (L.w - tw - 14, text_y), 0.42, C_TEXT_DIM)

        if test_mode and test_mode.enabled:
            self._draw_test_overlay(img, test_mode, L)

        # Toasts
        now = time.time()
        self.toasts = [t for t in self.toasts if t.until > now]
        for i, toast in enumerate(self.toasts[-3:]):
            ty = L.canvas_top + 30 + i * 36
            tw = cv2.getTextSize(toast.message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            self._rounded_rect(img, (L.w // 2 - tw // 2 - 14, ty - 18), (L.w // 2 + tw // 2 + 14, ty + 10), (35, 35, 42), -1, 8)
            cv2.rectangle(img, (L.w // 2 - tw // 2 - 14, ty - 18), (L.w // 2 + tw // 2 + 14, ty + 10), toast.color, 1)
            self._label(img, toast.message, (L.w // 2 - tw // 2, ty + 4), 0.55, toast.color, 1)


class UIHitTester:
    """Maps pointer position to toolbar actions."""

    def __init__(self, layout: Layout, renderer: UIRenderer):
        self.L = layout
        self.R = renderer

    def in_canvas(self, y: int) -> bool:
        return self.L.canvas_top <= y <= self.L.canvas_bottom

    def hit_color(self, x: int, y: int) -> int:
        x1, y1, x2, y2 = self.L.p_colors
        if y > self.L.header_h or not (x1 <= x <= x2 and y1 <= y <= y2):
            return -1
        best_i, best_d = -1, 1e9
        hit_r = SWATCH_R + SWATCH_HIT_EXTRA
        for i, (cx, cy) in enumerate(self.R._swatch_centers()):
            d = math.hypot(x - cx, y - cy)
            if d <= hit_r and d < best_d:
                best_i, best_d = i, d
        return best_i

    def hit_tool(self, x: int, y: int) -> Optional[str]:
        if y > self.L.header_h:
            return None
        for key, rect in zip(TOOL_KEYS, self.R._tool_rects()):
            bx1, by1, bx2, by2 = rect
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return key
        return None

    def hit_brush(self, x: int, y: int) -> Optional[str]:
        bx1, by1, bx2, by2 = self.L.p_brush
        if not (bx1 <= x <= bx2 and by1 <= y <= by2):
            return None
        mid_y = self.R._brush_mid_y()
        btn_off = min(44, (bx2 - bx1) // 6)
        if math.hypot(x - (bx1 + btn_off), y - mid_y) <= 22:
            return "minus"
        if math.hypot(x - (bx2 - btn_off), y - mid_y) <= 22:
            return "plus"
        return None


def compose_frame(camera_feed: np.ndarray, img_canvas: np.ndarray, layout: Layout) -> np.ndarray:
    """Merge canvas into camera view inside bordered region."""
    L = layout
    frame = np.zeros((L.h, L.w, 3), np.uint8)
    merged = camera_feed.copy()
    gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    merged = cv2.bitwise_and(merged, inv)
    merged = cv2.bitwise_or(merged, img_canvas)
    frame[L.canvas_top : L.canvas_bottom, L.canvas_margin : L.w - L.canvas_margin] = merged[
        L.canvas_top : L.canvas_bottom, L.canvas_margin : L.w - L.canvas_margin
    ]
    return frame


def draw_cursor(img, x, y, layout: Layout):
    if not (layout.canvas_top <= y <= layout.canvas_bottom):
        return
    cv2.circle(img, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_selection_cursor(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), color, cv2.FILLED)
    cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), (255, 255, 255), 1)
