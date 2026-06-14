"""
Smart Air Drawing & Digit Recognition System — main entry point.
"""

import time
from typing import Optional

import cv2
import numpy as np

from digit_classifier import DigitClassifier, PredictionResult
from hand_tracking_module import HandDetector
from shape_recognition_module import apply_stroke_correction, shape_display_name
from ui_components import (
    UIRenderer,
    UIHitTester,
    build_layout,
    compose_frame,
    draw_cursor,
    draw_selection_cursor,
    get_work_area_size,
    pen_color_from_index,
)
from utils import (
    PREDICTIONS_DIR,
    TestModeStats,
    accuracy_stats,
    save_drawing_image,
    save_prediction_record,
    save_test_results,
    setup_directories,
    setup_logging,
    speak_digit,
    timestamp_filename,
)

WINDOW_NAME = "Smart Air Drawing & Digit AI"
AUTO_PREDICT_IDLE_SEC = 3.0
MIN_INK_PIXELS = 200


class CanvasHistory:
    """Undo / redo stack for the drawing canvas."""

    def __init__(self, shape):
        self.shape = shape
        self.undo_stack: list = []
        self.redo_stack: list = []

    def snapshot(self, canvas: np.ndarray) -> None:
        self.undo_stack.append(canvas.copy())
        self.redo_stack.clear()
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self, current: np.ndarray) -> np.ndarray:
        if not self.undo_stack:
            return current
        self.redo_stack.append(current.copy())
        return self.undo_stack.pop()

    def redo(self, current: np.ndarray) -> np.ndarray:
        if not self.redo_stack:
            return current
        self.undo_stack.append(current.copy())
        return self.redo_stack.pop()


class SmartAirDrawingApp:
    """Orchestrates hand tracking, drawing, UI, and MNIST digit prediction."""

    def __init__(self):
        setup_directories()
        self.logger = setup_logging("main")
        self.layout = build_layout(*get_work_area_size())
        L = self.layout

        self.ui = UIRenderer(self.layout)
        self.hit = UIHitTester(self.layout, self.ui)

        self.logger.info("Initializing hand detector...")
        self.detector = HandDetector(min_detection_confidence=0.85)

        self.logger.info("Loading MNIST digit classifier...")
        self.classifier = DigitClassifier(auto_train=True)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, L.w)
        self.cap.set(4, L.h)

        self.pen_color = (0, 0, 255)
        self.color_index = 0
        self.tool_mode = "pen"
        self.brush_thickness = 15
        self.eraser_thickness = 50
        self.shape_auto_correct = True

        self.canvas = np.zeros((L.h, L.w, 3), np.uint8)
        self.history = CanvasHistory(self.canvas.shape)
        self.current_stroke: list = []
        self.is_drawing = False
        self.prev_canvas: Optional[np.ndarray] = None

        self.xp = self.yp = 0
        self.smooth_x = self.smooth_y = 0

        self.status_text = "READY"
        self.last_saved = 0.0
        self.last_resize = 0.0
        self.last_status_change = 0.0
        self.last_draw_time = 0.0
        self.auto_predict_enabled = True
        self.voice_enabled = True
        self._idle_predicted = False
        self.test_mode = TestModeStats()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, L.w, L.h)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        self.ui.add_toast("Smart Air Drawing AI Ready", kind="info")

    def _draw_color(self):
        return (0, 0, 0) if self.tool_mode == "eraser" else self.pen_color

    def _line_thickness(self):
        return self.eraser_thickness if self.tool_mode == "eraser" else self.brush_thickness

    def _finish_stroke(self):
        if self.tool_mode != "pen":
            return
        corrected, shape_type, _ = apply_stroke_correction(
            self.canvas,
            self.prev_canvas,
            self.current_stroke,
            self.pen_color,
            self.brush_thickness,
            enabled=self.shape_auto_correct,
        )
        self.canvas = corrected
        if shape_type:
            self.ui.add_toast(f"Shape: {shape_display_name(shape_type)}", kind="shape")

    def _extract_canvas_region(self) -> np.ndarray:
        L = self.layout
        return self.canvas[
            L.canvas_top : L.canvas_bottom,
            L.canvas_margin : L.w - L.canvas_margin,
        ].copy()

    def save_drawing(self) -> Optional[str]:
        region = self._extract_canvas_region()
        path = save_drawing_image(region, "digit")
        self.last_saved = time.time()
        self.last_status_change = time.time()
        self.status_text = "SAVED"
        self.ui.add_toast(f"Saved {path.name}", kind="save")
        self.logger.info("Drawing saved to %s", path)
        return str(path)

    def _clear_canvas(self) -> None:
        self.history.snapshot(self.canvas)
        self.canvas = np.zeros_like(self.canvas)
        self.prev_canvas = None
        self.ui.clear_prediction_display()
        self._idle_predicted = False

    def _record_test_result(self, predicted: int, ok: bool) -> None:
        expected = self.test_mode.target_digit
        self.test_mode.record(expected, predicted, ok)
        save_test_results(self.test_mode)
        acc = self.test_mode.accuracy_pct
        if ok:
            msg = f"CORRECT! {expected} = {predicted} | Acc: {acc:.1f}%"
            self.ui.add_toast(msg, color=(90, 230, 90), duration=3.0)
            self.status_text = f"TEST OK {acc:.0f}%"
        else:
            got = str(predicted) if predicted >= 0 else "none"
            msg = f"WRONG! Expected {expected}, got {got} | Acc: {acc:.1f}%"
            self.ui.add_toast(msg, color=(90, 90, 255), duration=3.0)
            self.status_text = f"TEST FAIL {acc:.0f}%"
        self.logger.info(
            "Test result expected=%s predicted=%s ok=%s acc=%.1f%% (%d/%d)",
            expected,
            predicted,
            ok,
            acc,
            self.test_mode.correct,
            self.test_mode.total,
        )

    def predict_digit(self, auto: bool = False) -> Optional[PredictionResult]:
        region = self._extract_canvas_region()
        path = save_drawing_image(region, "digit")
        result = self.classifier.predict_region(region)

        if result.is_empty:
            self.ui.add_toast(result.message, color=(0, 165, 255))
            self.status_text = "NO INK"
            if self.test_mode.enabled and self.test_mode.target_digit >= 0:
                self._record_test_result(-1, False)
            return None

        if not result.is_digit:
            self.ui.set_rejected(result.processed_28, result.message)
            self.status_text = "NOT A DIGIT"
            if not auto:
                self.ui.add_toast(result.message, color=(0, 165, 255))
            if self.test_mode.enabled and self.test_mode.target_digit >= 0:
                self._record_test_result(-1, False)
            return None

        preview_path = None
        import cv2 as _cv2

        preview_file = PREDICTIONS_DIR / timestamp_filename("preview", "png")
        _cv2.imwrite(str(preview_file), result.processed_28)
        preview_path = preview_file

        save_prediction_record(result.digit, result.confidence, path, preview_path)
        self.ui.set_prediction(result.digit, result.confidence, result.processed_28)
        self.status_text = f"AI: {result.digit}"
        msg = f"Prediction: {result.digit} ({result.confidence:.1f}%)"
        self.ui.add_toast(msg, kind="predict")
        self.logger.info(msg)

        if self.voice_enabled and not auto:
            speak_digit(result.digit)

        if self.test_mode.enabled and self.test_mode.target_digit >= 0:
            ok = result.digit == self.test_mode.target_digit
            self._record_test_result(result.digit, ok)

        stats = accuracy_stats()
        self.logger.info("History: %d predictions, avg conf %.1f%%", stats["total"], stats["avg_confidence"])
        return result

    def _handle_tool_action(self, tool: str) -> None:
        if tool == "pen":
            self.tool_mode = "pen"
            self.status_text = "READY"
        elif tool == "eraser":
            self.tool_mode = "eraser"
            self.status_text = "READY"
        elif tool == "clear":
            self._clear_canvas()
            self.status_text = "CLEARED"
            self.ui.add_toast("Canvas cleared", kind="clear")
        elif tool == "save":
            if time.time() - self.last_saved > 1.0:
                self.save_drawing()
        elif tool == "predict":
            self.predict_digit()
        elif tool == "undo":
            self.canvas = self.history.undo(self.canvas)
            self.ui.add_toast("Undo", kind="undo")

    def _handle_ui_click(self, x: int, y: int) -> None:
        color_hit = self.hit.hit_color(x, y)
        if color_hit >= 0:
            self.color_index = color_hit
            self.pen_color = pen_color_from_index(color_hit)
            self.tool_mode = "pen"
            self.status_text = "READY"
            return

        tool = self.hit.hit_tool(x, y)
        if tool:
            self._handle_tool_action(tool)
            return

        brush = self.hit.hit_brush(x, y)
        if brush == "minus" and time.time() - self.last_resize > 0.12:
            self.brush_thickness = max(2, self.brush_thickness - 2)
            self.last_resize = time.time()
        elif brush == "plus" and time.time() - self.last_resize > 0.12:
            self.brush_thickness = min(100, self.brush_thickness + 2)
            self.last_resize = time.time()

    def _process_hand(self, img, lm_list) -> tuple:
        cursor_draw = cursor_select = None
        L = self.layout

        if not lm_list:
            if self.is_drawing:
                self.is_drawing = False
                self._finish_stroke()
                self.current_stroke = []
            self.xp = self.yp = 0
            return cursor_draw, cursor_select, False

        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        fingers = self.detector.fingers_up()
        drawing = fingers[1] and not fingers[2] and not any(fingers[3:])

        if drawing and self.hit.in_canvas(y1) and L.canvas_margin <= x1 <= L.w - L.canvas_margin:
            cursor_draw = (x1, y1)
            if not self.is_drawing:
                self.history.snapshot(self.canvas)
                self.is_drawing = True
                self.prev_canvas = self.canvas.copy()
                self.current_stroke = []
            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1
                self.smooth_x, self.smooth_y = x1, y1
            self.smooth_x = int(x1 * 0.5 + self.smooth_x * 0.5)
            self.smooth_y = int(y1 * 0.5 + self.smooth_y * 0.5)
            cv2.line(
                self.canvas,
                (self.xp, self.yp),
                (self.smooth_x, self.smooth_y),
                self._draw_color(),
                self._line_thickness(),
            )
            if self.tool_mode == "pen":
                self.current_stroke.append((self.smooth_x, self.smooth_y))
            self.xp, self.yp = self.smooth_x, self.smooth_y
            self.last_draw_time = time.time()
            self._idle_predicted = False
        else:
            if self.is_drawing:
                self.is_drawing = False
                self._finish_stroke()
                self.current_stroke = []
            self.xp = self.yp = 0

            if fingers[1] and fingers[2] and not any(fingers[3:]):
                ux, uy = (x1 + x2) // 2, (y1 + y2) // 2
                if uy < L.header_h:
                    self._handle_ui_click(ux, uy)
                sel = self.pen_color if self.tool_mode == "pen" else (200, 200, 200)
                cursor_select = (x1, y1, x2, y2, sel)

        return cursor_draw, cursor_select, True

    def _maybe_auto_predict(self):
        if self.test_mode.enabled:
            return
        if not self.auto_predict_enabled or self.is_drawing or self._idle_predicted:
            return
        if self.last_draw_time <= 0:
            return
        if time.time() - self.last_draw_time < AUTO_PREDICT_IDLE_SEC:
            return
        gray = cv2.cvtColor(self._extract_canvas_region(), cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]) < MIN_INK_PIXELS:
            return
        self.predict_digit(auto=True)
        self._idle_predicted = True

    def _toggle_test_mode(self) -> None:
        self.test_mode.enabled = not self.test_mode.enabled
        if self.test_mode.enabled:
            self.auto_predict_enabled = False
            self.ui.add_toast("Test Mode ON - press 0-9 then draw", color=(0, 230, 255), duration=3.0)
            self.status_text = "TEST MODE"
        else:
            self.auto_predict_enabled = True
            if self.test_mode.total > 0:
                acc = self.test_mode.accuracy_pct
                save_test_results(self.test_mode)
                self.ui.add_toast(
                    f"Test done: {acc:.1f}% ({self.test_mode.correct}/{self.test_mode.total})",
                    color=(90, 230, 90),
                    duration=4.0,
                )
            else:
                self.ui.add_toast("Test Mode OFF", kind="info")
            self.status_text = "READY"

    def _handle_keys(self, key: int) -> bool:
        if key == ord("q"):
            return False
        if key == ord("t"):
            self._toggle_test_mode()
        elif self.test_mode.enabled and ord("0") <= key <= ord("9"):
            self.test_mode.target_digit = key - ord("0")
            self._clear_canvas()
            self.ui.add_toast(f"Target digit: {self.test_mode.target_digit}", color=(0, 230, 255))
            self.status_text = f"DRAW {self.test_mode.target_digit}"
        elif self.test_mode.enabled and key == ord("x"):
            self.test_mode.reset()
            save_test_results(self.test_mode)
            self.ui.add_toast("Test stats reset", kind="info")
        elif key == ord("s"):
            self.save_drawing()
        elif key == ord("p"):
            self.predict_digit()
        elif key == ord("u"):
            self.canvas = self.history.undo(self.canvas)
            self.ui.add_toast("Undo")
        elif key == ord("r"):
            self.canvas = self.history.redo(self.canvas)
            self.ui.add_toast("Redo")
        elif key == ord("a"):
            self.shape_auto_correct = not self.shape_auto_correct
            self.ui.add_toast(f"Auto-Shape: {'ON' if self.shape_auto_correct else 'OFF'}")
        elif key == ord("v"):
            self.voice_enabled = not self.voice_enabled
            self.ui.add_toast(f"Voice: {'ON' if self.voice_enabled else 'OFF'}")
        return True

    def run(self):
        L = self.layout
        self.logger.info("Application started (%dx%d)", L.w, L.h)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (L.w, L.h))
            frame = cv2.flip(frame, 1)
            frame = self.detector.find_hands(frame, draw=False)
            lm_list = self.detector.find_position(frame, draw=False)

            cursor_draw, cursor_select, hand_ok = self._process_hand(frame, lm_list)
            self._maybe_auto_predict()

            display = compose_frame(frame, self.canvas, L)
            if hand_ok:
                self.detector.draw_hand_overlay(
                    display, L.canvas_top, L.canvas_bottom, L.canvas_margin
                )

            if time.time() - self.last_saved < 1.2:
                self.status_text = "SAVED"
            elif self.status_text in ("SAVED", "CLEARED") and time.time() - self.last_status_change > 1.5:
                self.status_text = "TRACKING" if hand_ok else "READY"
            elif hand_ok and self.status_text == "READY":
                self.status_text = "TRACKING"
            elif not hand_ok and self.status_text == "TRACKING":
                self.status_text = "READY"

            self.ui.draw_frame(
                display,
                self.pen_color,
                self.brush_thickness,
                self.tool_mode,
                self.color_index,
                self.status_text,
                self.shape_auto_correct,
                self.test_mode,
            )

            if cursor_draw:
                draw_cursor(display, *cursor_draw, L)
            if cursor_select:
                draw_selection_cursor(display, *cursor_select)

            if time.time() - self.last_saved < 0.9:
                cy = (L.canvas_top + L.canvas_bottom) // 2
                self.ui._label(display, "SAVED!", (L.w // 2 - 55, cy), 1.1, (90, 230, 90), 2)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not self._handle_keys(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Application closed.")


def main():
    SmartAirDrawingApp().run()


if __name__ == "__main__":
    main()
