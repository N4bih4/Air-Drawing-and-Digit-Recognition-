"""
AI-based shape recognition and automatic stroke correction.

Uses geometric analysis (contour fitting, PCA line fit, circularity) to detect
rough air-drawn shapes and replace them with clean primitives.
"""

import cv2
import numpy as np
import math

MIN_POINTS = 12
MIN_SIZE_PX = 22
CONFIDENCE_THRESHOLD = 0.52


def _scalar(value):
    """Extract a Python float from cv2/numpy scalar or 1-element array."""
    return float(np.asarray(value).reshape(-1)[0])


def _path_length(points):
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(
            points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]
        )
    return total


def _bounding_size(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(xs) - min(xs), max(ys) - min(ys)


def _fit_line_endpoints(points):
    """Fit a line with PCA / cv2.fitLine; return snapped endpoints."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = _scalar(vx), _scalar(vy), _scalar(x0), _scalar(y0)

    # Project all points onto the line direction
    t_vals = []
    for px, py in points:
        t_vals.append((px - x0) * vx + (py - y0) * vy)
    t_min, t_max = min(t_vals), max(t_vals)

    p1 = (int(x0 + vx * t_min), int(y0 + vy * t_min))
    p2 = (int(x0 + vx * t_max), int(y0 + vy * t_max))

    # Mean distance from line (lower = straighter)
    err_sum = 0.0
    for px, py in points:
        dist = abs((px - x0) * vy - (py - y0) * vx)
        err_sum += dist
    mean_err = err_sum / len(points)
    span = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    score = max(0.0, 1.0 - mean_err / max(span * 0.15, 8.0))

    return p1, p2, score


def _circle_fit(points):
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cx, cy, radius = int(cx), int(cy), int(radius)

    if radius < 8:
        return None, 0.0

    dists = [math.hypot(p[0] - cx, p[1] - cy) for p in points]
    mean_d = sum(dists) / len(dists)
    if mean_d < 1:
        return None, 0.0
    variance = sum((d - mean_d) ** 2 for d in dists) / len(dists)
    std_ratio = math.sqrt(variance) / mean_d
    score = max(0.0, 1.0 - std_ratio * 2.2)

    perimeter = cv2.arcLength(contour, closed=True)
    area = cv2.contourArea(contour)
    if perimeter > 0:
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        score = min(1.0, score * 0.5 + circularity * 0.5)

    return (cx, cy, radius), score


def _quad_from_contour(points):
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1:
        return None, None, 0.0

    for eps_factor in (0.04, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]
                if w < 1 or h < 1:
                    continue
                aspect = min(w, h) / max(w, h)
                is_square = aspect > 0.82
                box = cv2.boxPoints(rect).astype(np.int32)
                score = 0.65 + 0.35 * aspect if is_square else 0.55 + 0.25 * aspect
                shape = "square" if is_square else "rectangle"
                return shape, box.reshape(-1, 1, 2), min(1.0, score)

    return None, None, 0.0


def _triangle_from_contour(points):
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1:
        return None, 0.0

    for eps_factor in (0.06, 0.08, 0.1, 0.12):
        approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
        if len(approx) == 3 and cv2.isContourConvex(approx):
            score = 0.7
            return approx, score
    return None, 0.0


def recognize_shape(points):
    """
    Analyze a stroke and return (shape_type, params, confidence).
    shape_type: line | circle | rectangle | square | triangle | none
    """
    if len(points) < MIN_POINTS:
        return "none", None, 0.0

    w, h = _bounding_size(points)
    if w < MIN_SIZE_PX and h < MIN_SIZE_PX:
        return "none", None, 0.0

    path_len = _path_length(points)
    if path_len < MIN_SIZE_PX:
        return "none", None, 0.0

    start, end = points[0], points[-1]
    gap = math.hypot(start[0] - end[0], start[1] - end[1])
    closed = gap < path_len * 0.38

    candidates = []

    # ── Open stroke → line ──
    if not closed or gap / path_len > 0.55:
        p1, p2, line_score = _fit_line_endpoints(points)
        straightness = gap / path_len if path_len else 0
        combined = line_score * 0.6 + min(1.0, straightness) * 0.4
        if straightness > 0.68 or line_score > 0.55:
            candidates.append(("line", (p1, p2), combined))

    # ── Closed stroke → circle / triangle / quad ──
    if closed or gap / path_len < 0.45:
        circle_data, circle_score = _circle_fit(points)
        if circle_data and circle_score > 0.45:
            candidates.append(("circle", circle_data, circle_score))

        tri, tri_score = _triangle_from_contour(points)
        if tri is not None and tri_score > 0.5:
            candidates.append(("triangle", tri, tri_score))

        quad_shape, quad_pts, quad_score = _quad_from_contour(points)
        if quad_shape and quad_score > 0.5:
            candidates.append((quad_shape, quad_pts, quad_score))

    if not candidates:
        return "none", None, 0.0

    # Prefer highest confidence; tie-break closed shapes by circle vs poly
    candidates.sort(key=lambda c: c[2], reverse=True)
    shape_type, params, confidence = candidates[0]

    if confidence < CONFIDENCE_THRESHOLD:
        return "none", None, confidence

    return shape_type, params, confidence


def draw_corrected_shape(canvas, shape_type, params, color, thickness):
    """Draw the corrected primitive onto the canvas."""
    if shape_type == "line":
        cv2.line(canvas, params[0], params[1], color, thickness, cv2.LINE_AA)
    elif shape_type == "circle":
        cv2.circle(canvas, (params[0], params[1]), params[2], color, thickness, cv2.LINE_AA)
    elif shape_type in ("rectangle", "square"):
        cv2.drawContours(canvas, [params], -1, color, thickness, cv2.LINE_AA)
    elif shape_type == "triangle":
        cv2.drawContours(canvas, [params], -1, color, thickness, cv2.LINE_AA)


def shape_display_name(shape_type):
    names = {
        "line": "Line",
        "circle": "Circle",
        "rectangle": "Rectangle",
        "square": "Square",
        "triangle": "Triangle",
    }
    return names.get(shape_type, "")


def apply_stroke_correction(canvas, backup_canvas, stroke, color, thickness, enabled=True):
    """
    If enabled, replace messy stroke with a clean shape on backup_canvas.
    Returns (new_canvas, shape_type or None, confidence).
    """
    if not enabled or backup_canvas is None or len(stroke) < MIN_POINTS:
        return canvas, None, 0.0

    shape_type, params, confidence = recognize_shape(stroke)
    if shape_type == "none":
        return canvas, None, confidence

    corrected = backup_canvas.copy()
    draw_corrected_shape(corrected, shape_type, params, color, thickness)
    return corrected, shape_type, confidence
