"""
MNIST-style image preprocessing for air-drawn digits.
Shared by training augmentation and live inference.
"""

import cv2
import numpy as np


def extract_ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Detect any pen color on dark background (not just grayscale)."""
    b, g, r = cv2.split(bgr)
    ink = np.maximum(np.maximum(r, g), b)
    _, binary = cv2.threshold(ink, 12, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_noise(binary: np.ndarray, min_area: int = 25) -> np.ndarray:
    """Remove tiny speckles from the canvas."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def keep_largest_digit(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected stroke (ignores stray dots)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _is_closed_loop(binary: np.ndarray) -> bool:
    """True for O/0/8-style closed shapes (ring or blob)."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return False
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 80:
        return False
    # Closed contour with significant perimeter → circle/zero-like
    return cv2.contourArea(cnt) > 200 or perimeter > 400


def thin_strokes_to_mnist(binary: np.ndarray) -> np.ndarray:
    """
    Reduce thick air strokes toward MNIST-like width.
    Safe for hollow shapes (0, 8) — falls back if erosion destroys the digit.
    """
    if cv2.countNonZero(binary) == 0:
        return binary

    before = cv2.countNonZero(binary)
    closed = _is_closed_loop(binary)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_r = float(dist.max())

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    if max_r <= 2.5:
        return binary

    # Closed loops (0, 8): minimal erosion — heavy erosion wipes out the ring
    if closed:
        erode_iters = min(2, max(1, int((max_r - 2.0) / 2.0)))
    else:
        erode_iters = min(4, max(1, int((max_r - 1.8) / 1.2)))

    thinned = cv2.erode(binary, kernel, iterations=erode_iters)
    thinned = cv2.dilate(thinned, kernel, iterations=1)
    after = cv2.countNonZero(thinned)

    # Fallback: erosion destroyed the digit (common with 0)
    if after < max(40, before * 0.2):
        mild = cv2.erode(binary, kernel, iterations=1)
        if cv2.countNonZero(mild) >= 40:
            return mild
        return binary

    return thinned


def fit_and_center_28x28(binary: np.ndarray) -> np.ndarray:
    """Crop, scale into 20x20 box, place in 28x28, center by mass."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    pad = max(6, int(max(w, h) * 0.22))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)
    cropped = binary[y1:y2, x1:x2]

    size = max(cropped.shape[0], cropped.shape[1], 1)
    scale = 20.0 / size
    new_w = max(1, int(cropped.shape[1] * scale))
    new_h = max(1, int(cropped.shape[0] * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    off_x = (28 - new_w) // 2
    off_y = (28 - new_h) // 2
    canvas28[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    m = cv2.moments(canvas28)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dx = int(14 - cx)
        dy = int(14 - cy)
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas28 = cv2.warpAffine(canvas28, M, (28, 28), borderValue=0)

    return canvas28


def _finalize_28x28(canvas28: np.ndarray) -> np.ndarray:
    """Light smoothing; keep stroke visible for rings and thin digits."""
    if cv2.countNonZero(canvas28) < 8:
        return canvas28
    blurred = cv2.GaussianBlur(canvas28, (3, 3), 0.3)
    _, out = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(out) >= 8:
        return out
    return canvas28


def preprocess_digit_image(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Full pipeline: BGR canvas region -> (flat 784, preview 28x28, is_empty).
    """
    if bgr is None or bgr.size == 0:
        blank = np.zeros((28, 28), dtype=np.uint8)
        return np.zeros((1, 784), dtype=np.float32), blank, True

    binary = extract_ink_mask(bgr)
    binary = remove_small_noise(binary)
    binary = keep_largest_digit(binary)

    ink_pixels = cv2.countNonZero(binary)
    if ink_pixels < 15:
        blank = np.zeros((28, 28), dtype=np.uint8)
        return np.zeros((1, 784), dtype=np.float32), blank, True

    thinned = thin_strokes_to_mnist(binary)
    canvas28 = fit_and_center_28x28(thinned)
    canvas28 = _finalize_28x28(canvas28)

    # If still empty, retry without thinning
    if cv2.countNonZero(canvas28) < 8:
        canvas28 = fit_and_center_28x28(binary)
        canvas28 = _finalize_28x28(canvas28)

    if cv2.countNonZero(canvas28) < 8:
        blank = np.zeros((28, 28), dtype=np.uint8)
        return np.zeros((1, 784), dtype=np.float32), blank, True

    flat = (canvas28.astype(np.float32) / 255.0).reshape(1, 784)
    return flat, canvas28, False


def augment_thick_stroke(flat_784: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate thick air-drawn stroke from a MNIST digit (for training)."""
    img = (flat_784.reshape(28, 28) * 255).astype(np.uint8)
    dilate = int(rng.integers(2, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(img, kernel, iterations=dilate)

    dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    thick = cv2.warpAffine(thick, M, (28, 28), borderValue=0)

    if rng.random() < 0.3:
        angle = float(rng.uniform(-12, 12))
        rot = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        thick = cv2.warpAffine(thick, rot, (28, 28), borderValue=0)

    return (thick.astype(np.float32) / 255.0).reshape(784)


def augment_figure_eight(flat_784: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Extra-thick figure-8 for digit 8 air drawing."""
    img = (flat_784.reshape(28, 28) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(img, k, iterations=int(rng.integers(3, 6)))
    dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    thick = cv2.warpAffine(thick, M, (28, 28), borderValue=0)
    return (thick.astype(np.float32) / 255.0).reshape(784)


def _count_holes(binary: np.ndarray) -> int:
    """Count enclosed holes inside the stroke (8 has 2, 0 has 1, 3/7 have 0)."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for i in range(len(contours)) if hierarchy[0][i][3] >= 0)


def analyze_digit_shape(preview_28: np.ndarray) -> dict:
    """Geometry hints from 28x28 ink layout."""
    img = preview_28 if preview_28.max() > 1 else (preview_28 * 255).astype(np.uint8)
    binary = (img > 127).astype(np.uint8) * 255
    mask = img > 127
    h, w = mask.shape
    total = float(np.sum(mask)) + 1.0
    top = float(np.sum(mask[: h // 2, :]))
    bot = float(np.sum(mask[h // 2 :, :]))
    top_band = float(np.sum(mask[: max(1, h // 4), :]))
    bot_band = float(np.sum(mask[max(0, 2 * h // 3) :, :]))
    left = float(np.sum(mask[:, : w // 3]))
    mid = float(np.sum(mask[:, w // 3 : 2 * w // 3]))
    right = float(np.sum(mask[:, 2 * w // 3 :]))
    bl = float(np.sum(mask[h // 2 :, : w // 2]))
    br = float(np.sum(mask[h // 2 :, w // 2 :]))
    holes = _count_holes(binary)
    top_mid = float(np.sum(mask[: h // 2, w // 4 : 3 * w // 4]))
    bot_mid = float(np.sum(mask[h // 2 :, w // 4 : 3 * w // 4]))
    loop_balance = abs(top / total - bot / total)
    balanced_loops = top / total > 0.28 and bot / total > 0.28
    seven_like = (
        top_band / total > 0.18
        and top / total > 0.42
        and left < right * 0.85
        and bot_band / total < 0.22
    )
    eight_like = (
        holes >= 2
        and balanced_loops
        and loop_balance < 0.055
        and top_mid / total > 0.10
        and bot_mid / total > 0.10
    )
    mid_band = float(np.sum(mask[h // 3 : 2 * h // 3, :]))
    top_left = float(np.sum(mask[: h // 2, : w // 2]))
    top_right = float(np.sum(mask[: h // 2, w // 2 :]))
    nine_tail = top / total > 0.38 and br > bl * 1.05 and bot_band / total > 0.06
    three_like = (
        holes <= 1
        and left / total < 0.32
        and right / total > 0.20
        and right > left * 1.15
        and bot / total > 0.32
        and top / total > 0.26
        and not seven_like
        and not eight_like
    )
    two_like = (
        holes <= 1
        and left / total < 0.30
        and top_band / total > 0.12
        and bot_band / total > 0.12
        and mid_band / total > 0.12
        and not three_like
        and not eight_like
    )
    six_like = (
        holes >= 1
        and bot / total > top / total * 1.03
        and loop_balance >= 0.055
        and not eight_like
    )
    nine_like = (
        (holes >= 1 and top / total > 0.34 and (nine_tail or top_right > top_left * 0.85))
        or nine_tail
    )
    four_like = (
        holes == 0
        and not eight_like
        and not three_like
        and mid_band / total > 0.16
        and top / total > 0.18
        and bot / total > 0.18
    )
    return {
        "top_share": top / total,
        "bot_share": bot / total,
        "left_share": left / total,
        "right_share": right / total,
        "holes": holes,
        "balanced_loops": balanced_loops,
        "nine_like_tail": nine_tail,
        "single_loop": top / total > 0.55 and bot / total < 0.3,
        "three_like": three_like,
        "two_like": two_like,
        "seven_like": seven_like,
        "eight_like": eight_like,
        "six_like": six_like,
        "nine_like": nine_like,
        "four_like": four_like,
    }


def _looks_like_rectangle(cnt: np.ndarray, area: float, aspect: float) -> bool:
    """True only for solid box-like shapes, not thick digit strokes (e.g. 8)."""
    peri = cv2.arcLength(cnt, True)
    if peri <= 1 or area <= 40:
        return False
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4:
        return False
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if min(rw, rh) <= 2 or aspect >= 3.5:
        return False
    box_fill = area / (rw * rh + 1)
    if box_fill < 0.62:
        return False
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1 or area / hull_area < 0.92:
        return False
    return True


def is_valid_digit_drawing(preview_28: np.ndarray, probs: np.ndarray) -> tuple[bool, str]:
    """
    Reject dots, lines, rectangles, and scribbles — only allow digit-like shapes (0-9).
    Returns (is_valid, reason_if_rejected).
    """
    img = preview_28 if preview_28.max() > 1 else (preview_28 * 255).astype(np.uint8)
    binary = (img > 127).astype(np.uint8) * 255
    ink = cv2.countNonZero(binary)
    if ink < 20:
        return False, "Too small - draw a full digit (0-9)."

    coords = cv2.findNonZero(binary)
    if coords is None:
        return False, "No shape found - draw a digit 0-9."

    x, y, w, h = cv2.boundingRect(coords)
    max_side = max(w, h, 1)
    min_side = max(min(w, h), 1)
    aspect = max_side / min_side
    fill = ink / (w * h + 1)
    shape = analyze_digit_shape(preview_28)

    top = float(np.max(probs))
    sorted_p = np.sort(probs)[::-1]
    margin = float(sorted_p[0] - sorted_p[1])

    # Dot / small blob — always reject
    if max_side <= 7 or (fill > 0.82 and max_side <= 11):
        return False, "Dot detected - draw 0-9 only."

    # Thin line (not 1) — always reject
    if aspect >= 4.5 and min_side <= 5:
        return False, "Line detected - draw 0-9 only."

    # When the model is confident about a digit, trust it over shape heuristics
    model_confident = top >= 0.48 or (top >= 0.38 and margin >= 0.12)
    digit_like = (
        shape["balanced_loops"]
        or shape["three_like"]
        or shape["seven_like"]
        or shape["eight_like"]
        or shape["six_like"]
        or shape["nine_like"]
        or shape["four_like"]
    )

    if not model_confident and not digit_like:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri > 1:
                approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)

                if _looks_like_rectangle(cnt, area, aspect):
                    return False, "Rectangle detected - draw 0-9 only."

                if len(approx) == 3 and area > 35:
                    return False, "Triangle detected - draw 0-9 only."

                circularity = 4 * np.pi * area / (peri * peri)
                if circularity > 0.88 and fill > 0.55 and max_side >= 10 and probs[0] < 0.35:
                    return False, "Circle detected - draw 0-9 only."

    if top < 0.32 and not digit_like:
        return False, "Not a digit - draw 0-9 only."

    if top < 0.45 and margin < 0.08 and not digit_like:
        return False, "Unclear shape - draw 0-9 only."

    return True, "OK"


def refine_prediction(probs: np.ndarray, preview_28: np.ndarray) -> tuple[int, float]:
    """
    Fix common air-drawn confusions using geometry + model probabilities.
    Especially 8 vs 9 (balanced loops vs top-heavy tail).
    """
    digit = int(np.argmax(probs))
    shape = analyze_digit_shape(preview_28)

    # 8 wrongly predicted — restore 6 or 3 when shape is clear
    if digit == 8 and shape["six_like"]:
        return 6, max(float(probs[6] * 100.0), 75.0)

    if digit == 8 and shape["three_like"]:
        return 3, max(float(probs[3] * 100.0), 75.0)

    # Only promote to 8 when figure-eight is unmistakable (2 holes, balanced)
    if digit in (3, 6, 9) and shape["eight_like"] and not shape["six_like"] and not shape["three_like"]:
        if probs[8] > 0.08 or (shape["holes"] >= 2 and shape["balanced_loops"]):
            return 8, max(float(probs[8] * 100.0), float(probs[digit] * 100.0))

    # 3 vs 2
    if digit == 2 and shape["three_like"] and not shape["two_like"]:
        return 3, max(float(probs[3] * 100.0), 72.0)

    if digit == 3 and shape["two_like"] and not shape["three_like"] and probs[2] > probs[3] * 0.7:
        return 2, float(probs[2] * 100.0)

    # 8 drawn in air often misclassified as 9
    if digit == 9 and shape["balanced_loops"] and probs[8] > 0.08:
        return 8, float(probs[8] * 100.0)

    if digit == 9 and not shape["nine_like_tail"] and shape["bot_share"] > 0.3 and probs[8] > probs[9] * 0.55:
        return 8, float(probs[8] * 100.0)

    # 9 with clear tail but predicted 8
    if digit == 8 and shape["nine_like_tail"] and probs[9] > probs[8] * 0.85:
        return 9, float(probs[9] * 100.0)

    # 0 vs 6/8: single round loop
    if digit in (6, 8, 9) and shape["single_loop"] and probs[0] > 0.1:
        if probs[0] >= probs[digit] * 0.5:
            return 0, float(probs[0] * 100.0)

    # 3 vs 7: thick air-drawn 3 often misread as 7
    if digit == 7 and shape["three_like"] and not shape["seven_like"]:
        return 3, max(float(probs[3] * 100.0), 72.0)

    if digit == 3 and shape["seven_like"] and not shape["three_like"] and probs[7] > probs[3] * 0.9:
        return 7, float(probs[7] * 100.0)

    # 4 vs 6 vs 9: digit 4 is open (no holes); 6 and 9 have enclosed loops
    holes = shape["holes"]
    if digit == 4 and holes >= 1:
        if shape["six_like"] or shape["bot_share"] > shape["top_share"] * 1.05:
            return 6, max(float(probs[6] * 100.0), 78.0)
        if shape["nine_like"] or shape["top_share"] >= shape["bot_share"]:
            return 9, max(float(probs[9] * 100.0), 78.0)

    if digit in (6, 9) and shape["four_like"] and holes == 0:
        return 4, max(float(probs[4] * 100.0), 75.0)

    if digit == 6 and shape["nine_like"] and not shape["six_like"]:
        return 9, max(float(probs[9] * 100.0), 72.0)

    if digit == 9 and shape["six_like"] and not shape["nine_like"]:
        return 6, max(float(probs[6] * 100.0), 72.0)

    if digit == 9 and shape["four_like"] and holes == 0 and not shape["nine_like"]:
        return 4, max(float(probs[4] * 100.0), 75.0)

    return digit, float(probs[digit] * 100.0)


def augment_hollow_zero(flat_784: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate air-drawn hollow circle for digit 0."""
    img = (flat_784.reshape(28, 28) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return augment_thick_stroke(flat_784, rng)
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    out = np.zeros((28, 28), dtype=np.uint8)
    thickness = int(rng.integers(3, 6))
    cv2.circle(out, (int(cx), int(cy)), max(4, int(radius)), 255, thickness)
    return (out.astype(np.float32) / 255.0).reshape(784)
