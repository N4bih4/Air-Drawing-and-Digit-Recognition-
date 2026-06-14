"""
MNIST digit classification: preprocessing + inference pipeline.
Uses scikit-learn (no TensorFlow required).
"""

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from mnist_preprocess import is_valid_digit_drawing, preprocess_digit_image, refine_prediction
from train_model import train
from utils import MODEL_PATH, setup_directories, setup_logging

logger = setup_logging("digit_classifier")


@dataclass
class PredictionResult:
    digit: int
    confidence: float
    processed_28: np.ndarray  # uint8 28x28 for UI preview
    is_empty: bool = False
    is_digit: bool = True
    message: str = ""


class DigitClassifier:
    """Loads MNIST MLP; preprocesses canvas strokes for prediction."""

    def __init__(self, auto_train: bool = True):
        setup_directories()
        self.model = None
        self._load_model(auto_train=auto_train)

    def _load_model(self, auto_train: bool = True) -> None:
        if MODEL_PATH.exists():
            logger.info("Loading digit model from %s", MODEL_PATH)
            self.model = joblib.load(MODEL_PATH)
            return

        if auto_train:
            logger.info("No model found — training MNIST classifier (one-time)...")
            train()
            self.model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

    def predict_region(self, canvas_bgr: np.ndarray) -> PredictionResult:
        """Predict from an already-cropped canvas image."""
        features, preview, is_empty = preprocess_digit_image(canvas_bgr)
        if is_empty:
            return PredictionResult(
                digit=-1,
                confidence=0.0,
                processed_28=preview,
                is_empty=True,
                message="Canvas is empty — draw a digit first.",
            )
        probs = self.model.predict_proba(features)[0]
        preview_u8 = preview if preview.max() > 1 else (preview * 255).astype(np.uint8)

        ok, reason = is_valid_digit_drawing(preview_u8, probs)
        if not ok:
            logger.info("Rejected non-digit shape: %s", reason)
            return PredictionResult(
                digit=-1,
                confidence=0.0,
                processed_28=preview,
                is_empty=False,
                is_digit=False,
                message=reason,
            )

        digit, confidence = refine_prediction(probs, preview_u8)
        logger.info(
            "Predict digit=%s conf=%.1f%% (raw=%s)",
            digit,
            confidence,
            int(np.argmax(probs)),
        )
        return PredictionResult(
            digit=digit,
            confidence=confidence,
            processed_28=preview,
            is_empty=False,
            is_digit=True,
            message="OK",
        )

    def predict(self, canvas_bgr: np.ndarray, crop_region=None) -> PredictionResult:
        """Full predict pipeline with optional crop on full-frame canvas."""
        if crop_region:
            x1, y1, x2, y2 = crop_region
            roi = canvas_bgr[y1:y2, x1:x2]
        else:
            roi = canvas_bgr
        return self.predict_region(roi)
