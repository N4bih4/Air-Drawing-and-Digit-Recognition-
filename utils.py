"""
Shared utilities: paths, logging, timestamps, storage, optional voice output.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

SAVED_DRAWINGS_DIR = PROJECT_ROOT / "saved_drawings"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_PATH = MODELS_DIR / "mnist_digit_model_v8.joblib"
HISTORY_PATH = PREDICTIONS_DIR / "prediction_history.json"
TEST_RESULTS_PATH = PREDICTIONS_DIR / "test_results.json"

ALL_DIRS = (
    SAVED_DRAWINGS_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    DATASET_DIR,
    SCREENSHOTS_DIR,
    LOGS_DIR,
)


def setup_directories() -> None:
    """Create project folders if they do not exist."""
    for folder in ALL_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "air_drawing") -> logging.Logger:
    """Configure console + file logging."""
    setup_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def timestamp_filename(prefix: str = "digit", ext: str = "png") -> str:
    """Example: digit_2026_06_07_14_25_30.png"""
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{prefix}_{ts}.{ext}"


def save_drawing_image(image, prefix: str = "digit") -> Path:
    """Save BGR canvas image with timestamp."""
    import cv2

    path = SAVED_DRAWINGS_DIR / timestamp_filename(prefix, "png")
    cv2.imwrite(str(path), image)
    return path


def save_prediction_record(
    digit: int,
    confidence: float,
    drawing_path: Path,
    processed_preview_path: Optional[Path] = None,
) -> Path:
    """Append prediction to history JSON and save a summary text file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "digit": int(digit),
        "confidence": round(float(confidence), 2),
        "drawing": str(drawing_path.name),
        "preview": processed_preview_path.name if processed_preview_path else None,
    }

    history: List[Dict[str, Any]] = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    history.append(record)
    history = history[-100:]
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    txt_path = PREDICTIONS_DIR / timestamp_filename("prediction", "txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Prediction: {digit}\n")
        f.write(f"Confidence: {confidence:.1f}%\n")
        f.write(f"Drawing: {drawing_path.name}\n")
        f.write(f"Time: {record['timestamp']}\n")

    return txt_path


def load_prediction_history(limit: int = 10) -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[-limit:]
    except (json.JSONDecodeError, OSError):
        return []


def accuracy_stats() -> Dict[str, Any]:
    """Basic stats from prediction history (count per digit)."""
    history = load_prediction_history(100)
    counts = {str(i): 0 for i in range(10)}
    confidences: List[float] = []
    for item in history:
        d = str(item.get("digit", ""))
        if d in counts:
            counts[d] += 1
        if "confidence" in item:
            confidences.append(float(item["confidence"]))
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {"counts": counts, "total": len(history), "avg_confidence": avg_conf}


@dataclass
class TestModeStats:
    """Live accuracy tracking during test mode."""

    enabled: bool = False
    target_digit: int = -1
    correct: int = 0
    total: int = 0
    per_digit: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_result: str = ""
    last_predicted: int = -1

    def __post_init__(self):
        if not self.per_digit:
            self.per_digit = {str(i): {"correct": 0, "total": 0} for i in range(10)}

    @property
    def accuracy_pct(self) -> float:
        return (self.correct / self.total * 100.0) if self.total else 0.0

    def record(self, expected: int, predicted: int, ok: bool) -> None:
        key = str(expected)
        self.per_digit[key]["total"] += 1
        self.total += 1
        if ok:
            self.correct += 1
            self.per_digit[key]["correct"] += 1
            self.last_result = "CORRECT"
        else:
            self.last_result = "WRONG"
        self.last_predicted = predicted

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
        self.target_digit = -1
        self.last_result = ""
        self.last_predicted = -1
        self.per_digit = {str(i): {"correct": 0, "total": 0} for i in range(10)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy_pct": round(self.accuracy_pct, 2),
            "per_digit": self.per_digit,
            "saved_at": datetime.now().isoformat(),
        }


def save_test_results(stats: TestModeStats) -> None:
    setup_directories()
    with open(TEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)


def load_test_results() -> Optional[Dict[str, Any]]:
    if not TEST_RESULTS_PATH.exists():
        return None
    try:
        with open(TEST_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def speak_digit(digit: int) -> None:
    """Optional voice: 'You drew digit 5.'"""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(f"You drew digit {digit}.")
        engine.runAndWait()
    except Exception:
        pass
