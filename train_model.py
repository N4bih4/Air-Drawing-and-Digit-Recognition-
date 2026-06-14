"""
Train MNIST digit classifier (scikit-learn MLP).
Includes thick-stroke augmentation to match air-drawn digits.

Run standalone: python train_model.py
"""

import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from mnist_preprocess import augment_figure_eight, augment_hollow_zero, augment_thick_stroke
from utils import DATASET_DIR, MODEL_PATH, MODELS_DIR, setup_directories, setup_logging

logger = setup_logging("train_model")


def load_mnist():
    """Download MNIST via OpenML and cache in dataset/."""
    setup_directories()
    cache_x = DATASET_DIR / "mnist_x.npy"
    cache_y = DATASET_DIR / "mnist_y.npy"

    if cache_x.exists() and cache_y.exists():
        logger.info("Loading cached MNIST from dataset/")
        return np.load(cache_x), np.load(cache_y)

    logger.info("Downloading MNIST dataset (first run, ~60 MB)...")
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    np.save(cache_x, x)
    np.save(cache_y, y)
    logger.info("MNIST cached to dataset/ (%s samples)", len(x))
    return x, y


def build_augmented_dataset(x: np.ndarray, y: np.ndarray, max_samples: int = 25000):
    """
    Original MNIST + thick-stroke variants (simulates air drawing).
    """
    rng = np.random.default_rng(42)
    n = min(max_samples, len(x))
    idx = rng.choice(len(x), n, replace=False)
    x_sub, y_sub = x[idx], y[idx]

    aug_x = [x_sub]
    aug_y = [y_sub]

    logger.info("Augmenting with thick air-drawn stroke variants...")
    thick_x = []
    thick_y = []
    for i in range(n):
        thick_x.append(augment_thick_stroke(x_sub[i], rng))
        thick_y.append(y_sub[i])
        if rng.random() < 0.5:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(y_sub[i])
        # Extra hollow-ring samples for digit 0
        if y_sub[i] == 0 and rng.random() < 0.7:
            thick_x.append(augment_hollow_zero(x_sub[i], rng))
            thick_y.append(0)
        # Extra thick figure-8 for digit 8 (often confused with 3, 6, 9)
        if y_sub[i] == 8 and rng.random() < 0.85:
            thick_x.append(augment_figure_eight(x_sub[i], rng))
            thick_y.append(8)
            if rng.random() < 0.65:
                thick_x.append(augment_figure_eight(x_sub[i], rng))
                thick_y.append(8)
            if rng.random() < 0.4:
                thick_x.append(augment_thick_stroke(x_sub[i], rng))
                thick_y.append(8)
        # Thick 9 variants to separate from 8
        if y_sub[i] == 9 and rng.random() < 0.5:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(9)
        # Extra thick 3/2 variants (often confused when drawn in air)
        if y_sub[i] == 3 and rng.random() < 0.75:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(3)
            if rng.random() < 0.5:
                thick_x.append(augment_thick_stroke(x_sub[i], rng))
                thick_y.append(3)
        if y_sub[i] == 2 and rng.random() < 0.65:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(2)
            if rng.random() < 0.4:
                thick_x.append(augment_thick_stroke(x_sub[i], rng))
                thick_y.append(2)
        # Extra thick 6 variants (often confused with 8)
        if y_sub[i] == 6 and rng.random() < 0.85:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(6)
            if rng.random() < 0.55:
                thick_x.append(augment_thick_stroke(x_sub[i], rng))
                thick_y.append(6)
        # Extra thick 7 variants to separate from 3
        if y_sub[i] == 7 and rng.random() < 0.5:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(7)
        # Extra thick 4/6/9 (often confused when drawn in air)
        if y_sub[i] in (4, 6, 9) and rng.random() < 0.8:
            thick_x.append(augment_thick_stroke(x_sub[i], rng))
            thick_y.append(y_sub[i])
            if rng.random() < 0.55:
                thick_x.append(augment_thick_stroke(x_sub[i], rng))
                thick_y.append(y_sub[i])

    aug_x.append(np.array(thick_x, dtype=np.float32))
    aug_y.append(np.array(thick_y, dtype=np.int32))

    x_out = np.vstack(aug_x)
    y_out = np.concatenate(aug_y)
    logger.info("Training set size: %s (with air-drawn augmentation)", len(x_out))
    return x_out, y_out


def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )


def train() -> Path:
    """Train model and save to models/mnist_digit_model_v2.joblib."""
    setup_directories()
    x, y = load_mnist()
    x_aug, y_aug = build_augmented_dataset(x, y)

    logger.info("Training scikit-learn MLP...")
    model = build_model()
    model.fit(x_aug, y_aug)

    acc = model.score(x_aug, y_aug)
    logger.info("Training accuracy: %.2f%%", acc * 100)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return MODEL_PATH


if __name__ == "__main__":
    train()
