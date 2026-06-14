# Smart Air Drawing & Digit Recognition System

AI-powered virtual whiteboard: draw in the air with hand gestures, save artwork, and classify handwritten digits using a **MNIST-trained neural network**.

## Features

### Drawing
- Real-time webcam + MediaPipe hand tracking
- Air draw with **1 finger**; toolbar select with **2 fingers**
- 12-color palette, pen, eraser, clear, save
- Adjustable brush size with live preview
- Undo / redo (`U` / `R`)
- AI shape auto-correction (line, circle, square, triangle) — toggle with `A`

### AI Digit Recognition
- MNIST neural network (scikit-learn MLP, digits **0–9**)
- No TensorFlow required — works on **Python 3.14+**
- Auto-download & cache dataset on first run
- Train once, load model on every startup
- **Predict** button or `P` key
- Auto-predict **2 seconds** after you stop drawing
- Confidence score + 28×28 processed preview
- Prediction history & statistics
- Optional voice: *"You drew digit 5."* (`V` to toggle)

### Professional UI
- Full-screen dark theme
- Toolbar: Colors | Tools | Brush | AI Prediction panel
- Status bar, toasts, save confirmation
- Hand skeleton overlay on canvas

## Project Structure

```
project/
├── main.py                  # Application entry (OOP orchestrator)
├── hand_tracking_module.py  # MediaPipe hand detection
├── shape_recognition_module.py
├── digit_classifier.py      # Preprocess + MNIST inference
├── train_model.py           # Train & save MNIST model
├── ui_components.py         # Layout, toolbar, prediction panel
├── utils.py                 # Paths, logging, storage, voice
├── models/                  # Trained model (auto-created)
├── dataset/                 # Cached MNIST arrays
├── saved_drawings/          # Timestamped PNG exports
├── predictions/             # Results + history JSON
├── screenshots/
├── logs/
├── requirements.txt
└── README.md
```

## Installation

**Requirements:** Python 3.9+, webcam, Windows/macOS/Linux

```bash
cd "d:\6th sem\AI\project"
pip install -r requirements.txt
```

> First launch downloads MediaPipe hand model + MNIST data and may **train the digit model (~1–3 min)**. Later runs load the saved model instantly.

### Train model manually (optional)

```bash
python train_model.py
```

## Run

```bash
python main.py
```

## Controls

| Input | Action |
|--------|--------|
| 1 finger | Draw on canvas |
| 2 fingers | Select toolbar (hover with midpoint) |
| `P` or **PRED** | Predict digit |
| `S` or **SAVE** | Save drawing |
| `U` | Undo |
| `R` | Redo |
| `A` | Toggle auto-shape correction |
| `V` | Toggle voice feedback |
| `Q` | Quit |

## Digit recognition tips

1. Draw **one digit** large and centered on the canvas.
2. Use **white** or a bright color for best MNIST matching.
3. Lift your finger — auto-predict runs after 2 seconds, or press **PRED**.
4. Check the **AI PREDICTION** panel for digit, confidence %, and 28×28 preview.

## Prediction pipeline

```
Camera → Hand Tracking → Canvas → Save PNG → Grayscale →
Threshold → Crop & Center → 28×28 → Normalize → MNIST CNN → Result
```

## Output files

| Folder | Content |
|--------|---------|
| `saved_drawings/` | `digit_2026_06_07_14_25_30.png` |
| `predictions/` | Result text + `preview_*.png` + `prediction_history.json` |
| `models/` | `mnist_digit_model.joblib` |
| `logs/` | `app.log` |

## Troubleshooting

- **Empty prediction:** Draw a clearer, larger digit before predicting.
- **Slow first start:** Training runs once; wait for *"Model saved"* in console.
- **TensorFlow errors:** This project uses **scikit-learn** instead of TensorFlow.
- **Install fails on Python 3.14:** Run `pip install scikit-learn joblib` manually if needed.
- **Voice not working:** Install `pyttsx3`; press `V` to toggle.

## License

Educational / academic project.
