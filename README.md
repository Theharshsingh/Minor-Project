# Privacy-Aware Facial Emotion and Engagement Analyzer for Online Learning

A final-year Python project that estimates student engagement from webcam or
pre-recorded video **entirely on the local machine**. No raw video is stored
or transmitted by default.

> **Disclaimer** — The engagement score is a project-defined estimate of
> observable behaviour (attention direction, blink rate, facial movement,
> participation signals). It is **not** a scientifically validated measure of
> emotion, concentration, or learning outcome.

---

## Ethical & Privacy Commitments

| Principle | Implementation |
|---|---|
| Informed consent | Consent prompt shown before every session |
| Local processing | All inference runs on-device; no cloud calls |
| No raw-video storage | `store_raw_video: false` in `config.yaml` |
| Data deletion | `python main.py --delete-session <id>` |
| Fairness | Accuracy tested across lighting conditions and skin tones |
| Limitations disclosed | See *Limitations* section below |

---

## Technology Stack

| Layer | Library |
|---|---|
| Video I/O | OpenCV ≥ 4.9 |
| Face landmarks & blendshapes | MediaPipe ≥ 0.10.14 |
| Numerical analysis | NumPy ≥ 1.26 |
| Expression classifier (optional) | scikit-learn ≥ 1.4 |
| Dashboard | OpenCV HUD overlay |
| Reports | pandas + fpdf2 |

---

## Folder Structure

```
Minor-Project/
├── app/
│   ├── capture.py          # Webcam / video input
│   ├── detector.py         # MediaPipe FaceLandmarker wrapper
│   ├── analyzers/
│   │   ├── eye.py          # EAR + blink / prolonged-closure detection
│   │   ├── mouth.py        # MAR + yawn detection
│   │   ├── head_pose.py    # Head pose + attention direction
│   │   ├── expression.py   # Broad expression classification
│   │   └── participation.py
│   ├── engagement.py       # Weighted engagement score engine
│   ├── storage.py          # SQLite read/write
│   └── report.py           # CSV / PDF export
├── dashboard/
│   └── streamlit_app.py
├── models/
│   └── face_landmarker.task   # Downloaded separately (see below)
├── tests/
├── config.yaml
├── requirements.txt
└── main.py
```

---

## Installation

```bash
# 1. Clone / enter the project
cd Minor-Project

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MediaPipe Face Landmarker model
python - <<'EOF'
import requests, pathlib
url = ("https://storage.googleapis.com/mediapipe-models/"
       "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
pathlib.Path("models").mkdir(exist_ok=True)
pathlib.Path("models/face_landmarker.task").write_bytes(requests.get(url).content)
print("Model downloaded.")
EOF
```

---

## Running

```bash
# Webcam (default)
python main.py

# Pre-recorded video
python main.py path/to/video.mp4
```

Press **Q** in the OpenCV window to end the session.

---

## Milestones

| # | Feature | Status |
|---|---|---|
| 1 | Webcam input + face landmark overlay | ✅ Done |
| 2 | EAR blink detection | 🔜 Next |
| 3 | MAR yawn detection | 🔜 |
| 4 | Head pose + attention | 🔜 |
| 5 | Expression classification | 🔜 |
| 6 | Participation analysis | 🔜 |
| 7 | Engagement score | 🔜 |
| 8 | CSV / PDF reports | ✅ Done |

---

## Limitations

- Accuracy degrades in poor lighting or with partial face occlusion.
- Single-camera 2-D head-pose estimation has inherent depth ambiguity.
- Blendshape scores vary across ethnicities and age groups; fairness testing
  is ongoing.
- The engagement score reflects **observable behaviour only** and cannot
  infer internal cognitive states.
- Not suitable for high-stakes decisions (grading, hiring, surveillance).
