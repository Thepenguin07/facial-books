# FacialBooks — Automated Attendance & Payroll System

A deep learning based employee attendance system that uses a CNN trained from scratch to recognize faces via webcam and automatically log entry/exit times with payroll generation.

---

## What It Does

- Detects and recognizes employee faces in real time using a webcam
- Automatically logs entry and exit times when a face is recognized
- Generates monthly attendance reports and salary records
- Exports data to Excel and CSV
- Full GUI built with CustomTkinter

---

## Project Structure

```
SHIFA-PYTHON/
│
├── MAINdl.py              # Entry point — starts the app
├── Application.py         # GUI (CustomTkinter) — all 4 tabs
├── Faceengine.py          # CNN model — training & face recognition
├── Dbmanager.py           # SQLite database — employees, attendance, payroll
├── Reportgenerator.py     # Excel/CSV report generation
├── augment_faces.py       # Script to augment training photos
│
├── Shifaface/             # Training photos (positive samples)
├── negatives/             # Other people's photos (negative samples)
├── models/                # Saved .h5 model files per employee
│
├── Requirements.txt       # Python dependencies
└── Readme.md              # This file
```

---

## How the CNN Works

This project trains a **binary CNN classifier from scratch** for each employee.

### Architecture
- 3 Convolutional blocks (32 → 64 → 128 filters)
- Each block: Conv2D → BatchNormalization → ReLU → MaxPooling → Dropout
- GlobalAveragePooling → Dense(256) → Dropout → Dense(1, sigmoid)

### Training
- Positive samples: photos of the employee (face crops, 96×96 px)
- Negative samples: photos of other people from the `negatives/` folder
- Data augmentation: rotation, zoom, flip, shift
- Loss function: Binary Crossentropy
- Optimizer: Adam (lr=5e-4)
- EarlyStopping on val_loss with patience=6

### Recognition
- Haar Cascade detects face regions in each camera frame
- Detected face is preprocessed and passed through the CNN
- If confidence score > 0.60 → employee is recognized
- 60-second cooldown prevents duplicate attendance logs

---

## Setup & Installation

### 1. Clone / download the project

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r Requirements.txt
```

### 4. Run the app
```bash
python MAINdl.py
```

---

## Registering an Employee

1. Go to the **Employees** tab
2. Fill in Name, Department, Hourly Rate
3. Click **Browse** and select a folder with 50–100 face photos of the person
4. Click **1. Capture / Load Faces**
5. Click **2. Train CNN & Register**
6. Wait for training to complete (2–5 minutes)

### Tips for better accuracy
- Use clear, well-lit front-facing photos
- Include variety — different angles, with/without glasses, different lighting
- Add at least 50–80 photos of other people to the `negatives/` folder

---

## Generating Augmented Training Data

If you have fewer photos, run the augmentation script to generate more:

```bash
python augment_faces.py
```

This reads photos from `Shifaface/` and generates 250 augmented versions (rotations, flips, brightness changes, zoom, translations).

---

## Attendance & Payroll

- **Attendance tab** — view and filter attendance logs by month/employee, export to Excel
- **Payroll tab** — generate monthly salary (hours × hourly rate), export to Excel/CSV
- **Manual Override** — log entry/exit manually from the Dashboard sidebar

---

## Dependencies

| Library | Purpose |
|---|---|
| tensorflow | CNN training and inference |
| opencv-python | Camera feed and face detection |
| customtkinter | Modern dark-themed GUI |
| Pillow | Image display in GUI |
| openpyxl | Excel report generation |
| numpy | Array operations |

---

## Known Limitations

- One CNN model per employee — does not scale well beyond 10–15 employees on CPU
- Haar Cascade face detector can miss faces at extreme angles or in poor lighting
- Model confidence (~60%) is moderate — may occasionally misidentify in similar lighting conditions
- Training requires at least 20 positive face images and a populated `negatives/` folder

---

## Tech Stack

- Python 3.10
- TensorFlow / Keras
- OpenCV
- CustomTkinter
- SQLite

