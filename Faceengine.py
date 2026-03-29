"""
Faceengine.py  —  FacialBooks Deep Learning Engine
=====================================================
WHAT THIS FILE DOES (for your teacher):
  This module replaces the black-box `face_recognition` library with a
  Convolutional Neural Network (CNN) that we BUILD and TRAIN ourselves
  using TensorFlow / Keras.

Deep learning concepts implemented here:
  1. Data collection   — organise face photos, detect + crop face regions
  2. Preprocessing     — resize to 64x64, normalise pixel values to [0,1]
  3. CNN architecture  — Conv2D -> MaxPool -> Flatten -> Dense layers
  4. Training loop     — binary cross-entropy loss, Adam optimiser,
                         backpropagation handled by TensorFlow automatically
  5. Inference         — load saved .h5 model, predict() on a camera frame

WHY A CNN FOR FACES?
  A plain pixel comparison fails because lighting, angle, expression change
  every image. A CNN learns a hierarchy:
    Conv Layer 1 -> edges (vertical, horizontal, diagonal lines)
    Conv Layer 2 -> shapes (curves, corners, eye contours)
    Conv Layer 3 -> face patterns (eye socket, nose bridge, jaw)
    Dense Layers -> which combination of patterns = Person X

BINARY CLASSIFICATION:
  Since we train on ONE person, we use a binary classifier:
    Class 1 = Person X  (positive samples)
    Class 0 = Not Person X  (negative samples — other faces)
  Output: single Sigmoid neuron -> value in [0.0, 1.0]
  Value > THRESHOLD (0.75) -> "it is them"
"""
import os
import cv2
import numpy as np
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ── Constants ──────────────────────────────────────────────────────────────────

IMG_SIZE      = 32
CHANNELS      = 3
BATCH_SIZE    = 32
EPOCHS        = 15
THRESHOLD     = 0.75
MIN_PHOTOS    = 20
MODELS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
NEGATIVES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negatives")
os.makedirs(MODELS_DIR,    exist_ok=True)
os.makedirs(NEGATIVES_DIR, exist_ok=True)

# Haar Cascade for face detection (built into OpenCV — NOT the face_recognition lib)
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_face_region(image_bgr: np.ndarray):
    """
    Finds the largest face in a BGR image using Haar Cascade.
    Returns a cropped BGR face patch or None.
    Used only to locate the face region — deep learning is done by the CNN.
    """
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    m  = int(0.1 * min(w, h))
    x1 = max(0, x - m);           y1 = max(0, y - m)
    x2 = min(image_bgr.shape[1], x + w + m)
    y2 = min(image_bgr.shape[0], y + h + m)
    return image_bgr[y1:y2, x1:x2]


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Converts a BGR face crop to a normalised float32 tensor.
    Steps:
      1. Resize to IMG_SIZE x IMG_SIZE  (CNN requires fixed input size)
      2. BGR -> RGB
      3. Divide by 255.0               (normalisation: [0,255] -> [0.0,1.0])
         Normalisation makes gradient descent converge faster — large raw
         pixel values would cause unstable weight updates.
    """
    resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# ── CNN Architecture ───────────────────────────────────────────────────────────

def build_cnn_model() -> tf.keras.Model:
    """
    Builds a Convolutional Neural Network for binary face classification.

    Input (64 x 64 x 3)
      |
      v  Conv2D(32 filters, 3x3 kernel) + ReLU
         Each filter slides over the image learning one pattern.
         ReLU: f(x) = max(0, x) — adds non-linearity, enables learning
         complex decision boundaries. Avoids vanishing gradient.
         Output: 62 x 62 x 32 feature maps.
      |
      v  MaxPooling2D(2x2)
         Picks max value in every 2x2 block -> halves spatial size.
         Effect: translation invariance (shifted face still recognised),
         fewer parameters, retains dominant activations only.
         Output: 31 x 31 x 32.
      |
      v  Conv2D(64, 3x3) + ReLU  -> 29x29x64
      v  MaxPooling2D(2x2)        -> 14x14x64
      |
      v  Conv2D(128, 3x3) + ReLU -> 12x12x128  (high-level face patterns)
      v  MaxPooling2D(2x2)        -> 6x6x128
      |
      v  Flatten: 6*6*128 = 4608 values -> 1D vector
      |
      v  Dropout(0.5)
         Randomly zeroes 50% of neurons during training.
         Regularisation: prevents memorising training photos (overfitting).
         Disabled at inference time automatically.
      |
      v  Dense(128) + ReLU
         Learns which combination of high-level features = Person X.
      |
      v  Dense(1) + Sigmoid
         Sigmoid: output in [0.0, 1.0]
         -> 1.0 means "confident it IS Person X"
         -> 0.0 means "confident it is NOT Person X"
         We classify as Person X when output > THRESHOLD (0.75).
    """
    return models.Sequential([
        layers.Conv2D(32,  (3,3), activation='relu', padding='valid',
                      input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name='conv1'),
        layers.MaxPooling2D((2,2), name='pool1'),
        layers.Conv2D(64,  (3,3), activation='relu', padding='valid', name='conv2'),
        layers.MaxPooling2D((2,2), name='pool2'),
        layers.Conv2D(128, (3,3), activation='relu', padding='valid', name='conv3'),
        layers.MaxPooling2D((2,2), name='pool3'),
        layers.Flatten(name='flatten'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dense(1,   activation='sigmoid', name='output'),
    ], name='FaceNet_scratch')


# ── FaceEngine ─────────────────────────────────────────────────────────────────

class FaceEngine:
    """
    Public API consumed by Application.py.

    Training flow:
      capture_face_images()  ->  collect + crop face photos
      train_model()          ->  build CNN, train from scratch, save .h5

    Inference flow:
      refresh_encodings()    ->  load saved .h5 models from disk
      recognize_face()       ->  CNN predict() on one video frame
      run_attendance_camera()->  continuous webcam loop
    """

    def __init__(self, db_manager):
        self.db = db_manager
        self._models: dict = {}    # employee_id -> tf.keras.Model
        self.refresh_encodings()

    def refresh_encodings(self):
        """Load each registered employee's saved CNN from MODELS_DIR."""
        self._models.clear()
        for emp in self.db.get_all_employees():
            eid  = emp["id"]
            path = os.path.join(MODELS_DIR, f"employee_{eid}.h5")
            if os.path.exists(path):
                try:
                    self._models[eid] = tf.keras.models.load_model(path)
                    print(f"[FaceEngine] Loaded model: {emp['name']} (ID {eid})")
                except Exception as e:
                    print(f"[FaceEngine] Load failed for {eid}: {e}")
        print(f"[FaceEngine] {len(self._models)} model(s) ready.")

    # ── Collect training images ──────────────────────────────────────────────

    def capture_face_images(self, employee_name: str, num_samples: int = 50,
                             progress_callback=None,
                             image_folder: str = None) -> list:
        """
        Gathers and saves face crops for training.
        Returns list of saved .jpg paths (positive samples).

        If image_folder is provided: reads photos from that folder.
        Otherwise: captures from webcam.
        """
        safe  = "".join(c if c.isalnum() else "_" for c in employee_name)
        pos_d = os.path.join(MODELS_DIR, f"train_{safe}", "positive")
        os.makedirs(pos_d, exist_ok=True)

        # -- Folder mode -------------------------------------------------------
        if image_folder and os.path.isdir(image_folder):
            EXT   = {".jpg",".jpeg",".png",".bmp",".webp"}
            files = [f for f in os.listdir(image_folder)
                     if os.path.splitext(f)[1].lower() in EXT]
            paths = []
            for i, fname in enumerate(files, 1):
                img  = cv2.imread(os.path.join(image_folder, fname))
                if img is None: continue
                face = detect_face_region(img)
                if face is None:
                    print(f"  [{i}/{len(files)}] No face — skip {fname}")
                    continue
                dest = os.path.join(pos_d, f"{len(paths):04d}.jpg")
                cv2.imwrite(dest, face)
                paths.append(dest)
                if progress_callback: progress_callback(len(paths), len(files))
                print(f"  [{i}/{len(files)}] OK: {fname}")
            print(f"[FaceEngine] Saved {len(paths)} face crops.")
            return paths

        # -- Webcam mode -------------------------------------------------------
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam."); return []
        captured, last_t = [], 0
        while len(captured) < num_samples:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            face  = detect_face_region(frame)
            disp  = frame.copy()
            lbl   = (f"Face OK  {len(captured)}/{num_samples}"
                     if face is not None else "No face — move closer")
            cv2.putText(disp, lbl, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,220,0) if face is not None else (50,50,220), 2)
            cv2.imshow(f"Capture: {employee_name}", disp)
            now = time.time()
            if face is not None and now - last_t >= 0.5:
                dest = os.path.join(pos_d, f"{len(captured):04d}.jpg")
                cv2.imwrite(dest, face)
                captured.append(dest); last_t = now
                if progress_callback: progress_callback(len(captured), num_samples)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        cap.release(); cv2.destroyAllWindows()
        return captured

    # ── Train CNN from scratch ──────────────────────────────────────────────

    def train_model(self, employee_id: int, employee_name: str,
                    image_paths: list, progress_callback=None) -> bool:
        """
        Builds and trains a CNN from scratch for binary face recognition.

        LOSS — Binary Cross-Entropy:
          L = -(y*log(p) + (1-y)*log(1-p))
          y=1 (true face)   -> penalises if p is low
          y=0 (other face)  -> penalises if p is high

        BACKPROPAGATION:
          TensorFlow computes dL/dW for every weight W via chain rule.
          Gradients flow from output -> Dense -> Flatten -> Conv layers.
          Weights are nudged in the direction that reduces L.

        ADAM OPTIMISER:
          Adaptive per-parameter learning rates.
          Tracks gradient mean (momentum) and variance for each weight.
          Stable training — handles the sparse, varied image gradients well.

        DATA AUGMENTATION (ImageDataGenerator):
          Each epoch shows random flips, rotations, brightness shifts.
          Effective dataset grows without collecting more photos.
          Crucial when we have few training images.
        """
        print(f"\n[FaceEngine] Training CNN for '{employee_name}' (ID={employee_id})")

        if len(image_paths) < MIN_PHOTOS:
            print(f"[ERROR] Need >= {MIN_PHOTOS} photos, got {len(image_paths)}")
            return False

        # Positive samples (Person X, label=1)
        X_pos, y_pos = [], []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None: continue
            X_pos.append(preprocess_face(img)); y_pos.append(1.0)
        print(f"  Positives: {len(X_pos)}")

        # Negative samples (other people, label=0)
        X_neg, y_neg = [], []
        neg_files = [os.path.join(NEGATIVES_DIR, f)
                     for f in os.listdir(NEGATIVES_DIR)
                     if f.lower().endswith((".jpg",".jpeg",".png"))]
        for p in neg_files[:200]:
            img  = cv2.imread(p)
            if img is None: continue
            face = detect_face_region(img)
            X_neg.append(preprocess_face(face if face is not None else img))
            y_neg.append(0.0)

        if len(X_neg) == 0:
            print("  WARNING: ./negatives/ folder is empty.")
            print("  Add photos of other people there for better accuracy.")
            print("  Using random-noise synthetic negatives as fallback.")
            for arr in X_pos:
                X_neg.append(np.random.uniform(0,1,arr.shape).astype(np.float32))
                y_neg.append(0.0)
        print(f"  Negatives: {len(X_neg)}")

        # Combine, shuffle, 80/20 split
        X   = np.array(X_pos + X_neg, dtype=np.float32)
        y   = np.array(y_pos + y_neg, dtype=np.float32)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        sp   = int(0.8 * len(X))
        X_tr, X_vl = X[:sp], X[sp:]
        y_tr, y_vl = y[:sp], y[sp:]
        print(f"  Train: {len(X_tr)}  Val: {len(X_vl)}")

        # Data augmentation
        dg = ImageDataGenerator(
            rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
            horizontal_flip=True, brightness_range=[0.7,1.3], zoom_range=0.1)
        dg.fit(X_tr)

        # Build + compile
        model = build_cnn_model()
        model.summary()
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])

        model_path = os.path.join(MODELS_DIR, f"employee_{employee_id}.h5")

        class _PCB(tf.keras.callbacks.Callback):
            def __init__(self, cb, total):
                super().__init__(); self._cb = cb; self._tot = total
            def on_epoch_end(self, epoch, logs=None):
                if self._cb:
                    self._cb(epoch+1, self._tot, logs.get('val_accuracy', 0.0))

        cbs = [
            ModelCheckpoint(model_path, monitor='val_loss',
                            save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5,
                          restore_best_weights=True, verbose=1),
        ]
        if progress_callback:
            cbs.append(_PCB(progress_callback, EPOCHS))

        print(f"\n  Fitting model — up to {EPOCHS} epochs ...\n")
        history = model.fit(
            dg.flow(X_tr, y_tr, batch_size=BATCH_SIZE),
            epochs=EPOCHS, validation_data=(X_vl, y_vl),
            callbacks=cbs, verbose=1)

        if os.path.exists(model_path):
            self._models[employee_id] = tf.keras.models.load_model(model_path)
            best = max(history.history.get('val_accuracy', [0]))
            print(f"\n  Done. Best val accuracy: {best:.1%}  |  Saved: {model_path}")
            return True

        print("[FaceEngine] Training failed — model file missing.")
        return False

    # ── Inference ───────────────────────────────────────────────────────────

    def recognize_face(self, frame_rgb: np.ndarray) -> list:
        """
        Detects faces in one RGB frame, runs each employee's CNN on them.
        Returns list of {id, name, location, distance}.

        HOW INFERENCE WORKS:
          1. Haar Cascade -> face bounding boxes
          2. preprocess_face() -> 64x64 normalised tensor
          3. model.predict(tensor) -> sigmoid score in [0,1]
          4. Highest score above THRESHOLD -> that employee is recognised
        """
        if not self._models: return []

        frame_bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
        if not len(detections): return []

        emp_names = {e["id"]: e["name"] for e in self.db.get_all_employees()}
        results   = []

        for (x, y, w, h) in detections:
            tensor = np.expand_dims(
                preprocess_face(frame_bgr[y:y+h, x:x+w]), axis=0)  # (1,64,64,3)

            best_id, best_name, best_score = None, "Unknown", 0.0
            for eid, mdl in self._models.items():
                score = float(mdl.predict(tensor, verbose=0)[0][0])
                if score > THRESHOLD and score > best_score:
                    best_score = score
                    best_id    = eid
                    best_name  = emp_names.get(eid, f"Emp {eid}")

            results.append({"id": best_id, "name": best_name,
                            "location": (y, x+w, y+h, x),
                            "distance": round(1.0 - best_score, 4)})
        return results

    # ── Webcam attendance loop ───────────────────────────────────────────────

    def run_attendance_camera(self, on_recognize_callback, stop_event):
        """
        Continuous webcam loop. Calls on_recognize_callback(emp_id, name)
        with 30-second cooldown. stop_event: threading.Event.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam."); return
        last_action: dict = {}
        print("[FaceEngine] Attendance camera started.")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for info in self.recognize_face(rgb):
                t, r, b, l = info["location"]
                name, eid  = info["name"], info["id"]
                col = (0,200,0) if eid else (0,0,200)
                cv2.rectangle(frame, (l,t), (r,b), col, 2)
                cv2.putText(frame, name, (l, t-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                if eid:
                    now = time.time()
                    if now - last_action.get(eid, 0) > 30:
                        last_action[eid] = now
                        if on_recognize_callback:
                            on_recognize_callback(eid, name)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("FacialBooks — Attendance Camera", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                stop_event.set(); break
        cap.release(); cv2.destroyAllWindows()
        print("[FaceEngine] Camera stopped.")
