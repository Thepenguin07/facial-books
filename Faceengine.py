import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback

IMG_SIZE   = 96
BATCH_SIZE = 16
EPOCHS     = 30
THRESHOLD  = 0.60
MIN_PHOTOS = 20

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
NEG_DIRS   = [
    os.path.join(BASE_DIR, "kaggle_faces"),
    os.path.join(BASE_DIR, "negatives"),
]
os.makedirs(MODELS_DIR, exist_ok=True)

CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_face_region(image: np.ndarray) -> np.ndarray | None:
    """Detect largest face in BGR image, return cropped region or None."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return image[y:y+h, x:x+w]


def preprocess_face(face: np.ndarray) -> np.ndarray:
    """Resize, BGR→RGB, normalise to [0, 1]."""
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face.astype(np.float32) / 255.0


def build_model() -> tf.keras.Model:
    """Pure CNN from scratch — 3 conv blocks + classifier head."""
    from tensorflow.keras import regularizers
    reg = regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3,3), padding="same", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

class _GUIProgressCallback(Callback):
    def __init__(self, total_epochs, progress_fn):
        super().__init__()
        self._total = total_epochs
        self._prog  = progress_fn

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        val_acc = logs.get("val_accuracy", logs.get("accuracy", 0.0))
        self._prog(epoch + 1, self._total, float(val_acc))


class FaceEngine:

    def __init__(self, db_manager):
        self.db     = db_manager
        self.models = {}
        self.refresh_encodings()

    def refresh_encodings(self):
        """Reload all per-employee models from disk."""
        self.models.clear()
        for emp in self.db.get_all_employees():
            path = os.path.join(MODELS_DIR, f"{emp['id']}.h5")
            if os.path.exists(path):
                try:
                    self.models[emp["id"]] = tf.keras.models.load_model(path)
                except Exception as e:
                    print(f"[FaceEngine] Could not load model {emp['id']}: {e}")

    def capture_face_images(self, name, num_samples=50,
                            progress_callback=None, image_folder=None):
        save_dir = os.path.join(MODELS_DIR, f"train_{name.replace(' ', '_')}")
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        image_exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if image_folder and os.path.isdir(image_folder):
            all_files = []
            for root, _, files in os.walk(image_folder):
                for f in files:
                    if os.path.splitext(f)[1].lower() in image_exts:
                        all_files.append(os.path.join(root, f))

            total = len(all_files)
            for idx, fpath in enumerate(all_files):
                img = cv2.imread(fpath)
                if img is None:
                    continue
                face = detect_face_region(img)
                if face is None:
                    # if no face detected, use whole image (may already be a crop)
                    face = img
                out = os.path.join(save_dir, f"f{idx:04d}.jpg")
                cv2.imwrite(out, face)
                saved_paths.append(out)
                if progress_callback:
                    progress_callback(len(saved_paths), total or 1)
        else:
            cap   = cv2.VideoCapture(0)
            count = 0
            while count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                face  = detect_face_region(frame)
                if face is not None:
                    out = os.path.join(save_dir, f"w{count:04d}.jpg")
                    cv2.imwrite(out, face)
                    saved_paths.append(out)
                    count += 1
                    if progress_callback:
                        progress_callback(count, num_samples)
                cv2.imshow("Capturing – press ESC to stop", frame)
                if cv2.waitKey(1) == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()

        print(f"[FaceEngine] Collected {len(saved_paths)} face images for '{name}'")
        return saved_paths
    
    def load_negatives(self, limit=300):
        negatives  = []
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for neg_dir in NEG_DIRS:
            if not os.path.isdir(neg_dir):
                print(f"[FaceEngine] Negatives dir not found: {neg_dir}")
                continue
            files = [f for f in os.listdir(neg_dir)
                     if os.path.splitext(f)[1].lower() in image_exts]
            print(f"[FaceEngine] Found {len(files)} negatives in {neg_dir}")
            np.random.shuffle(files)
            for fname in files:
                if len(negatives) >= limit:
                    break
                img = cv2.imread(os.path.join(neg_dir, fname))
                if img is None:
                    continue
                face = detect_face_region(img)
                crop = face if face is not None else img
                negatives.append(preprocess_face(crop))

        print(f"[FaceEngine] Loaded {len(negatives)} negatives.")

        if 0 < len(negatives) < 80:
            augmented = []
            for face in negatives:
                augmented.append(face[:, ::-1, :])
                augmented.append(np.clip(face * 1.3, 0, 1).astype(np.float32))
                augmented.append(np.clip(face * 0.7, 0, 1).astype(np.float32))
            negatives = negatives + augmented
            print(f"[FaceEngine] Augmented negatives to {len(negatives)}.")

        return negatives

    def train_model(self, employee_id, employee_name,
                    image_paths, progress_callback=None):
        if len(image_paths) < MIN_PHOTOS:
            print(f"[FaceEngine] Too few images ({len(image_paths)}).")
            return False

        X_pos = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            # images in train folder are already face crops — just preprocess
            X_pos.append(preprocess_face(img))

        if len(X_pos) < MIN_PHOTOS:
            print(f"[FaceEngine] Too few readable positives ({len(X_pos)}).")
            return False

        X_neg = self.load_negatives(limit=max(len(X_pos) * 4, 300))
        if not X_neg:
            print("[FaceEngine] No negatives found — cannot train reliably.")
            return False

        X = np.array(X_pos + X_neg, dtype=np.float32)
        y = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=np.float32)
        idx  = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        # Stratified split — keep class balance in both train and val sets
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        pos_split = int(0.8 * len(pos_idx))
        neg_split = int(0.8 * len(neg_idx))
        tr_idx  = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
        val_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])
        np.random.shuffle(tr_idx)
        np.random.shuffle(val_idx)
        X_tr,  y_tr  = X[tr_idx],  y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        total = n_pos + n_neg
        class_weight = {
            0: total / (2.0 * n_neg) if n_neg else 1.0,
            1: total / (2.0 * n_pos) if n_pos else 1.0,
        }

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, min_delta=0.005),
        ]
        if progress_callback:
            callbacks.append(_GUIProgressCallback(EPOCHS, progress_callback))

        model = build_model()
        print(f"\n[FaceEngine] Training for '{employee_name}' "
              f"({len(X_pos)} pos / {len(X_neg)} neg)\n")

        try:
            model.fit(
                datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE),
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1,
            )
        except Exception as e:
            print(f"[FaceEngine] Training error: {e}")
            return False

        path = os.path.join(MODELS_DIR, f"{employee_id}.h5")
        model.save(path)
        self.models[employee_id] = model
        print(f"[FaceEngine] Model saved → {path}")
        return True

    def recognize_face(self, frame_rgb: np.ndarray) -> list:
        if not self.models:
            return []

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detections = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

        if len(detections) == 0:
            return []

        employees = {e["id"]: e["name"] for e in self.db.get_all_employees()}
        results   = []

        for (x, y, w, h) in detections:
            crop       = frame_bgr[y:y+h, x:x+w]
            face       = preprocess_face(crop)
            face_batch = np.expand_dims(face, axis=0)

            best_id    = None
            best_score = 0.0

            for eid, mdl in self.models.items():
                try:
                    score = float(mdl.predict(face_batch, verbose=0)[0][0])
                except Exception:
                    continue
                print(f"[DEBUG] emp={eid} score={score:.4f} threshold={THRESHOLD}")
                if score > THRESHOLD and score > best_score:
                    best_score = score
                    best_id    = eid

            results.append({
                "id":       best_id,
                "name":     employees.get(best_id, "Unknown"),
                "location": (y, x+w, y+h, x),
                "distance": round(1.0 - best_score, 4),
            })

        return results
