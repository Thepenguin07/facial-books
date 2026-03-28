"""
Faceengine.py

Handles all computer vision operations:
- Capturing face images for registration
- Real-time face detection and recognition
- Generating 128-d face embeddings via dlib/face_recognition
"""

import cv2
import face_recognition
import numpy as np
import os
import time
from datetime import datetime


# face_data folder will be created next to this file
FACE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_data")
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# Recognition tolerance: lower = stricter (0.6 is standard)
RECOGNITION_TOLERANCE = 0.55


class FaceEngine:
    """Core facial recognition engine using dlib (via face_recognition library)."""

    def __init__(self, db_manager):
        self.db = db_manager
        self._known_encodings = []   # List of (employee_id, name, encoding)
        self.refresh_encodings()

    def refresh_encodings(self):
        """Reloads all known face encodings from the database."""
        self._known_encodings = self.db.get_all_encodings()
        print(f"[FaceEngine] Loaded {len(self._known_encodings)} face encodings.")

    def capture_face_images(self, employee_name: str, num_samples: int = 5,
                             progress_callback=None) -> list:
        """
        Opens a webcam window, captures `num_samples` face images from the user,
        and returns an averaged 128-d face encoding.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return []

        encodings_collected = []
        sample_count = 0
        last_capture_time = 0
        capture_interval = 1.0  # seconds between captures

        print(f"[FaceEngine] Capturing faces for '{employee_name}'. Please look at the camera.")

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            display = frame.copy()
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display, (left, top), (right, bottom), (0, 200, 0), 2)

            cv2.putText(display,
                        f"Samples: {sample_count}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "Press Q to cancel",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            cv2.imshow(f"Registering: {employee_name} - Press Q to cancel", display)

            current_time = time.time()
            if face_locations and current_time - last_capture_time >= capture_interval:
                face_encs = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encs:
                    encodings_collected.append(face_encs[0])
                    sample_count += 1
                    last_capture_time = current_time

                    # Save sample image
                    safe_name = "".join(c if c.isalnum() else "_" for c in employee_name)
                    img_path = os.path.join(FACE_DATA_DIR, f"{safe_name}_{sample_count}.jpg")
                    cv2.imwrite(img_path, frame)

                    if progress_callback:
                        progress_callback(sample_count, num_samples)

                    print(f"  Sample {sample_count}/{num_samples} captured.")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if not encodings_collected:
            print("[WARNING] No face encodings captured.")
            return []

        avg_encoding = np.mean(encodings_collected, axis=0)
        print(f"[FaceEngine] Registration complete. Averaged {len(encodings_collected)} samples.")
        return avg_encoding.tolist()

    def recognize_face(self, frame_rgb: np.ndarray) -> list:
        """
        Detects and identifies faces in a single RGB frame.
        Returns list of dicts with id, name, location, distance.
        """
        face_locations = face_recognition.face_locations(frame_rgb, model="hog")
        if not face_locations:
            return []

        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            emp_id = None
            min_distance = 1.0

            if self._known_encodings:
                known_encs = [e[2] for e in self._known_encodings]
                distances = face_recognition.face_distance(known_encs, encoding)
                best_idx = np.argmin(distances)
                min_distance = float(distances[best_idx])

                if min_distance < RECOGNITION_TOLERANCE:
                    emp_id, name, _ = self._known_encodings[best_idx]

            results.append({
                "id": emp_id,
                "name": name,
                "location": location,
                "distance": min_distance
            })

        return results

    def run_attendance_camera(self, on_recognize_callback, stop_event):
        """
        Runs a continuous webcam loop for attendance marking.
        Calls on_recognize_callback(employee_id, name) when a known face is detected.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        last_action: dict = {}   # {employee_id: timestamp}
        cooldown = 30  # seconds between re-triggering same employee

        print("[FaceEngine] Attendance camera started.")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recognized = self.recognize_face(rgb)

            for info in recognized:
                top, right, bottom, left = info["location"]
                name = info["name"]
                emp_id = info["id"]

                color = (0, 200, 0) if emp_id else (0, 0, 200)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if emp_id:
                    now = time.time()
                    if now - last_action.get(emp_id, 0) > cooldown:
                        last_action[emp_id] = now
                        if on_recognize_callback:
                            on_recognize_callback(emp_id, name)

            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to stop",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            cv2.imshow("FacialBooks - Attendance Camera", frame)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[FaceEngine] Attendance camera stopped.")