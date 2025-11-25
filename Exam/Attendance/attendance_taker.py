import dlib
import numpy as np
import cv2
import os
import pandas as pd
import sqlite3
import datetime
import logging


class FaceRecognizer:
    def __init__(self, subject_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.subject_name = subject_name.strip().capitalize()

        # === Folder setup ===
        self.students_base = os.path.join(base_dir, "students")
        self.subject_folder = os.path.join(self.students_base, self.subject_name)

        if not os.path.exists(self.subject_folder):
            raise FileNotFoundError(f"⚠️ Subject folder '{self.subject_name}' not found in: {self.subject_folder}")

        # === Model Paths ===
        predictor_path = os.path.join(base_dir, 'data', 'data_dlib', 'shape_predictor_68_face_landmarks.dat')
        reco_model_path = os.path.join(base_dir, 'data', 'data_dlib', 'dlib_face_recognition_resnet_model_v1.dat')
        db_path = os.path.join(base_dir, 'attendance.db')

        # === Load dlib models ===
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_reco_model = dlib.face_recognition_model_v1(reco_model_path)

        # === Setup database ===
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                name TEXT,
                time TEXT,
                date DATE,
                subject TEXT,
                UNIQUE(name, date, subject)
            )
        ''')
        self.conn.commit()
        self.conn.close()

        # === Load known faces ===
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_faces_from_folder()

        logging.info(f"✅ Initialized dlib face recognizer for subject: {self.subject_name}")

    # -------------------------------------------------------------------------
    def load_faces_from_folder(self):
        """Load face encodings for all students in the subject folder."""
        print(f"📁 Loading student faces from {self.subject_folder}")
        for student_name in os.listdir(self.subject_folder):
            student_path = os.path.join(self.subject_folder, student_name)
            if not os.path.isdir(student_path):
                continue

            for img_file in os.listdir(student_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(student_path, img_file)
                    img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    dets = self.detector(rgb_img, 1)

                    if len(dets) > 0:
                        shape = self.predictor(rgb_img, dets[0])
                        face_descriptor = self.face_reco_model.compute_face_descriptor(rgb_img, shape)
                        self.face_features_known_list.append(np.array(face_descriptor))
                        self.face_name_known_list.append(student_name)
        print(f"✅ Loaded {len(self.face_name_known_list)} faces for subject {self.subject_name}")

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_distance(descriptor1, descriptor2):
        return np.linalg.norm(descriptor1 - descriptor2)

    # -------------------------------------------------------------------------
    def mark_attendance(self, name):
        """Insert attendance entry if not already marked."""
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attendance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        cursor.execute(
            "SELECT * FROM attendance WHERE name=? AND date=? AND subject=?",
            (name, current_date, self.subject_name)
        )
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(
                "INSERT INTO attendance (name, time, date, subject) VALUES (?, ?, ?, ?)",
                (name, current_time, current_date, self.subject_name)
            )
            conn.commit()
            print(f"✅ Attendance marked for {name} ({self.subject_name}) at {current_time}")
        conn.close()

    # -------------------------------------------------------------------------
    def start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("⚠️ Cannot access camera.")
        return cap

    def stop_camera(self, cap):
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    def process_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.detector(rgb_frame, 1)

        success_message = None

        for det in dets:
            shape = self.predictor(rgb_frame, det)
            face_descriptor = np.array(self.face_reco_model.compute_face_descriptor(rgb_frame, shape))

            distances = [self.compute_distance(face_descriptor, known) for known in self.face_features_known_list]
            name = "Unknown"

            if distances:
                min_idx = np.argmin(distances)
                if distances[min_idx] < 0.45:  # threshold tuning
                    name = self.face_name_known_list[min_idx]
                    self.mark_attendance(name)
                    success_message = f"✅ {name} marked present!"

            x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show message if marked
        if success_message:
            cv2.putText(frame, success_message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

        cv2.imshow(f"Attendance - {self.subject_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_camera(cap)
            return "quit"


# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎓 DLIB Attendance System (Subject Specific)")
    subject = input("Enter the subject name for attendance: ").strip()
    recog = FaceRecognizer(subject)
    cap = recog.start_camera()

    print(f"📸 Starting attendance for {subject}... (Press 'Q' to quit)")
    while True:
        result = recog.process_frame(cap)
        if result == "quit":
            print(f"✅ Attendance session for '{subject}' ended.")
            break
