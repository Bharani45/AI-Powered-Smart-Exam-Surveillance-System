import os
import dlib
import csv
import numpy as np
import logging
import cv2

# === Folder structure ===
# students/
# ├── Math/
# │   ├── John/
# │   │   ├── 1.jpg
# │   │   └── 2.jpg
# │   ├── Alice/
# │   │   ├── 1.jpg
# │   │   └── 2.jpg
# ├── Physics/
# │   ├── John/
# │   └── Bob/

base_dir = os.path.dirname(os.path.abspath(__file__))
path_students_base = os.path.join(base_dir, "students")  # main folder containing subjects
path_data = os.path.join(base_dir, "data")
path_dlib = os.path.join(path_data, "data_dlib")

# === Load models ===
predictor = dlib.shape_predictor(os.path.join(path_dlib, 'shape_predictor_68_face_landmarks.dat'))
face_reco_model = dlib.face_recognition_model_v1(
    os.path.join(path_dlib, 'dlib_face_recognition_resnet_model_v1.dat')
)


def return_128d_features(path_img):
    """Return 128D face descriptor for a given image."""
    img_rd = cv2.imread(path_img)
    if img_rd is None:
        logging.warning(f"⚠️ Unable to read image: {path_img}")
        return 0

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img_gray, 1)

    if len(faces) == 0:
        logging.warning(f"⚠️ No face detected in {path_img}")
        return 0

    shape = predictor(img_rd, faces[0])
    face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    return face_descriptor


def return_features_mean_student(student_folder_path):
    """Return average 128D feature vector for all images of a student."""
    features_list = []
    photos_list = [f for f in os.listdir(student_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for photo in photos_list:
        feature_128d = return_128d_features(os.path.join(student_folder_path, photo))
        if feature_128d != 0:
            features_list.append(feature_128d)

    if len(features_list) == 0:
        logging.warning(f"⚠️ No valid faces found in {student_folder_path}")
        return np.zeros(128, dtype=float)
    else:
        return np.mean(features_list, axis=0)


def main():
    logging.basicConfig(level=logging.INFO)
    subjects = [s for s in os.listdir(path_students_base) if os.path.isdir(os.path.join(path_students_base, s))]

    for subject in subjects:
        subject_path = os.path.join(path_students_base, subject)
        students = [s for s in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, s))]
        features_csv_path = os.path.join(path_data, f"features_{subject}.csv")

        logging.info(f"📘 Processing subject: {subject} ({len(students)} students)")
        with open(features_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for student in students:
                student_folder = os.path.join(subject_path, student)
                logging.info(f"👤 Processing student: {student}")
                features_mean = return_features_mean_student(student_folder)

                # ✅ FIX: convert to list and prepend name
                row = [student] + features_mean.tolist()
                writer.writerow(row)
            logging.info(f"✅ Saved features to {features_csv_path}")

    logging.info("🎯 Feature extraction completed for all subjects.")


if __name__ == "__main__":
    main()
