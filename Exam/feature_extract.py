import os
import cv2
import dlib
import numpy as np
import csv

# Paths
STUDENTS_FOLDER = "students"  # Folder with student subfolders
SHAPE_PREDICTOR_PATH = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
FACE_RECO_MODEL_PATH = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
OUTPUT_CSV = "features_all.csv"

# Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_RECO_MODEL_PATH)

def extract_features(image_path):
    img = cv2.imread(image_path)
    faces = detector(img, 1)
    if len(faces) == 0:
        return None
    shape = predictor(img, faces[0])
    return np.array(face_rec_model.compute_face_descriptor(img, shape))

def main():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        for student_name in os.listdir(STUDENTS_FOLDER):
            student_folder = os.path.join(STUDENTS_FOLDER, student_name)
            if not os.path.isdir(student_folder):
                continue

            features_list = []
            for file in os.listdir(student_folder):
                if file.lower().endswith((".jpg", ".png")):
                    img_path = os.path.join(student_folder, file)
                    features = extract_features(img_path)
                    if features is not None:
                        features_list.append(features)

            if features_list:
                mean_features = np.mean(features_list, axis=0)
                writer.writerow([student_name] + mean_features.tolist())

    print("Feature extraction completed and saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
