import os
import dlib
import csv
import numpy as np
import logging
import cv2

path_images_from_camera = "data/data_faces_from_camera/"

# Use dlib for shape prediction and face recognition
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)

    # We are assuming one face per image for the registered photos
    faces = [dlib.rectangle(0, 0, img_gray.shape[1], img_gray.shape[0])]

    logging.info(f"Processing image: {path_img}")

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("No face detected in the saved image.")
    return face_descriptor


def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for photo in photos_list:
            features_128d = return_128d_features(os.path.join(path_face_personX, photo))
            if features_128d != 0:
                features_list_personX.append(features_128d)
    else:
        logging.warning(f"Warning: No images in {path_face_personX}")

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX


def main():
    logging.basicConfig(level=logging.INFO)
    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            logging.info(f"Processing person: {person}")
            features_mean_personX = return_features_mean_personX(os.path.join(path_images_from_camera, person))

            try:
                person_name = person.split('_', 2)[-1]
            except IndexError:
                person_name = person

            features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
            writer.writerow(features_mean_personX)
        logging.info("Saved all features to data/features_all.csv")


if __name__ == '__main__':
    main()