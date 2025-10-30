import cv2
import os
import dlib
import numpy as np
import logging
import time

# Use OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dlib for face landmarks and recognition model
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        self.current_face_dir = ""
        self.cap = cv2.VideoCapture(0)

    def pre_work_mkdir(self):
        if not os.path.isdir(self.path_photos_from_camera):
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        if os.listdir(self.path_photos_from_camera):
            person_list = os.listdir(self.path_photos_from_camera)
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.existing_faces_cnt = max(person_num_list)
        else:
            self.existing_faces_cnt = 0

    def create_face_folder(self, name):
        self.existing_faces_cnt += 1
        self.current_face_dir = os.path.join(self.path_photos_from_camera, f"person_{self.existing_faces_cnt}_{name}")
        os.makedirs(self.current_face_dir)
        logging.info(f"Created folder: {self.current_face_dir}")
        self.ss_cnt = 0

    def save_current_face(self, frame, face_rect):
        x, y, w, h = face_rect
        face_img = frame[y:y+h, x:x+w]
        self.ss_cnt += 1
        save_path = os.path.join(self.current_face_dir, f"img_face_{self.ss_cnt}.jpg")
        cv2.imwrite(save_path, face_img)
        logging.info(f"Saved: {save_path}")

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()

        person_name = input("Enter the name of the person: ")
        self.create_face_folder(person_name)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Register Face", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                if len(faces) == 1:
                    self.save_current_face(frame, faces[0])
                else:
                    print("No face or multiple faces detected. Please ensure only one face is in the frame.")
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()

if __name__ == '__main__':
    main()