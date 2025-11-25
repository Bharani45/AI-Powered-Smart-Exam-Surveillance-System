from ultralytics import YOLO
import cv2
import face_recognition
import os
import numpy as np
import smtplib
from email.message import EmailMessage
from datetime import datetime


class CheatingDetector:
    """
    Class wrapper around your original script.
    Usage:
        detector = CheatingDetector()
        detector.run()
    """

    def __init__(self,
                 subject_name=None,
                 model_filename="best50epochs-high.pt",
                 video_filename="test_video.mp4",
                 base_students_folder=os.path.join("Attendance", "students"),
                 output_filename="sidecam_output_identified_alert.mp4"):
        # preserve your email config exactly
        self.SENDER_EMAIL = "bharanimxie@gmail.com"
        self.SENDER_PASSWORD = "tbbt dasy dvql fhzc"
        self.RECEIVER_EMAIL = "bharanimxib@gmail.com"

        # resolve paths relative to this file so running from different cwd won't break things
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_filename)
        self.video_path = os.path.join(base_dir, video_filename)
        self.output_path = os.path.join(base_dir, output_filename)

        # subject name + folder path
        self.subject_name = subject_name.strip().capitalize() if subject_name else "General"
        self.students_folder = os.path.join(base_dir, base_students_folder, self.subject_name)

        if not os.path.exists(self.students_folder):
            raise FileNotFoundError(
                f"⚠️ The subject folder '{self.subject_name}' was not found at:\n{self.students_folder}"
            )

        print(f"📚 Loading student data for subject: {self.subject_name}")

        # load model and encodings (keeps same behavior)
        self.model = YOLO(self.model_path)
        self.known_face_encodings, self.known_face_names = self.load_student_encodings(self.students_folder)

        # same mapping & thresholds
        self.custom_names = {0: "class0", 1: "class1", 2: "phone", 3: "cheating"}
        self.class_conf_thresholds = {0: 0.9, 1: 0.9, 2: 0.30, 3: 0.50}

        # spam-prevention set (same logic)
        self.reported_infractions = set()

    def send_email_alert(self, student_name, infraction_type, frame_image):
        """Sends an email with an attached image of the incident."""
        print(f"Preparing to send email alert for {student_name}...")

        msg = EmailMessage()
        msg['Subject'] = f"🚨 {infraction_type.capitalize()} Alert: {student_name} Detected"
        msg['From'] = self.SENDER_EMAIL
        msg['To'] = self.RECEIVER_EMAIL

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
        Hello,

        An automated alert has been triggered.

        Student: {student_name}
        Infraction: {infraction_type.capitalize()}
        Timestamp: {timestamp}

        An image of the incident is attached.

        - Automated Detection System
        """
        msg.set_content(body)

        _, buffer = cv2.imencode('.jpg', frame_image)
        image_bytes = buffer.tobytes()
        msg.add_attachment(image_bytes, maintype='image', subtype='jpeg', filename=f'{student_name}_incident.jpg')

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.SENDER_EMAIL, self.SENDER_PASSWORD)
                smtp.send_message(msg)
            print(f"✅ Email alert for {student_name} sent successfully to {self.RECEIVER_EMAIL}!")
        except Exception as e:
            print(f"❌ FAILED to send email. Error: {e}")

    def load_student_encodings(self, base_folder_path):
        """
        Load all student images from subfolders and compute encodings.
        Returns (encodings_list, names_list)
        """
        student_encodings = []
        student_names = []
        print("Loading ALL student images from sub-folders for higher accuracy...")

        if not os.path.exists(base_folder_path):
            print(f"[ERROR] The directory '{base_folder_path}' does not exist. Please create it.")
            return [], []

        for student_name in os.listdir(base_folder_path):
            student_folder_path = os.path.join(base_folder_path, student_name)
            if os.path.isdir(student_folder_path):
                encodings_for_student_count = 0
                for filename in os.listdir(student_folder_path):
                    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                        image_path = os.path.join(student_folder_path, filename)
                        try:
                            image = face_recognition.load_image_file(image_path)
                            current_encodings = face_recognition.face_encodings(image)
                            for encoding in current_encodings:
                                student_encodings.append(encoding)
                                student_names.append(student_name)
                                encodings_for_student_count += 1
                        except Exception as e:
                            print(f"  - ERROR processing {image_path}: {e}")
                if encodings_for_student_count > 0:
                    print(f"  - Successfully encoded {encodings_for_student_count} images/faces for student: {student_name}")
                else:
                    print(f"  - FAILED: Could not find or encode any faces for student '{student_name}'.")

        print("Student encoding complete.")
        return student_encodings, student_names

    def run(self, conf_override=0.1):
        """
        Run the detection loop (same logic as your original script).
        conf_override: float, the confidence passed to model(...) (default 0.1 as original).
        """
        # open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {self.video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=conf_override)
            annotated_frame = frame.copy()
            boxes = results[0].boxes

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()

                if conf >= self.class_conf_thresholds.get(cls_id, 0.25):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = self.custom_names.get(cls_id, "Unknown")
                    label = f"{class_name} {conf:.2f}"
                    color = (0, 255, 0)

                    if class_name == "phone":
                        frame_h, frame_w, _ = frame.shape
                        box_w, box_h = x2 - x1, y2 - y1
                        expand_factor_up, expand_factor_sides = 3.0, 1.5
                        y1, x1 = y1 - int(box_h * expand_factor_up), x1 - int(box_w * expand_factor_sides)
                        x2, y2 = x2 + int(box_w * expand_factor_sides), y2 + int(box_h * 0.2)
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)

                    if class_name in ["phone", "cheating"]:
                        color = (0, 0, 255)
                        person_roi = frame[y1:y2, x1:x2]

                        if person_roi.size > 0:
                            rgb_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            roi_face_encodings = face_recognition.face_encodings(rgb_person_roi)

                            if roi_face_encodings and self.known_face_encodings:
                                face_to_check = roi_face_encodings[0]
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_to_check)
                                best_match_index = np.argmin(face_distances)

                                tolerance = 0.6
                                if face_distances[best_match_index] < tolerance:
                                    identified_name = self.known_face_names[best_match_index]
                                else:
                                    identified_name = "Unknown"

                                label = f"{identified_name} - {class_name.upper()}!"
                                alert_message = "Phone use detected!" if class_name == "phone" else "Cheating detected!"
                                print(f"🚨 ALERT: {alert_message} Person: {identified_name}")

                                # Create a unique key for the student AND the infraction type
                                infraction_key = (identified_name, class_name)

                                # Check if this specific infraction has been reported
                                if identified_name != "Unknown" and infraction_key not in self.reported_infractions:
                                    self.send_email_alert(identified_name, class_name, annotated_frame)
                                    # Add the specific infraction key to the set
                                    self.reported_infractions.add(infraction_key)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(annotated_frame)
            cv2.imshow("Cheating Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Processing finished and video saved.")
if __name__ == "__main__":
    print("🎓 Real-time Cheating Detection System")
    subject_name = input("Enter the subject name: ").strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    subject_folder = os.path.join(base_dir, "Attendance", "students", subject_name.capitalize())

    if not os.path.exists(subject_folder):
        print(f"⚠️ The subject folder '{subject_name}' was not found at:\n{subject_folder}")
        exit(1)

    # ✅ Pass only subject_name
    detector = CheatingDetector(subject_name=subject_name)

    # Run detection
    detector.run()


