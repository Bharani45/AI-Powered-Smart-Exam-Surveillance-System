from ultralytics import YOLO
import cv2
import face_recognition
import os
import numpy as np
import smtplib
from email.message import EmailMessage
from datetime import datetime

# =============================================================================
# ## SECTION 1: EMAIL CONFIGURATION (NO CHANGES) ##
# =============================================================================
SENDER_EMAIL = "bharanimxie@gmail.com"  # <--- ENTER YOUR EMAIL HERE

SENDER_PASSWORD = "tbbt dasy dvql fhzc"  # <--- ENTER YOUR APP PASSWORD HERE

RECEIVER_EMAIL = "bharanimxib@gmail.com"


def send_email_alert(student_name, infraction_type, frame_image):
    """Sends an email with an attached image of the incident."""
    print(f"Preparing to send email alert for {student_name}...")

    msg = EmailMessage()
    msg['Subject'] = f"🚨 {infraction_type.capitalize()} Alert: {student_name} Detected"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

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
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ Email alert for {student_name} sent successfully to {RECEIVER_EMAIL}!")
    except Exception as e:
        print(f"❌ FAILED to send email. Error: {e}")


# =============================================================================
# ## STUDENT ENCODING (NO CHANGES) ##
# =============================================================================
STUDENTS_FOLDER = "students"


def load_student_encodings(base_folder_path):
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
                print(
                    f"  - Successfully encoded {encodings_for_student_count} images/faces for student: {student_name}")
            else:
                print(f"  - FAILED: Could not find or encode any faces for student '{student_name}'.")

    print("Student encoding complete.")
    return student_encodings, student_names


known_face_encodings, known_face_names = load_student_encodings(STUDENTS_FOLDER)

# =============================================================================
# ## YOLO AND VIDEO SETUP (NO CHANGES) ##
# =============================================================================
model = YOLO("best50epochs-high.pt")
cap = cv2.VideoCapture("test_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("sidecam_output_identified_alert.mp4", fourcc, fps, (width, height))
custom_names = {0: "class0", 1: "class1", 2: "phone", 3: "cheating"}
class_conf_thresholds = {0: 0.9, 1: 0.9, 2: 0.30, 3: 0.50}

# =============================================================================
# ## MODIFIED SECTION: SPAM PREVENTION SETUP ##
# =============================================================================
# Renamed for clarity. This will now store tuples like ('StudentName', 'phone')
reported_infractions = set()

# =============================================================================
# ## MAIN VIDEO PROCESSING LOOP (MODIFIED) ##
# =============================================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.1)
    annotated_frame = frame.copy()
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()

        if conf >= class_conf_thresholds.get(cls_id, 0.25):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = custom_names.get(cls_id, "Unknown")
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

                    if roi_face_encodings and known_face_encodings:
                        face_to_check = roi_face_encodings[0]
                        face_distances = face_recognition.face_distance(known_face_encodings, face_to_check)
                        best_match_index = np.argmin(face_distances)

                        tolerance = 0.6
                        if face_distances[best_match_index] < tolerance:
                            identified_name = known_face_names[best_match_index]
                        else:
                            identified_name = "Unknown"

                        label = f"{identified_name} - {class_name.upper()}!"
                        alert_message = "Phone use detected!" if class_name == "phone" else "Cheating detected!"
                        print(f"🚨 ALERT: {alert_message} Person: {identified_name}")

                        # ===============================================================
                        # ## MODIFIED SECTION: EMAIL TRIGGER LOGIC ##
                        # ===============================================================
                        # Create a unique key for the student AND the infraction type
                        infraction_key = (identified_name, class_name)

                        # Check if this specific infraction has been reported
                        if identified_name != "Unknown" and infraction_key not in reported_infractions:
                            send_email_alert(identified_name, class_name, annotated_frame)
                            # Add the specific infraction key to the set
                            reported_infractions.add(infraction_key)
                        # ===============================================================

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