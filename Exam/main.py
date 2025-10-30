from ultralytics import YOLO
import supervision as sv
import cv2

model = YOLO("best50epochs.pt")
box_annotator = sv.BoxAnnotator()

# Map original class IDs to your custom labels
custom_labels = {2: "Phone", 3: "Cheating"}  # Use your YOLO class IDs

cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Overwrite the class_name data in detections
    detections.data["class_name"] = [custom_labels[int(cls)] for cls in detections.class_id]

    annotated_frame = box_annotator.annotate(frame, detections=detections)

    cv2.imshow("Webcam", annotated_frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
