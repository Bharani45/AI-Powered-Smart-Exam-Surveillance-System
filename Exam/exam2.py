from ultralytics import YOLO
import cv2

# Load model
model = YOLO("talking.pt")

# Open video file
cap = cv2.VideoCapture("demo3.mp4")

# Get video details
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("talk_output.mp4", fourcc, fps, (width, height))

# Custom names
custom_names = {
    0: "class0",
    1: "talking",
    2: "not talking",
}

# Different confidence thresholds for each class
class_conf_thresholds = {
    0: 0.9,   # class0
    1: 0.30,   # class1
    2: 0.30,  # phone#cheating
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # run inference with low global threshold
    results = model(frame, conf=0.1)

    # override names
    results[0].names = custom_names

    # get boxes
    boxes = results[0].boxes  # tensor with xyxy, conf, cls

    # filter by per-class thresholds
    keep_idx = []
    for i, (conf, cls) in enumerate(zip(boxes.conf, boxes.cls)):
        cls = int(cls.item())
        conf = conf.item()
        if conf >= class_conf_thresholds.get(cls, 0.25):
            keep_idx.append(i)

    # keep only filtered detections
    results[0].boxes = boxes[keep_idx]

    # get annotated frame
    annotated = results[0].plot()

    out.write(annotated)
    cv2.imshow("Cheating Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
