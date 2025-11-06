"""
yolo_webcam.py
Real-time webcam demo using Ultralytics YOLO (e.g. YOLOv8).
Author: (you)
Requirements: ultralytics, opencv-python

Usage:
    python yolo_webcam.py
Optional args (edit in code or extend with argparse):
    MODEL = "yolov8n.pt"  # lightweight model
    CONFIDENCE_THRESHOLD = 0.25
    WEBCAM_ID = 0
"""

import time
import cv2
from ultralytics import YOLO

# ---------- CONFIG ----------
MODEL = "yolov8n.pt"          # change to yolov8s.pt, yolov8m.pt, or path to your fine-tuned model
WEBCAM_ID = 0                # camera index (0 is default)
CONFIDENCE_THRESHOLD = 0.25  # minimum confidence to show a detection
WINDOW_NAME = "YOLO Webcam"
# ----------------------------

def draw_boxes(frame, boxes, confidences, classes, names):
    """
    Draw bounding boxes and labels on the frame.
    boxes: list of [x1, y1, x2, y2]
    confidences: list of floats
    classes: list of int class ids
    names: dict or list mapping class id -> name
    """
    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {conf:.2f}"
        # box color (B, G, R) chosen by class id
        color = tuple(int(c) for c in ((cls * 37) % 255, (cls * 73) % 255, (cls * 151) % 255))
        # rectangle and filled label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def main():
    # Load model
    print(f"Loading model {MODEL} ...")
    model = YOLO(MODEL)  # loads model (will download if not available)
    names = model.names if hasattr(model, "names") else {}
    print("Model loaded. Classes:", names)

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check WEBCAM_ID.")
        return

    # Set a smaller frame size for speed (optional)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0.0
    fps_smooth = 0.0
    print("Starting webcam stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed reading frame from webcam.")
            break

        # Optionally resize for speed (uncomment to use)
        # frame = cv2.resize(frame, (640, 480))

        # Run inference (Ultralytics API accepts numpy arrays)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)  # set conf filter here

        # results is a Results object; usually results[0] for the frame
        r = results[0]
        boxes = []
        confidences = []
        classes = []

        # Extract boxes safely
        if hasattr(r, "boxes") and r.boxes is not None:
            # r.boxes.xyxy -> tensor Nx4
            try:
                xyxy = r.boxes.xyxy.cpu().numpy()  # may need .cpu() if on GPU
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()
            except Exception:
                # fallback to .tolist()
                xyxy = r.boxes.xyxy.tolist()
                confs = r.boxes.conf.tolist()
                cls_ids = r.boxes.cls.tolist()

            for b, c, cl in zip(xyxy, confs, cls_ids):
                # apply confidence threshold (already applied by model, but safe)
                if float(c) >= CONFIDENCE_THRESHOLD:
                    boxes.append(b)
                    confidences.append(float(c))
                    classes.append(int(cl))

        # Draw detections
        draw_boxes(frame, boxes, confidences, classes, names)

        # Compute FPS
        curr_time = time.time()
        dt = curr_time - prev_time if prev_time > 0 else 0.0
        prev_time = curr_time
        fps = 1.0 / dt if dt > 0 else 0.0
        # Smooth FPS
        fps_smooth = fps_smooth * 0.9 + fps * 0.1 if fps_smooth else fps

        # Show FPS on frame
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display
        cv2.imshow(WINDOW_NAME, frame)

        # Quit with 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
