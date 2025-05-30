import cv2
import os
import yaml
from ultralytics import YOLO
from collections import defaultdict

# === CONFIGURATION ===
MODEL_PATH = "my_model.pt"
VIDEO_PATH = "conveyor.mp4"
OUTPUT_VIDEO_PATH = "output_conveyor.mp4"
DATA_YAML_PATH = "data.yaml"
MIN_CONFIDENCE = 0.3
TOLERANCE = 5
FRAME_SKIP = 1

# === LOAD CLASS NAMES ===
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# === COUNT OBJECTS WITH TRACKING ===
def count_objects_in_video(video_path, model, class_names, counting_line_y=None, output_path=None, tolerance=5, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video: {frame_width}x{frame_height}, {fps} FPS")

    if counting_line_y is None:
        counting_line_y = int(frame_height * 0.4)
    print(f"Counting line at y={counting_line_y}")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    object_counts = defaultdict(int)
    total_count = 0
    counted_ids = set()
    frame_count = 0

    try:
        results = model.track(source=video_path, stream=True, persist=True, conf=MIN_CONFIDENCE, iou=0.7, tracker="bytetrack.yaml")
    except Exception as e:
        print(f"Tracking failed: {e}. Falling back to prediction mode.")
        results = model.predict(source=video_path, stream=True, conf=MIN_CONFIDENCE, iou=0.7)

    for result in results:
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = result.orig_img
        if not result.boxes:
            if out:
                out.write(frame)
            continue

        for box in result.boxes:
            class_id = int(box.cls)
            class_name = class_names[class_id]
            confidence = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            object_id = int(box.id.item()) if hasattr(box, 'id') and box.id is not None else f"{class_name}_{center_x}_{center_y}_{frame_count}"

            if (counting_line_y - tolerance <= center_y <= counting_line_y + tolerance and object_id not in counted_ids):
                object_counts[class_name] += 1
                total_count += 1
                counted_ids.add(object_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.line(frame, (0, counting_line_y), (frame_width, counting_line_y), (0, 0, 255), 2)
        y_text = 30
        cv2.putText(frame, f"Total Objects: {total_count}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for class_name, count in object_counts.items():
            y_text += 30
            cv2.putText(frame, f"{class_name}: {count}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if out:
            out.write(frame)

    cap.release()
    if out:
        out.release()

    print(f"\nTotal Objects Detected: {total_count}")
    print("Counts by Class:")
    for class_name, count in object_counts.items():
        print(f"{class_name}: {count}")

    return object_counts, total_count

if __name__ == "__main__":
    class_names = load_class_names(DATA_YAML_PATH)
    model = YOLO(MODEL_PATH)

    object_counts, total_count = count_objects_in_video(
        video_path=VIDEO_PATH,
        model=model,
        class_names=class_names,
        counting_line_y=None,
        output_path=OUTPUT_VIDEO_PATH,
        tolerance=TOLERANCE,
        frame_skip=FRAME_SKIP
    )
