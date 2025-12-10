import cv2
from ultralytics import YOLO
import threading
import requests
import time

'''
curl "http://<ESP_IP>/led?label=paper"
curl "http://<ESP_IP>/led?label=plastic"
curl "http://<ESP_IP>/led?label=glass"
curl "http://<ESP_IP>/led?label=metal"
curl "http://<ESP_IP>/led?label=off"
'''

ESP32_IP = ""  
URL = f"http://{ESP32_IP}:81/stream"

CLASS_TO_LED = {
    "paper": "paper",
    "plastic": "plastic",
    "glass": "glass",
    "metal": "metal",
}
PERSISTENCE_FRAMES = 3
OFF_AFTER_NO_DETECTION_FRAMES = 10

model = YOLO("best.pt")

#cap = cv2.VideoCapture(URL)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream. Check ESP32 IP and WiFi.")
    exit()

cv2.namedWindow("ESP32-S3 YOLO Inference", cv2.WINDOW_NORMAL)


def send_label_to_esp(label: str):
    if label is None:
        return

    try:
        if label == 'off':
            url = f"http://{ESP32_IP}:80/led?label=off"
        else:
            mapped = CLASS_TO_LED.get(label, None)
            if mapped is None:
                return
            url = f"http://{ESP32_IP}:80/led?label={mapped}"
        resp = requests.get(url, timeout=0.5)
    except requests.RequestException:
        pass


def most_confident_label(result):
    try:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs = boxes.conf
        cls_idxs = boxes.cls

        conf_vals = confs.cpu().numpy() if hasattr(confs, 'cpu') else confs
        cls_vals = cls_idxs.cpu().numpy() if hasattr(cls_idxs, 'cpu') else cls_idxs

        per_class_max = {}
        for c_idx, conf in zip(cls_vals, conf_vals):
            c = int(c_idx)
            conf_f = float(conf)
            if c not in per_class_max or conf_f > per_class_max[c]:
                per_class_max[c] = conf_f

        if not per_class_max:
            return None

        best_class_idx = max(per_class_max.items(), key=lambda kv: kv[1])[0]
        return model.names[int(best_class_idx)]
    except Exception:
        return None


last_label = None
label_count = 0
last_sent_label = None
frames_without_detection = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame")
            break

        results = model(frame, stream=True)

        sent_label_this_frame = None
        label_for_frame = None
        annotated_frame = None
        for result in results:
            annotated_frame = result.plot()
            label_for_frame = most_confident_label(result)
            break

        if annotated_frame is not None:
            cv2.imshow("ESP32-S3 YOLO Inference", annotated_frame)

        if label_for_frame is None:
            frames_without_detection += 1
            last_label = None
            label_count = 0
            if frames_without_detection >= OFF_AFTER_NO_DETECTION_FRAMES:
                if last_sent_label != 'off':
                    send_label_to_esp('off')
                    last_sent_label = 'off'
        else:
            frames_without_detection = 0
            if label_for_frame == last_label:
                label_count += 1
            else:
                last_label = label_for_frame
                label_count = 1

            if label_for_frame is not None and label_count >= PERSISTENCE_FRAMES:
                if last_sent_label != label_for_frame:
                    send_label_to_esp(label_for_frame)
                    last_sent_label = label_for_frame
                sent_label_this_frame = label_for_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    try:
        send_label_to_esp('off')
        print("Sent 'off' to ESP32 to disable all LEDs")
    except Exception as e:
        print("Error sending 'off' to ESP32:", e)

    cap.release()
    cv2.destroyAllWindows()