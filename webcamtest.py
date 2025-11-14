from ultralytics import YOLO
import cv2

model_path = "runs/classify_v1/train/weights/best.pt"
model = YOLO(model_path)  

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    probs = results.probs  
    class_id = probs.top1
    class_name = model.names[class_id]
    confidence = probs.top1conf

    text = f"{class_name} ({confidence:.2f})"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("YOLO Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
