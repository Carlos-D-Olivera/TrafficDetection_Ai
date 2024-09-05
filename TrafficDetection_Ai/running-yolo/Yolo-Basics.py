from ultralytics import YOLO
import cv2
model = YOLO('../Yolo-weights/yolov8l.pt')
results = model("Images/2.webp", show =True)
cv2.waitKey(0)