from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 640)

model = YOLO('../Yolo-weights/yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxex = r.boxes
        for box in boxex:

            #Rectangle on opencv

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            """
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            """
            #cvzone
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img, f'{conf}', (max(35,x1), max(35,y1-20)))



    cv2.imshow('img', img)
    cv2.waitKey(1)