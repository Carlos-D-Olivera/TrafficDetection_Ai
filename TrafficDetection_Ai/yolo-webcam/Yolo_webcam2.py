from ultralytics import YOLO
import cv2
import cvzone
import math
import threading

# Modelo YOLO
model = YOLO('../Yolo-weights/yolov8n.pt')

# Nombres de clases
classNames = ["Humano", "bicycle", "carro", "motocicleta", "", "bus", "", "truck", "",
              "traffic light", "", "stop sign", "", "", "", "",
              "", "", "", "", "", "", "", "", "", "",
              "", "", "", "", "", "", "", "", "",
              "", "", "", "", "", "", "",
              "", "Cuchillo/Navaja", "", "", "", "", "", "", "",
              "", "", "", "", "", "", "", "", "",
              "", "", "", "", "", "", "", "cell phone",
              "", "", "", "", "", "", "", "", "Tijeras",
              "", "", ""]


# Función para procesar cada cámara
def process_camera(camera_id, window_name, ancho, altura):
    cap = cv2.VideoCapture(camera_id)
    cap.set(3, ancho)  # Ancho
    cap.set(4, altura)  # Altura

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordenadas del rectángulo
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


                # Clase
                cls = int(box.cls[0])
                # Confianza
                conf = math.ceil((box.conf[0] * 100)) / 100

                if f'{classNames[cls]}' != "":
                    # Rectángulo de cvzone
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))


                    # Texto en la imagen
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

        # Mostrar el video
        cv2.imshow(window_name, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


# Crear hilos para dos cámaras
thread1 = threading.Thread(target=process_camera, args=(0, 'Camara 1', 200, 100))
thread2 = threading.Thread(target=process_camera, args=(1, 'Camara 2', 1440, 720))

# Iniciar los hilos+
thread1.start()
thread2.start()

# Esperar a que los hilos terminen
thread1.join()
thread2.join()
