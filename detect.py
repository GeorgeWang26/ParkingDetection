import json
import cv2
from ultralytics import YOLO
import requests

URL = "127.0.0.1:5000/update-spots-cv"

with open("coordinates_scaled.json", "r") as f:
    data = json.load(f)

model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    frame = frame[130:200, 110:570]
    
    results = model.predict(frame)
    r = results[0]
    
    lots_occu = [False] * (len(data["x"])-1)
    for i in range(len(r.boxes.cls)):
        clss = r.boxes.cls[i]
        xyxy = r.boxes.xyxy[i]
        # 2:car, 3:motorcycle, 5:bus, 7:truck
        if clss==2 or clss==3 or clss==5 or clss==7:
            for j in range(len(data["x"])-1):
                # if ((car cross left lot bound) or (car cross right lot bound) or (car within lot bound)) and (y direction in bound)
                if (xyxy[0] < data["x"][j][0] and data["x"][j][0] < xyxy[2] and (xyxy[2]-data["x"][j][0])/(data["x"][j+1][0]-data["x"][j][0]) > 0.8) \
                        or (xyxy[0] < data["x"][j+1][0] and data["x"][j+1][0] < xyxy[2] and (data["x"][j+1][0]-xyxy[0])/(data["x"][j+1][0]-data["x"][j][0]) > 0.8) \
                        or (data["x"][j][0] < xyxy[0] and xyxy[2] < data["x"][j+1][0]) \
                    and (xyxy[3]-xyxy[1])/(data["y_max"]-data["y_min"]) > 0.5:
                    # print((xyxy[3]-xyxy[1])/(data["y_max"]-data["y_min"]))
                    lots_occu[j] = True
                    break

    requests.post(URL, lots_occu)    
    for i in range(len(lots_occu)):
        if lots_occu[i] == True:
            cv2.rectangle(frame, (data["x"][i][0], data["y_min"]), (data["x"][i+1][0], data["y_max"]), (0, 0, 255), 10)
        else:
            cv2.rectangle(frame, (data["x"][i][0], data["y_min"]), (data["x"][i+1][0], data["y_max"]), (0, 255, 0), 10)
    cv2.imshow("cam", frame)
    cv2.waitKey(1)


cam.release()
cv2.destroyAllWindows()
