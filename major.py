from ultralytics import YOLO 


import cv2

model = YOLO("C:/Users/Prathamesh/OneDrive/Desktop/MAJOR/best1.pt")
model.predict(source="0",show=True, conf=0.5)