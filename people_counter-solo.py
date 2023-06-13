import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(0) #0 or 1

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=10e10, min_hits=30, iou_threshold=0.3)

col = ((204, 255, 153))

while True:
    success, img = cap.read()

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
    results = model(img)
    
    detections = np.empty((0, 5))
    object_count = 0  # Variable to store the count of objects in the frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'person' and conf > 0.75:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                object_count += 1  # Increment the object count

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=1, colorR=col, colorC=col)
        cv2.putText(img, f'Person ID: {int(id)}', (max(0, x1) - 10, max(35, y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, col, 6)

    cv2.putText(img, str(object_count), (135, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)

    img = cv2.resize(img, (1200,800)) #adjust height and weight to your preferences
    cv2.imshow("People counter with Yolov8", img)
    if cv2.waitKey(1) & 0xFF == ord("q"): #press q to quit the program
            break
