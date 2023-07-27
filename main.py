import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

#converting classes.txt to a list
with open("classes.txt", "r") as cls:

    lines = cls.readlines()
    new_lines = [x[:-1] for x in lines]

#object detection


cap = cv2.VideoCapture("wp2.mp4")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    # print(results)
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    # print(classes)

    count = 0
    for cls, bbox in zip(classes, bboxes):
        (x,y,x2,y2) = bbox

        if cls == 0: #cls 0 corresponds to class person
            count = count + 1
            cv2.rectangle(frame, (x,y), (x2,y2), (255, 0, 0), 2)
            cv2.putText(frame, new_lines[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


    # cv2.putText(frame, str(count), (0,50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2)
    cv2.putText(frame, ('Persons = ' + str(count)), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    cv2.imshow("Img", frame)

    key = cv2.waitKey(1)
    if key == 3:
        break

cap.release()
cv2.destroyAllWindows()