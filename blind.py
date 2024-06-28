## 합치는거

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap = cv2.VideoCapture("Videos/whitecane11.mp4")
# model = YOLO("Yolo-Weights/yolov8n.pt")

modelPath1 = (
    r"C:\siyeon\Yolo\Object-Detection\runs\detect\yolov8n_whitecan002\weights\best.pt"
)

modelPath2 = "Yolo-Weights/yolov8n.pt"

model1 = YOLO(modelPath1)
model2 = YOLO(modelPath2)

# greenLower = (36, 25, 25)
# greenUpper = (70, 255, 255)

classNames = [
    "whiteCane",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "traffic light",
    "boat",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
classNames2 = [
    "person",
    "whiteCane",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "traffic light",
    "boat",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

start_time = time.time()

greenLower = (36, 25, 25)
greenUpper = (70, 255, 255)


while True:
    success, img = cap.read()
    if not success:
        break
    # results = model(img, stream=True)
    results1 = model1(img, stream=True)
    results2 = model2(img, stream=True)

    for r in results1:
        boxes = r.boxes
        # print(len(boxes))

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1.item())
            y1 = int(y1.item())
            x2 = int(x2.item())
            y2 = int(y2.item())
            print(x1, y1, x2, y2)
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if (currentClass == "whiteCane") and conf > 0.5:
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=0.8,
                    thickness=1,
                    offset=3,
                )
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        for r in results2:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1.item())
                y1 = int(y1.item())
                x2 = int(x2.item())
                y2 = int(y2.item())
                print(x1, y1, x2, y2)
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = img[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence
                cls = int(box.cls[0])
                currentClass = classNames2[cls]

                cvzone.putTextRect(
                    img,
                    f"{classNames2[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=0.8,
                    thickness=1,
                    offset=3,
                )

                # if (currentClass == "traffic light") and conf > 0.1:
        current_time = time.time()  # 현재 시간 가져오기
        if current_time - start_time >= 13.2:  # 시작한 지 15초가 지났는지 확인
            cvzone.putTextRect(
                img,
                "go!",
                (50, 50),
                scale=3,
                thickness=3,
                colorR=(0, 255, 0),
                offset=10,
            )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
