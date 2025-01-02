import os.path
import os.path as osp
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import shutil

np.random.seed(42)

model = YOLO("./yolov8x.pt")

classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
           14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
           29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
           41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
           49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
           57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
           64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
           71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
           78: 'hair drier', 79: 'toothbrush'}

need_classes = [2, 5, 6, 7]

all_images = glob('/mnt/data/custom_car_dataset/**/**/*.jpg') + glob('/mnt/data/custom_car_dataset/**/**/*.jpeg')
all_images.extend(glob("/mnt/data/vokzal_images/**/*.jpg"))
all_images.extend(glob("/mnt/data/vokzal_images/**/*.jpeg"))

np.random.shuffle(all_images)
threshold = 0.95

for idx, path in enumerate(all_images):
    image = cv2.imread(path)
    result = model.predict(path, classes=need_classes)[0]

    h, w = result.orig_shape
    boxes = result.boxes.cpu()
    res_boxes = []
    for res in boxes:
        conf = res.conf
        if conf > threshold:
            coordinates = res.xywh[0].numpy()
            res_boxes.append(coordinates)
    res_boxes = np.array(res_boxes)
    if len(res_boxes) == 0 or np.any(res_boxes) == False:
        continue
    else:
        res_boxes = np.array(res_boxes)
        res_boxes[:, 0::2] /= w
        res_boxes[:, 1::2] /= h

        np.savetxt(path.replace(".jpg", ".cb").replace(".jpeg", ".cb"), res_boxes)
        print(idx, path, "recorded")
