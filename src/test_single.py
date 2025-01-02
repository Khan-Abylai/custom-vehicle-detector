import glob
import os.path

import torch
import torch.nn as nn
import cv2
import numpy as np
import warnings

from models.models import LPDetector
import utils
import tools.bbox_utils as bu
from tools import transforms

import time

def draw_car_box(img, box, color=(0, 0, 255)):
    cv2.circle(img, (int(box[0]), int(box[1])), 5, color, -1)

    x1 = int((box[0] - box[2] / 2.))
    y1 = int((box[1] - box[3] / 2.))
    x2 = int((box[0] + box[2] / 2.))
    y2 = int((box[1] + box[3] / 2.))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    return img

transform = transforms.DualCompose(
    [transforms.ImageOnly(transforms.Transpose()), transforms.ImageOnly(transforms.Normalize())])

img_w = 640
img_h = 640
img_size = (img_w, img_h)
model = LPDetector(img_size).cuda()
checkpoint = '../weights/car_detector_2.pth'
model = nn.DataParallel(model)
checkpoint = torch.load(checkpoint)['state_dict']
model.load_state_dict(checkpoint)
model.eval()


ls = glob.glob('../sync_data/east_vokzal/*.jpeg')

for i, path in enumerate(ls):
    start_time = time.time()

    img = cv2.imread(path)
    if img is None:
        continue

    h, w, _ = img.shape

    img_for_pred = img.copy()
    x= transform(cv2.resize(img, img_size))[0][0]
    x = torch.from_numpy(x).float()
    x = torch.stack([x]).cuda()

    car_output = model(x)

    car_output = car_output.cpu().detach().numpy()
    car_output = car_output.reshape(1, -1, 5)

    rx = float(w) / img_w
    ry = float(h) / img_h
    cars_pred = bu.nms_np(car_output[0], conf_thres=0.90, nms_thres=0.5)

    end_time = time.time() - start_time
    print(f'exec time :{end_time} {i} out of {len(ls)}')
    cars_pred[..., 0::2] *= rx
    cars_pred[..., 1::2] *= ry
    for car in cars_pred:
        img = draw_car_box(img, car, color=(0, 0, 255))
        img_for_pred = draw_car_box(img_for_pred, car, color=(0, 0, 255))

    cv2.imwrite(f"../sync_data/others/{os.path.basename(path)}", img_for_pred)
