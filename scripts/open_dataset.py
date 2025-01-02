import os
import os.path as osp
import shutil
from glob import glob
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

def xyxy_to_xywh(xyxy):
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

folder = '/mnt/car_dataset/images'
out_folder = '/mnt/car_dataset/for_training'

all_images=glob(osp.join(folder, "**", "*.jpg"))

for idx, item in enumerate(all_images):
    annotation = item.replace('/images/','/labels/').replace(".jpg", '.txt')
    with open(annotation, 'r') as f:
        content = np.array([np.array([float(y) for y in x.split()]) for x in f.read().split('\n') if len(x) !=0])
    h,w,_ = cv2.imread(item).shape

    content = np.delete(content, 0,1)
    b = content[:, content.min(axis=0) >= 0]

    if b.shape[1] == 0:
        continue

    path = osp.join(out_folder, osp.basename(osp.dirname(item)), osp.basename(item))
    cb_path = path.replace(".jpg", ".cb")

    shutil.copy(item,path)
    np.savetxt(cb_path, content)

    print(idx, item)