from torch.utils.data import Dataset
import numpy as np
import cv2
import warnings
import os
import pandas as pd
import lmdb
import pyarrow as pa
import six
try:
    from src import utils
except:
    import utils as utils
try:
    import src.config as config
except:
    import config

class CarDataset(Dataset):
    def __init__(self, txt_files, transforms, size=(512, 512), data_dir='', train=False):
        image_filenames = []
        for txt_file in txt_files:
            with open(os.path.join(data_dir, txt_file)) as f:
                image_filenames.append(np.array(f.read().splitlines()))
        self.image_filenames = np.concatenate(image_filenames, axis=0)
        if train:
            np.random.shuffle(self.image_filenames)

        self.size = size
        self.transformation = transforms
        self.data_dir = data_dir


    def __getitem__(self, index):
        image_filename = os.path.join(self.data_dir, self.image_filenames[index])
        with open(image_filename, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            return self[(index + 1) % len(self)]

        car_filename = image_filename.replace('.jpg', '.cb').replace(".jpeg", ".cb").replace(".png", ".cb")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image = cv2.imread(image_filename)
            image = cv2.resize(image, self.size)
            car_boxes = np.loadtxt(car_filename).reshape(-1,4)
        if sum(car_boxes[:, 2] < 0) > 0 or sum(car_boxes[:, 3] < 0) > 0 or sum(car_boxes[:, 2] > 1) > 0 or sum(
                car_boxes[:, 3] > 1) > 0 or len(car_boxes) > config.MAX_OBJECTS:
            return self[(index + 1) % len(self)]

        return self.transformation(image, car_boxes)

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    train_transforms, val_transforms = utils.get_transforms()
    train_dataset = CarDataset(txt_files=config.train_txt_files, transforms=train_transforms,
                               size=(640, 640), data_dir='/mnt', train=True)
    for idx, (img, car_boxes) in enumerate(train_dataset):
        print(img,car_boxes.shape)