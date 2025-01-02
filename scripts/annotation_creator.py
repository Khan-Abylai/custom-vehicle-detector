import os
import os.path as osp
import shutil
from glob import glob
import numpy as np

from sklearn.model_selection import train_test_split

# image_folder = '/mnt/vokzal_images'
# label_folder = '/mnt/vokzal_labels'
#
# images = np.array(glob(osp.join(image_folder, "**", "*.jpeg")))
# labels = np.array(glob(osp.join(label_folder, "*.sb")))
#
# labels_dict = {}
#
# for label in labels:
#     labels_dict[osp.basename(label.replace(".sb", ".jpeg"))] = label
#
# print(labels_dict)
#
# for image in images:
#     basename = osp.basename(image)
#
#     if basename in labels_dict:
#         shutil.copy(labels_dict[basename], image.replace(".jpeg", ".cb"))
#         print(basename)
# main_folder = '/mnt'
#
# target_folder = '/mnt/custom_car_dataset'
#
# all_images= [osp.join(target_folder.replace(main_folder+'/', ""),osp.basename(osp.dirname(osp.dirname(x))),osp.basename(osp.dirname(x)),osp.basename(x)) for x in glob(osp.join(target_folder, "**","**", "*.jpeg")) if osp.exists(x.replace(".jpeg", ".cb"))]
#

import pandas as pd
all_images = pd.read_csv('/mnt/data/filename.txt', header=None)

train, test = train_test_split(all_images, test_size=0.2, random_state=42)
print(train.shape)
print(test.shape)

np.savetxt("/mnt/data/train.txt", train, delimiter=',', fmt='%s')
np.savetxt("/mnt/data/test.txt", test, delimiter=',', fmt='%s')


# file = '/mnt/custom_car_dataset/baqorda/test.txt'
#
# with open(file, 'r') as f:
#     content = [x.replace('\n', '') for x in f.readlines()]
#
# for idx, image_path in enumerate(content):
#     print(idx, image_path)
#     image_path = osp.join('/mnt', image_path)
#     cb_file_path = image_path.replace('.jpeg', '.cb')
#     car_boxes = np.loadtxt(cb_file_path).reshape(-1, 4)
#     print(car_boxes)

# base_folder = "/mnt/data"
#
# all_images = glob(f'{base_folder}/custom_car_dataset/**/**/*.jpg') + glob(
#     f'{base_folder}/custom_car_dataset/**/**/*.jpeg')
# all_images.extend(glob(f"{base_folder}/vokzal_images/**/*.jpg"))
# all_images.extend(glob(f"{base_folder}/vokzal_images/**/*.jpeg"))
# print(len(all_images))
# all_images = [x for x in all_images if osp.exists(x.replace(".jpeg", ".cb").replace(".jpg", ".cb")) and
#               x.replace(".jpeg", ".pb").replace(".jpg", ".pb")]
#
# all_images = np.array(all_images)
#
#
# np.savetxt(osp.join(base_folder, "filename.txt"), all_images, delimiter=" ", fmt="%s")