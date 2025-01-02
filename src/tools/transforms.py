import random
import cv2
import numpy as np
from imgaug import augmenters as ia
import torch
try:
    from src.config import CAR_COORDINATE_DIMENSIONS, MAX_OBJECTS
except:
    from config import CAR_COORDINATE_DIMENSIONS, MAX_OBJECTS

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, car_boxes=None):
        for t in self.transforms:
            img, car_boxes = t(img, car_boxes)
        return img, car_boxes


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, img, car_boxes=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            img, car_boxes = t(img, car_boxes)
        return img, car_boxes


class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, img, car_boxes=None):
        if random.random() < self.prob:
            img, car_boxes = self.first(img, car_boxes)
        else:
            img, car_boxes = self.second(img, car_boxes)
        return img, car_boxes


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, car_boxes=None):
        return self.trans(img), car_boxes


class BoxOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, car_boxes):
        car_boxes = self.trans(car_boxes)
        return img, car_boxes


class FillBox:

    def __call__(self, car_boxes):
        return (
            np.concatenate((car_boxes, np.zeros((MAX_OBJECTS - len(car_boxes), CAR_COORDINATE_DIMENSIONS - 1))),
                           axis=0))


class ScaleDown:
    def __init__(self, scale=0.5, prob=.5):
        self.scale = scale
        self.prob = prob

    def __call__(self, img, car_boxes=None):
        if random.random() < self.prob:
            scale = random.uniform(0.01, self.scale)
            h, w, _ = img.shape

            new_h = round(h * (1 - scale))
            new_w = round(w * (1 - scale))

            if new_h % 2 == 1:
                new_h += 1
            if new_w % 2 == 1:
                new_w += 1

            dh = h - new_h
            dw = w - new_w

            top = np.random.randint(0, dh)
            left = np.random.randint(0, dw)
            bottom = dh - top
            right = dw - left
            img = cv2.resize(img, (new_w, new_h))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 0)

            if car_boxes is not None and car_boxes.sum() != 0:
                zeros_indicies = np.where(~car_boxes.any(axis=1))
                car_boxes[:, [0, 2]] *= (new_w / w)
                car_boxes[:, [1, 3]] *= (new_h / h)
                car_boxes[:, 0] += (left / w)
                car_boxes[:, 1] += (top / h)
                car_boxes[zeros_indicies] = 0

        return img, car_boxes


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, car_boxes=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if car_boxes is not None and car_boxes.sum() != 0:
                zeros_indicies = np.where(~car_boxes.any(axis=1))
                car_boxes[:, 0] = 1 - car_boxes[:, 0]
                car_boxes[zeros_indicies] = 0

        return img, car_boxes


def to_rectange(box):
    x1, x2 = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
    y1, y2 = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2

    return np.array(((x1, y1), (x1, y2), (x2, y1), (x2, y2))).transpose((2, 0, 1))


class Rotate:
    def __init__(self, limit=25, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, car_boxes=None):

        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            height, width, _ = img.shape
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

            img = cv2.warpAffine(img, M=mat, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

            if car_boxes is not None and car_boxes.sum() != 0:
                car_boxes[:, [0, 2]] *= width
                car_boxes[:, [1, 3]] *= height
                car_boxes = to_rectange(car_boxes)
                car_boxes = cv2.transform(car_boxes, mat)

                x1 = np.amin(car_boxes[:, :, :1], axis=(1, 2))
                y1 = np.amin(car_boxes[:, :, 1:2], axis=(1, 2))

                x2 = np.amax(car_boxes[:, :, :1], axis=(1, 2))
                y2 = np.amax(car_boxes[:, :, 1:2], axis=(1, 2))

                box_width = x2 - x1
                box_height = y2 - y1

                center_x = x1 + box_width / 2
                center_y = y1 + box_height / 2

                new_box = np.stack((center_x, center_y, box_width, box_height), axis=1)
                new_box[:, [0, 2]] /= width
                new_box[:, [1, 3]] /= height
                new_box[np.where(~car_boxes.any(axis=1))] = 0
                new_box = new_box.clip(0, .99)
                car_boxes = new_box.copy()

        return (img, car_boxes)

    def fix(self, x, y, slope, intercept, img_w, img_h):
        center_x, center_y = img_w / 2, img_h / 2
        if x < center_x and y < center_y:
            if x < 0 or y < 0:
                return (-intercept / slope, 0) if (intercept < 0 or intercept > img_h) else (0, intercept)
        elif x > center_x and y < center_y:
            if x > img_w or y < 0:
                return (-intercept / slope, 0) if (intercept < 0 or intercept > img_h) else (
                    img_w, img_w * slope + intercept)
        elif x < center_x and y > center_y:
            if x < 0 or y > img_h:
                return ((img_h - intercept) / slope, img_h) if (intercept < 0 or intercept > img_h) else (0, intercept)
        elif x > center_x and y > center_y:
            if x > img_w or y > img_h:
                return ((img_h - intercept) / slope, img_h) if (intercept < 0 or intercept > img_h) else (
                    img_w, img_w * slope + intercept)
        return x, y


class Transpose:

    def __call__(self, img):
        return img.transpose((2, 0, 1))


class RandomChannelPermute:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = img[:, :, np.random.permutation(3)]
        return img


class RandomGrayScale:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img


class Resize:

    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)


class Normalize:

    def __init__(self):
        self.MAX_PIXEL_VALUE = 255.0

    def __call__(self, img,  car_boxes=None):
        return 2 * (img / self.MAX_PIXEL_VALUE - 0.5), car_boxes


class ToTensor:

    def __call__(self, img, car_boxes=None):
        return (torch.from_numpy(img).float(),
                torch.from_numpy(car_boxes).float())


class ImgAugmenter(object):

    def __init__(self, prob=0.5):
        self.prob = prob
        self.tranformation = ia.Noop()

    def __call__(self, img):
        if random.random() < self.prob:
            img = self.tranformation.augment_image(img)
        return img


class AdditiveGaussianNoise(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(AdditiveGaussianNoise, self).__init__()
        self.prob = prob
        self.tranformation = ia.AdditiveGaussianNoise(
            scale=(2, 15), per_channel=0.5)


class SaltAndPepper(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
        self.tranformation = ia.SaltAndPepper(p=(0.02, 0.08))


class Dropout(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(Dropout, self).__init__()
        self.prob = prob
        self.tranformation = ia.Dropout(p=(0.02, 0.08))


class GaussianBlur(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(GaussianBlur, self).__init__()
        self.prob = prob
        self.tranformation = ia.GaussianBlur(sigma=(0.1, 1.2))

class AverageBlur(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(AverageBlur, self).__init__()
        self.prob = prob
        self.tranformation = ia.AverageBlur(k=(1, 5))


class Add(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(Add, self).__init__()
        self.prob = prob
        self.tranformation = ia.Add(value=(-80, 120), per_channel=0.5)


class ContrastNormalization(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(ContrastNormalization, self).__init__()
        self.prob = prob
        self.tranformation = ia.ContrastNormalization(
            (0.5, 2.5), per_channel=0.5)


class AddToHueAndSaturation(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(AddToHueAndSaturation, self).__init__()
        self.prob = prob
        self.tranformation = ia.AddToHueAndSaturation(
            (-40, 40), from_colorspace="BGR")


class Sharpen(ImgAugmenter):

    def __init__(self, prob=0.5):
        super(Sharpen, self).__init__()
        self.prob = prob
        self.tranformation = ia.Sharpen(alpha=(0, 0.75), lightness=(0.75, 1.2))
