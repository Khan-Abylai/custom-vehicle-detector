import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as fn
try:
    from src.tools.bbox_utils import BBoxUtilsCar, BBoxUtilsPlate, BBoxUtilsCarWithAnchors, BBoxUtilsPlateWithAnchors, \
        BBoxUtils
    import src.config as config
except:
    from tools.bbox_utils import BBoxUtilsCar, BBoxUtilsPlate, BBoxUtilsCarWithAnchors, BBoxUtilsPlateWithAnchors, \
        BBoxUtils
    import config as config

class CarYoloBlock(nn.Module):

    def __init__(self, image_size, grid_size, conf_scale=2, coord_scale=2, noobject_scale=0.05, iou_threshold=0.7,
                 conf_threshold=0.5, cuda=None):
        super().__init__()
        self.bbox_attrs = 5
        self.w, self.h = image_size
        self.grid_size = grid_size
        self.grid_w = self.w // grid_size
        self.grid_h = self.h // grid_size
        self.noobject_scale = noobject_scale
        self.coord_scale = coord_scale
        self.conf_scale = conf_scale
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.bbox_u = BBoxUtilsCar()

        x_y_offset = np.stack(np.meshgrid(np.arange(self.grid_w), np.arange(self.grid_h)), axis=2)
        self.x_y_offset = torch.tensor(x_y_offset, dtype=torch.float).unsqueeze(0).cuda()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        if cuda is None:
            self.cuda = 0
        else:
            self.cuda = cuda

    def forward(self, coordinates, target=None, validate=False):
        batch_size = coordinates.size(0)

        coordinates = coordinates.permute(0, 2, 3, 1).contiguous()
        coordinates[..., :2] = torch.sigmoid(coordinates[..., :2])
        coordinates[..., -1] = torch.sigmoid(coordinates[..., -1])

        prediction_boxes = coordinates.detach().clone() if target is not None else coordinates

        prediction_boxes[..., :2] = prediction_boxes[..., :2] + self.x_y_offset
        prediction_boxes[..., 2:4] = torch.exp(prediction_boxes[..., 2:4])

        if target is not None:
            x = coordinates[..., 0]
            y = coordinates[..., 1]
            w = coordinates[..., 2]
            h = coordinates[..., 3]

            object_confidence = coordinates[..., -1]

            output = self.bbox_u.create_car_targets_bounding_boxes_vectorized(prediction_boxes, target, self.grid_w,
                                                                              self.grid_h, self.iou_threshold,
                                                                              self.conf_threshold, validate, cuda=self.cuda)

            correct_predictions, incorrect_predictions, objects, mask, target_x, target_y, target_w, target_h = output

            loss_x = self.coord_scale * self.mse_loss(x * mask, target_x * mask)
            loss_y = self.coord_scale * self.mse_loss(y * mask, target_y * mask)
            loss_w = self.coord_scale * self.mse_loss(w * mask, target_w * mask)
            loss_h = self.coord_scale * self.mse_loss(h * mask, target_h * mask)

            loss_conf = self.conf_scale * self.bce_loss(object_confidence * mask,
                                                        mask) + self.noobject_scale * self.bce_loss(
                object_confidence * (1 - mask), mask * (1 - mask))

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf

            return correct_predictions, incorrect_predictions, objects, loss

        prediction_boxes[..., :-1] = prediction_boxes[..., :-1] * self.grid_size

        return prediction_boxes.view(batch_size, self.grid_w * self.grid_h, self.bbox_attrs)


class PlateYoloBlock(nn.Module):

    def __init__(self, image_size, grid_size, conf_scale=2, coord_scale=2, noobject_scale=0.05, iou_threshold=0.7,
                 conf_threshold=0.5, calc_weights=False):
        super().__init__()
        self.bbox_attrs = 13
        self.w, self.h = image_size
        self.grid_size = grid_size
        self.grid_w = self.w // grid_size
        self.grid_h = self.h // grid_size
        self.noobject_scale = noobject_scale
        self.coord_scale = coord_scale
        self.conf_scale = conf_scale
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.bbox_u = BBoxUtilsPlate()
        self.calc_weights = calc_weights

        x_y_offset = np.stack(np.meshgrid(np.arange(self.grid_w), np.arange(self.grid_h)), axis=2)
        self.x_y_offset = torch.tensor(x_y_offset, dtype=torch.float).unsqueeze(0).cuda()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, coordinates, target=None, validate=False):
        batch_size = coordinates.size(0)

        coordinates = coordinates.permute(0, 2, 3, 1).contiguous()

        coordinates[..., :2] = torch.sigmoid(coordinates[..., :2])
        coordinates[..., -1] = torch.sigmoid(coordinates[..., -1])

        prediction_boxes = coordinates.detach().clone() if target is not None else coordinates

        prediction_boxes[..., 2:4] = torch.exp(prediction_boxes[..., 2:4])
        prediction_boxes[..., :2] = prediction_boxes[..., :2] + self.x_y_offset

        if target is not None:
            x = coordinates[..., 0]
            y = coordinates[..., 1]
            w = coordinates[..., 2]
            h = coordinates[..., 3]

            x1 = coordinates[..., 4]
            y1 = coordinates[..., 5]
            x2 = coordinates[..., 6]
            y2 = coordinates[..., 7]
            x3 = coordinates[..., 8]
            y3 = coordinates[..., 9]
            x4 = coordinates[..., 10]
            y4 = coordinates[..., 11]

            object_confidence = coordinates[..., -1]

            output = self.bbox_u.create_plate_targets_bounding_boxes_vectorized(prediction_boxes, target, self.grid_w,
                                                                                self.grid_h, self.iou_threshold,
                                                                                self.conf_threshold, validate,
                                                                                calc_weights=self.calc_weights)

            correct_predictions, incorrect_predictions, objects, mask, loss_weights, target_x, target_y, target_w, target_h = output[
                                                                                                                              :9]
            target_x1, target_y1, target_x2, target_y2, target_x3, target_y3, target_x4, target_y4 = output[9:]

            loss_x = self.coord_scale * self.mse_loss(x * mask * loss_weights, target_x * mask * loss_weights) * 2
            loss_y = self.coord_scale * self.mse_loss(y * mask * loss_weights, target_y * mask * loss_weights) * 2

            loss_w = self.coord_scale * self.mse_loss(w * mask * loss_weights, target_w * mask * loss_weights)
            loss_h = self.coord_scale * self.mse_loss(h * mask * loss_weights, target_h * mask * loss_weights)

            loss_x1 = self.coord_scale * self.mse_loss(x1 * mask * loss_weights, target_x1 * mask * loss_weights)
            loss_y1 = self.coord_scale * self.mse_loss(y1 * mask * loss_weights, target_y1 * mask * loss_weights)

            loss_x2 = self.coord_scale * self.mse_loss(x2 * mask * loss_weights, target_x2 * mask * loss_weights)
            loss_y2 = self.coord_scale * self.mse_loss(y2 * mask * loss_weights, target_y2 * mask * loss_weights)

            loss_x3 = self.coord_scale * self.mse_loss(x3 * mask * loss_weights, target_x3 * mask * loss_weights)
            loss_y3 = self.coord_scale * self.mse_loss(y3 * mask * loss_weights, target_y3 * mask * loss_weights)

            loss_x4 = self.coord_scale * self.mse_loss(x4 * mask * loss_weights, target_x4 * mask * loss_weights)
            loss_y4 = self.coord_scale * self.mse_loss(y4 * mask * loss_weights, target_y4 * mask * loss_weights)

            loss_conf = self.conf_scale * self.bce_loss(object_confidence * mask,
                                                        mask) + self.noobject_scale * self.bce_loss(
                object_confidence * (1 - mask), mask * (1 - mask))

            loss = (
                        loss_x + loss_y + loss_w + loss_h + loss_x1 + loss_y1 + loss_x2 + loss_y2 + loss_x3 + loss_y3 + loss_x4 + loss_y4 + loss_conf)

            return [correct_predictions, incorrect_predictions, objects, loss]

        prediction_boxes[..., :-1] = prediction_boxes[..., :-1] * self.grid_size
        return prediction_boxes.view(batch_size, self.grid_w * self.grid_h, self.bbox_attrs)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.training:
            return fn.dropout2d(fn.relu(self.bn(self.conv(x))), 0.25)
        else:
            return fn.relu(self.bn(self.conv(x)))


class LinearConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x, t=None, validate=False):
        return self.conv(x)


class OrdinaryConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
