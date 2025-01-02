import torch
try:
    import src.config as config
except:
    import config as config
import numpy as np


class BBoxUtilsCar:
    def __init__(self):
        self.initiated = False

    def initiate(self, batch_size, grid_h, grid_w):
        self.initiated = True
        self.mask = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_x = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_w = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_h = torch.zeros(batch_size, grid_h, grid_w).float().cuda()

    def zero_out(self):
        self.mask *= 0
        self.target_x *= 0
        self.target_y *= 0
        self.target_w *= 0
        self.target_h *= 0

    def create_car_targets_bounding_boxes_vectorized(self, predictions, target, grid_w, grid_h, iou_threshold=0.7,
                                                     conf_threshold=0.5, validate=False, cuda=None):

        batch_size = target.size(0)
        if not self.initiated:
            self.initiate(batch_size, grid_h, grid_w)
        else:
            self.zero_out()

        correct_predictions = 0

        non_zero_ind = target.sum(axis=2) > 0

        target[:, :, 0::2] *= grid_w
        target[:, :, 1::2] *= grid_h
        gi_array = torch.clamp(torch.clamp(target[:, :, 0], min=0), max=grid_w - 1).long()
        gj_array = torch.clamp(torch.clamp(target[:, :, 1], min=0), max=grid_h - 1).long()

        self.target_x[:, gj_array, gi_array] = target[:, :, 0] - gi_array
        self.target_y[:, gj_array, gi_array] = target[:, :, 1] - gj_array

        ind = torch.Tensor([list(range(batch_size))] * config.MAX_OBJECTS).long().T.to(f"cuda:{cuda}")
        self.target_w[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = torch.log(
            target[:, :, 2][non_zero_ind])
        self.target_h[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = torch.log(
            target[:, :, 3][non_zero_ind])
        self.mask[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = 1

        if validate:
            for batch_index in range(batch_size):
                for object_index in range(config.MAX_OBJECTS):
                    if target[batch_index, object_index].sum() == 0:
                        continue

                    gi = gi_array[batch_index, object_index]
                    gj = gj_array[batch_index, object_index]

                    gt_box = target[batch_index, object_index].unsqueeze(0)

                    pred_box = predictions[batch_index, gj, gi].unsqueeze(0)

                    if bbox_iou(gt_box, pred_box, x1y1x2y2=False) > iou_threshold and pred_box[
                        ..., -1] > conf_threshold:
                        correct_predictions += 1
        objects = non_zero_ind.sum()
        incorrect_predictions = ((1 - self.mask) * predictions[..., -1] > conf_threshold).sum().item()

        return (correct_predictions, incorrect_predictions, objects, self.mask, self.target_x, self.target_y,
                self.target_w, self.target_h)


class BBoxUtilsCarWithAnchors:
    def __init__(self):
        self.initiated = False
        self.anchors = torch.Tensor(config.CAR_ANCHORS).cuda()

    def initiate(self, batch_size, grid_h, grid_w):
        self.initiated = True

        self.target_output = torch.zeros(batch_size, grid_h, grid_w, len(config.CAR_ANCHORS),
                                         config.CAR_COORDINATE_DIMENSIONS).float().cuda()

    def zero_out(self):
        self.target_output *= 0

    def iou(self, box1, boxes):
        area1 = box1[0] * box1[1]
        area2 = boxes[:, 0] * boxes[:, 1]
        area_inter = torch.minimum(box1[0], boxes[:, 0]) * torch.minimum(box1[1], boxes[:, 1])
        return area_inter / (area2 + area1 - area_inter)

    def create_car_targets_bounding_boxes_with_anchors(self, predictions, target, grid_w, grid_h, iou_threshold=0.7,
                                                       conf_threshold=0.5, validate=False):

        batch_size = target.size(0)
        if not self.initiated:
            self.initiate(batch_size, grid_h, grid_w)
        else:
            self.zero_out()

        correct_predictions = 0
        non_zero_ind = target.sum(axis=2) > 0

        target[:, :, 0] *= grid_w
        target[:, :, 1] *= grid_h
        gi_array = torch.clamp(torch.clamp(target[:, :, 0], min=0), max=grid_w - 1).long()
        gj_array = torch.clamp(torch.clamp(target[:, :, 1], min=0), max=grid_h - 1).long()

        for batch_ind in range(batch_size):
            for box_ind, target_box in enumerate(target[batch_ind]):
                if sum(target_box) == 0:
                    continue
                iou_anchors = self.iou(target_box[2:], self.anchors)
                anchors_ind = torch.argmax(iou_anchors)

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 0] = \
                    target_box[0] - gi_array[batch_ind, box_ind]

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 1] = \
                    target_box[1] - gj_array[batch_ind, box_ind]

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 2] = \
                    torch.log(target_box[2] / self.anchors[anchors_ind][0])

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 3] = \
                    torch.log(target_box[3] / self.anchors[anchors_ind][1])

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 4] = 1

                if validate:
                    pred = predictions[
                        batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind]
                    pred[0] = pred[0] / grid_w
                    pred[1] = pred[1] / grid_h
                    pred[2] = pred[2] * self.anchors[anchors_ind, 0]
                    pred[3] = pred[3] * self.anchors[anchors_ind, 1]
                    target_box[0] = target_box[0] / grid_w
                    target_box[1] = target_box[1] / grid_h

                    if bbox_iou(target_box.unsqueeze(0), pred.unsqueeze(0)) > iou_threshold and pred[
                        -1] > conf_threshold:
                        correct_predictions += 1

        objects = non_zero_ind.sum()

        return correct_predictions, objects, self.target_output


class BBoxUtilsPlate:
    def __init__(self):
        self.initiated = False

    def initiate(self, batch_size, grid_h, grid_w):
        self.initiated = True
        self.mask = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.loss_weights = torch.ones(batch_size, grid_h, grid_w).float().cuda()
        self.target_x = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_w = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_h = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_x1 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y1 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_x2 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y2 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_x3 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y3 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_x4 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y4 = torch.zeros(batch_size, grid_h, grid_w).float().cuda()

    def zero_out(self):
        self.mask *= 0
        self.loss_weights *= 0
        self.loss_weights += 1
        self.target_x *= 0
        self.target_y *= 0
        self.target_w *= 0
        self.target_h *= 0
        self.target_x1 *= 0
        self.target_y1 *= 0
        self.target_x2 *= 0
        self.target_y2 *= 0
        self.target_x3 *= 0
        self.target_y3 *= 0
        self.target_x4 *= 0
        self.target_y4 *= 0

    def create_plate_targets_bounding_boxes_vectorized(self, predictions, target, grid_w, grid_h, iou_threshold=0.7,
                                                       conf_threshold=0.5, validate=False, calc_weights=False):

        batch_size = target.size(0)
        if not self.initiated:
            self.initiate(batch_size, grid_h, grid_w)
        else:
            self.zero_out()

        correct_predictions = 0

        non_zero_ind = target[:, :, :4].sum(axis=2) > 0

        target[:, :, 0::2] *= grid_w
        target[:, :, 1::2] *= grid_h
        gi_array = torch.clamp(torch.clamp(target[:, :, 0], min=0), max=grid_w - 1).long()
        gj_array = torch.clamp(torch.clamp(target[:, :, 1], min=0), max=grid_h - 1).long()

        self.target_x[:, gj_array, gi_array] = target[:, :, 0] - gi_array
        self.target_y[:, gj_array, gi_array] = target[:, :, 1] - gj_array

        ind = torch.Tensor([list(range(batch_size))] * config.MAX_OBJECTS).long().T
        self.target_w[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = torch.log(
            target[:, :, 2][non_zero_ind])
        self.target_h[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = torch.log(
            target[:, :, 3][non_zero_ind])

        self.target_x1[:, gj_array, gi_array] = target[:, :, 4]
        self.target_y1[:, gj_array, gi_array] = target[:, :, 5]
        self.target_x2[:, gj_array, gi_array] = target[:, :, 6]
        self.target_y2[:, gj_array, gi_array] = target[:, :, 7]
        self.target_x3[:, gj_array, gi_array] = target[:, :, 8]
        self.target_y3[:, gj_array, gi_array] = target[:, :, 9]
        self.target_x4[:, gj_array, gi_array] = target[:, :, 10]
        self.target_y4[:, gj_array, gi_array] = target[:, :, 11]

        self.mask[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = 1
        if calc_weights:
            self.loss_weights[
                ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = config.APPROX_MAX_PLATE / (
                        target[:, :, 2][non_zero_ind] / grid_w)

        if validate:
            for batch_index in range(batch_size):
                for object_index in range(config.MAX_OBJECTS):
                    if target[batch_index, object_index].sum() == 0:
                        continue

                    gt_box = target[batch_index, object_index, :4].unsqueeze(0).cuda()

                    gi = gi_array[batch_index, object_index]
                    gj = gj_array[batch_index, object_index]

                    pred_box = predictions[batch_index, gj, gi]
                    pred_box = torch.FloatTensor([pred_box[0], pred_box[1], pred_box[2], pred_box[3]]).unsqueeze(
                        0).cuda()

                    iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                    if iou > iou_threshold and predictions[batch_index, gj, gi, -1] > conf_threshold:
                        correct_predictions += 1

        objects = non_zero_ind.sum()

        incorrect_predictions = ((1 - self.mask) * predictions[..., -1] > conf_threshold).sum().item()

        return (correct_predictions, incorrect_predictions, objects, self.mask, self.loss_weights,
                self.target_x, self.target_y, self.target_w, self.target_h,
                self.target_x1, self.target_y1, self.target_x2, self.target_y2,
                self.target_x3, self.target_y3, self.target_x4, self.target_y4)


class BBoxUtilsPlateWithAnchors:
    def __init__(self):
        self.initiated = False
        self.anchors = torch.Tensor(config.PLATE_ANCHORS).cuda()

    def initiate(self, batch_size, grid_h, grid_w):
        self.initiated = True
        self.target_output = torch.zeros(batch_size, grid_h, grid_w, len(config.PLATE_ANCHORS),
                                         config.PLATE_COORDINATE_DIMENSIONS).float().cuda()

    def zero_out(self):
        self.target_output *= 0

    def iou(self, box1, boxes):
        area1 = box1[0] * box1[1]
        area2 = boxes[:, 0] * boxes[:, 1]
        area_inter = torch.minimum(box1[0], boxes[:, 0]) * torch.minimum(box1[1], boxes[:, 1])
        return area_inter / (area2 + area1 - area_inter)

    def create_plate_targets_bounding_boxes_with_anchors(self, predictions, target, grid_w, grid_h, iou_threshold=0.7,
                                                         conf_threshold=0.5, validate=False):

        batch_size = target.size(0)
        if not self.initiated:
            self.initiate(batch_size, grid_h, grid_w)
        else:
            self.zero_out()

        correct_predictions = 0
        non_zero_ind = target[:, :, :4].sum(axis=2) > 0

        target[:, :, 0] *= grid_w
        target[:, :, 1] *= grid_h
        gi_array = torch.clamp(torch.clamp(target[:, :, 0], min=0), max=grid_w - 1).long()
        gj_array = torch.clamp(torch.clamp(target[:, :, 1], min=0), max=grid_h - 1).long()

        for batch_ind in range(batch_size):
            for box_ind, target_box in enumerate(target[batch_ind]):
                if sum(target_box[:4]) == 0:
                    continue
                if target_box[4] == 0 or target_box[5] == 0:
                    continue
                if target_box[6] == 0 or target_box[7] == 0:
                    continue
                if target_box[8] == 0 or target_box[9] == 0:
                    continue
                if target_box[10] == 0 or target_box[11] == 0:
                    continue

                iou_anchors = self.iou(target_box[2:], self.anchors)
                anchors_ind = torch.argmax(iou_anchors)

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 0] = \
                    target_box[0] - gi_array[batch_ind, box_ind]

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 1] = \
                    target_box[1] - gj_array[batch_ind, box_ind]

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 2] = \
                    torch.log(target_box[2] / self.anchors[anchors_ind][0])

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 3] = \
                    torch.log(target_box[3] / self.anchors[anchors_ind][1])

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 4] = \
                    torch.log(abs(target_box[4]) / self.anchors[anchors_ind][0])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 5] = \
                    torch.log(abs(target_box[5]) / self.anchors[anchors_ind][1])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 6] = \
                    torch.log(abs(target_box[6]) / self.anchors[anchors_ind][0])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 7] = \
                    torch.log(abs(target_box[7]) / self.anchors[anchors_ind][1])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 8] = \
                    torch.log(abs(target_box[8]) / self.anchors[anchors_ind][0])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 9] = \
                    torch.log(abs(target_box[9]) / self.anchors[anchors_ind][1])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 10] = \
                    torch.log(abs(target_box[10]) / self.anchors[anchors_ind][0])
                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 11] = \
                    torch.log(abs(target_box[11]) / self.anchors[anchors_ind][1])

                self.target_output[
                    batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind, 12] = 1

                if validate:
                    pred = predictions[
                        batch_ind, gj_array[batch_ind, box_ind], gi_array[batch_ind, box_ind], anchors_ind]
                    pred[0] = pred[0] / grid_w
                    pred[1] = pred[1] / grid_h
                    pred[2] = pred[2] * self.anchors[anchors_ind, 0]
                    pred[3] = pred[3] * self.anchors[anchors_ind, 1]
                    target_box[0] = target_box[0] / grid_w
                    target_box[1] = target_box[1] / grid_h

                    if bbox_iou(target_box.unsqueeze(0), pred.unsqueeze(0)) > iou_threshold and pred[
                        -1] > conf_threshold:
                        correct_predictions += 1

        objects = non_zero_ind.sum()

        return correct_predictions, objects, self.target_output


def bbox_iou(box1, box2, x1y1x2y2=False):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                     0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                     0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def nms_np(predictions, conf_thres=0.2, nms_thres=0.2, include_conf=False):
    filter_mask = (predictions[:, -1] >= conf_thres)
    predictions = predictions[filter_mask]

    if len(predictions) == 0:
        return np.array([])

    output = []

    while len(predictions) > 0:
        max_index = np.argmax(predictions[:, -1])

        if include_conf:
            output.append(predictions[max_index])
        else:
            output.append(predictions[max_index, :-1])

        ious = bbox_iou_np(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

        predictions = predictions[ious < nms_thres]

    return np.stack(output)


def bbox_iou_np(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1, 0, None)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


class BBoxUtils:
    def __init__(self):
        self.initiated = False

    def initiate(self, batch_size, grid_h, grid_w):
        self.initiated = True
        self.mask = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_x = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_y = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_w = torch.zeros(batch_size, grid_h, grid_w).float().cuda()
        self.target_h = torch.zeros(batch_size, grid_h, grid_w).float().cuda()

    def zero_out(self):
        self.mask *= 0
        self.target_x *= 0
        self.target_y *= 0
        self.target_w *= 0
        self.target_h *= 0

    def create_targets_bounding_boxes(self, predictions, target, grid_w, grid_h, iou_threshold=0.7,
                                                       conf_threshold=0.5, validate=False):

        batch_size = target.size(0)
        if not self.initiated:
            self.initiate(batch_size, grid_h, grid_w)
        else:
            self.zero_out()

        correct_predictions = 0

        non_zero_ind = target[:, :, :4].sum(axis=2) > 0

        target[:, :, 0::2] *= grid_w
        target[:, :, 1::2] *= grid_h
        gi_array = torch.clamp(torch.clamp(target[:, :, 0], min=0), max=grid_w - 1).long()
        gj_array = torch.clamp(torch.clamp(target[:, :, 1], min=0), max=grid_h - 1).long()

        self.target_x[:, gj_array, gi_array] = target[:, :, 0] - gi_array
        self.target_y[:, gj_array, gi_array] = target[:, :, 1] - gj_array

        ind = torch.Tensor([list(range(batch_size))] * config.MAX_OBJECTS).long().T
        self.target_w[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = torch.log(
            target[:, :, 2][non_zero_ind])
        self.target_h[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = torch.log(
            target[:, :, 3][non_zero_ind])

        self.mask[ind[non_zero_ind], gj_array[non_zero_ind], gi_array[non_zero_ind]] = 1

        if validate:
            for batch_index in range(batch_size):
                for object_index in range(config.MAX_OBJECTS):
                    if target[batch_index, object_index].sum() == 0:
                        continue

                    gt_box = target[batch_index, object_index, :4].unsqueeze(0).cuda()

                    gi = gi_array[batch_index, object_index]
                    gj = gj_array[batch_index, object_index]

                    pred_box = predictions[batch_index, gj, gi]
                    pred_box = torch.FloatTensor([pred_box[0], pred_box[1], pred_box[2], pred_box[3]]).unsqueeze(
                        0).cuda()

                    iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                    if iou > iou_threshold and predictions[batch_index, gj, gi, -1] > conf_threshold:
                        correct_predictions += 1

        objects = non_zero_ind.sum()

        incorrect_predictions = ((1 - self.mask) * predictions[..., -1] > conf_threshold).sum().item()

        return (correct_predictions, incorrect_predictions, objects, self.mask,
                self.target_x, self.target_y, self.target_w, self.target_h)