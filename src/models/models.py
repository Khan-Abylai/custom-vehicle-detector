import torch.nn as nn
import torch
try:
    from src.models.block import LinearConvBlock, OrdinaryConvBlock, CarYoloBlock
    import src.config as config
except:
    from models.block import LinearConvBlock, OrdinaryConvBlock, CarYoloBlock
    import config as config

class LPDetector(nn.Module):

    def __init__(self, image_size, cuda=None):
        super().__init__()

        self.plate_features = nn.Sequential(OrdinaryConvBlock(3, 16), OrdinaryConvBlock(16, 16),
                                            nn.MaxPool2d(kernel_size=2, stride=2), OrdinaryConvBlock(16, 32),
                                            OrdinaryConvBlock(32, 32), nn.MaxPool2d(kernel_size=2, stride=2),
                                            OrdinaryConvBlock(32, 64), OrdinaryConvBlock(64, 64),
                                            nn.MaxPool2d(kernel_size=2, stride=2), OrdinaryConvBlock(64, 128),
                                            OrdinaryConvBlock(128, 128), nn.MaxPool2d(kernel_size=2, stride=2),
                                            OrdinaryConvBlock(128, 256), OrdinaryConvBlock(256, 256), )

        self.car_features = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), OrdinaryConvBlock(256, 512),
                                          OrdinaryConvBlock(512, 512), nn.MaxPool2d(kernel_size=2, stride=2),
                                          OrdinaryConvBlock(512, 1024), OrdinaryConvBlock(1024, 1024), )

        self.yolo_cars = nn.ModuleList()
        self.yolo_cars.append(
            LinearConvBlock(in_channels=1024, out_channels=config.CAR_COORDINATE_DIMENSIONS, kernel_size=1, stride=1))
        self.yolo_cars.append(
            CarYoloBlock(image_size, grid_size=64, conf_scale=2, coord_scale=2, noobject_scale=0.2, iou_threshold=0.85,
                         conf_threshold=0.8, cuda=cuda))

    def forward(self, imgs, car_boxes=None, validate=False):
        yolo_plates_output = self.plate_features(imgs)
        yolo_cars_output = self.car_features(yolo_plates_output)

        for module in self.yolo_cars:
            yolo_cars_output = module(yolo_cars_output, car_boxes, validate=validate)

        return yolo_cars_output
