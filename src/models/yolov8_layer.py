### Modified darknet2pytorch.py to support YOLOv8 ###

import torch
import torch.nn as nn


class YOLOv8DetectionLayer(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8DetectionLayer, self).__init__()
        self.num_classes = num_classes
        self.strides = [8, 16, 32]  # Example strides for different feature maps

    def forward(self, feature_maps):
        predictions = []
        for i, feature_map in enumerate(feature_maps):
            batch_size, num_channels, h, w = feature_map.shape
            stride = self.strides[i]

            # YOLOv8 anchor-free head directly regresses to center points, sizes, and class confidences
            feature_map = feature_map.permute(0, 2, 3, 1).contiguous()
            feature_map = feature_map.view(batch_size, -1, num_channels)

            # Here num_channels should include center_x, center_y, width, height, and num_classes
            center_x = torch.sigmoid(feature_map[..., 0]) * stride
            center_y = torch.sigmoid(feature_map[..., 1]) * stride
            width = torch.exp(feature_map[..., 2]) * stride
            height = torch.exp(feature_map[..., 3]) * stride
            class_scores = torch.sigmoid(feature_map[..., 4:4 + self.num_classes])

            # Combine into the final output shape
            prediction = torch.cat([center_x.unsqueeze(-1), center_y.unsqueeze(-1),
                                    width.unsqueeze(-1), height.unsqueeze(-1), class_scores], dim=-1)
            predictions.append(prediction)

        return torch.cat(predictions, dim=1)  # Combine all predictions from all feature maps


class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()
        self.backbone = YOLOv8Backbone(num_classes)
        self.head = YOLOv8DetectionLayer(num_classes)

    def forward(self, x):
        # Get feature maps from backbone and pass to the detection head
        feature_maps = self.backbone(x)
        output = self.head(feature_maps)
        return output