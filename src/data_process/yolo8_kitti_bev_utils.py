### Modified kitti_bev_utils.py to support YOLOv8's anchor-free approach ###

import numpy as np
from shapely.geometry import Polygon

def build_yolo_target(labels, img_size, num_classes):
    """
    Converts the given labels into YOLOv8 compatible anchor-free targets.

    Args:
        labels: A list of labels, where each label contains [class, x, y, w, l, sin(yaw), cos(yaw)].
        img_size: The size of the image (height, width).
        num_classes: The number of classes for the dataset.

    Returns:
        targets: A tensor with shape [num_objects, 5 + num_classes], including center_x, center_y, width, height, and class confidences.
    """
    targets = []
    for label in labels:
        cls, x, y, w, l, _, _ = label
        center_x = x / img_size[1]  # Normalized to [0, 1]
        center_y = y / img_size[0]  # Normalized to [0, 1]
        width = w / img_size[1]  # Normalized to [0, 1]
        height = l / img_size[0]  # Normalized to [0, 1]

        # Create one-hot encoding for class label
        class_confidences = [0] * num_classes
        class_confidences[int(cls)] = 1

        target = [center_x, center_y, width, height] + class_confidences
        targets.append(target)

    return np.array(targets, dtype=np.float32)

### Utility functions for data processing remain the same ###

# Functions like `read_labels_for_bevbox` and other geometric transformations can remain the same
# as their main role is to preprocess the data for target conversion or visualization.
