### Modified find_anchors.py to remove anchor calculation ###



import os
import sys
import numpy as np
from shapely.geometry import Polygon

sys.path.append('../')
from data_process import kitti_data_utils
from data_process.yolo8_kitti_bev_utils import build_yolo_target
import src.config.kitti_config as cnf

class FindAnchors:
    def __init__(self, dataset_dir, img_size, num_classes):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.num_classes = num_classes

        self.lidar_dir = os.path.join(self.dataset_dir, 'training', "velodyne")
        self.image_dir = os.path.join(self.dataset_dir, 'training', "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, 'training', "calib")
        self.label_dir = os.path.join(self.dataset_dir, 'training', "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', 'trainval.txt')
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]

        self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)

    def load_targets(self, sample_id):
        """Load images and targets for the training and validation phase"""
        objects = self.get_label(sample_id)
        labels, no_object_labels = kitti_data_utils.read_labels_for_bevbox(objects)
        # Targets formatted as [class, x, y, w, l, sin(yaw), cos(yaw)]
        targets = build_yolo_target(labels, self.img_size, self.num_classes)
        return targets

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples that don't have valid objects for current training class."""
        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, no_object_labels = kitti_data_utils.read_labels_for_bevbox(objects)
            if not no_object_labels:
                labels[:, 1:] = kitti_data_utils.transformation.camera_to_lidar_box(
                    labels[:, 1:], calib.V2C, calib.R0, calib.P
                )
            valid_list = [label[0] for label in labels if int(label[0]) in cnf.CLASS_NAME_TO_ID.values()]
            if valid_list:
                sample_id_list.append(sample_id)
        return sample_id_list

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        return kitti_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        return kitti_data_utils.read_label(label_file)


if __name__ == '__main__':
    dataset_dir = '../../dataset/kitti'
    img_size = (608, 608)
    num_classes = len(cnf.CLASS_NAME_TO_ID)
    anchors_solver = FindAnchors(dataset_dir, img_size, num_classes)
    sample_id = anchors_solver.sample_id_list[0]
    targets = anchors_solver.load_targets(sample_id)
    print(targets)