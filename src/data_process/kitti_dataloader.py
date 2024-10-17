"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout


def create_yolotrain_dataloader(configs):
    """用于根据 configs 创建训练数据加载器"""

    # 定义了一组Lidar点云数据的随机变换
    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    # 图像数据的增强操作
    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)

    # 实例化训练数据集对象
    train_dataset = KittiDataset(configs.dataset_dir, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=configs.multiscale_training,
                                 num_samples=configs.num_samples, mosaic=configs.mosaic,
                                 random_padding=configs.random_padding)

    # 初始化数据采样器
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.yolo_collate_fn)  # 使用yolo_collate_fn

    return train_dataloader, train_sampler


def create_train_dataloader(configs):
    """用于根据 configs 创建训练数据加载器"""

    # 定义了一组Lidar点云数据的随机变换，使用了旋转和缩放操作。
    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    # 定义了一组图像数据的增强操作，包括水平翻转和Cutout。
    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)

    # 实例化训练数据集对象
    train_dataset = KittiDataset(configs.dataset_dir, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=configs.multiscale_training,
                                 num_samples=configs.num_samples, mosaic=configs.mosaic,
                                 random_padding=configs.random_padding)

    # 初始化数据采样器，若启用分布式训练则使用 DistributedSampler，用于在多个 GPU 之间划分数据
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 创建训练数据加载器。参数包括数据集、批量大小、是否打乱数据（当无采样器时打乱）、是否使用固定内存、并行加载线程数、以及自定义的 collate_fn 用于整理批次数据
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    # 返回训练数据加载器和采样器
    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """用于根据 configs 配置创建验证数据加载器"""

    # 实例化验证数据集对象，不应用任何数据增强或多尺度训练
    val_sampler = None
    val_dataset = KittiDataset(configs.dataset_dir, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    # 如果启用了分布式训练，则使用 DistributedSampler
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    # 创建验证数据加载器，设置 shuffle=False 不打乱数据
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)
    # 返回验证数据加载器
    return val_dataloader


def create_test_dataloader(configs):
    """用于根据 configs 配置创建测试数据加载器"""

    # 测试数据加载器与验证数据加载器类似，但 mode 设置为 'test'
    test_dataset = KittiDataset(configs.dataset_dir, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np
    from easydict import EasyDict as edict

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.kitti_config as cnf

    # 使用 argparse 定义一系列命令行参数，用于控制数据增强和训练过程的配置
    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--hflip_prob', type=float, default=0.,
                        help='The probability of horizontal flip')
    parser.add_argument('--cutout_prob', type=float, default=0.,
                        help='The probability of cutout augmentation')
    parser.add_argument('--cutout_nholes', type=int, default=1,
                        help='The number of cutout area')
    parser.add_argument('--cutout_ratio', type=float, default=0.3,
                        help='The max ratio of the cutout area')
    parser.add_argument('--cutout_fill_value', type=float, default=0.,
                        help='The fill value in the cut out area, default 0. (black)')
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--mosaic', action='store_true',
                        help='If true, compose training samples as mosaics')
    parser.add_argument('--random-padding', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--show-train-data', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--save_img', action='store_true',
                        help='If true, save the images')

    # 加载数据集并且可视化
    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')

    if configs.save_img:
        print('saving validation images')
        configs.saved_dir = os.path.join(configs.dataset_dir, 'validation_data')
        if not os.path.isdir(configs.saved_dir):
            os.makedirs(configs.saved_dir)

    if configs.show_train_data:
        dataloader, _ = create_train_dataloader(configs)
        print('len train dataloader: {}'.format(len(dataloader)))
    else:
        dataloader = create_val_dataloader(configs)
        print('len val dataloader: {}'.format(len(dataloader)))

    print('\n\nPress x to see the next sample >>> Press Esc to quit...')

    for batch_i, (img_files, imgs, targets) in enumerate(dataloader):
        if not (configs.mosaic and configs.show_train_data):
            img_file = img_files[0]
            img_rgb = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, RGB_Map=None)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        # Rescale target
        targets[:, 2:6] *= configs.img_size
        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        img_bev = imgs.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))

        for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
            # Draw rotated box
            bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)

        if configs.mosaic and configs.show_train_data:
            if configs.save_img:
                fn = os.path.basename(img_file)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), img_bev)
            else:
                cv2.imshow('mosaic_sample', img_bev)
        else:
            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=configs.output_width)
            if configs.save_img:
                fn = os.path.basename(img_file)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), out_img)
            else:
                cv2.imshow('single_sample', out_img)

        if not configs.save_img:
            if cv2.waitKey(0) & 0xff == 27:
                break
