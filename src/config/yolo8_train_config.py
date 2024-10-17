# yolo8_train_config.py
import argparse
import torch

def parse_train_configs():
    parser = argparse.ArgumentParser(description='YOLOv8 Training Configuration')

    # General training configuration
    parser.add_argument('--epochs', type=int, default=20, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=608, help='Input image size for training and evaluation')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of target classes')
    parser.add_argument('--data_augmentation', type=bool, default=True, help='Use data augmentation during training')

    # Add missing attributes for working directory and random seed
    parser.add_argument('--working_dir', type=str, default='../', help='The working directory of the project')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--saved_fn', type=str, default='yolov8_model', help='Name of the saved model file')
    parser.add_argument('--arch', type=str, default='yolov8', choices=['darknet', 'yolov4', 'yolov8'],
                        help='Model architecture to use for training')
    parser.add_argument('--hflip_prob', type=float, default=0.5, help='Probability of applying horizontal flip augmentation')

    # Cutout augmentation settings
    parser.add_argument('--cutout_nholes', type=int, default=1, help='Number of holes to cut out from image')
    parser.add_argument('--cutout_ratio', type=float, default=0.2, help='Ratio of each hole relative to image size')
    parser.add_argument('--cutout_fill_value', type=int, default=0, help='Pixel value to fill the cutout regions')
    parser.add_argument('--cutout_prob', type=float, default=0.5, help='Probability of applying cutout augmentation')

    # Dataset settings
    parser.add_argument('--dataset_dir', type=str, default='../dataset/kitti', help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples used in training dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads for loading data')
    parser.add_argument('--random_padding', type=bool, default=False, help='Use random padding when using mosaic augmentation')

    # Optimizer settings
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Type of optimizer to use during training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (ignored if using adamw)')

    # Learning rate scheduler settings
    parser.add_argument('--lr_type', type=str, default='cosine', choices=['cosine', 'multi_step'],
                        help='Type of learning rate scheduler')
    parser.add_argument('--burn_in', type=int, default=1000, help='Number of iterations to gradually increase the learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs for cosine scheduler')

    # Data augmentation settings
    parser.add_argument('--mosaic', type=bool, default=True, help='Use mosaic augmentation (YOLOv8 specific)')
    parser.add_argument('--mixup', type=bool, default=True, help='Use mixup augmentation (YOLOv8 specific)')
    parser.add_argument('--multiscale_training', type=bool, default=True, help='Use multiscale training during training')

    # Distributed training settings
    # Device setting
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--gpu_idx', type=int, default=0, help='Index of the GPU to use for training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend for PyTorch')
    parser.add_argument('--dist_url', type=str, default='env://', help='URL for setting up distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--rank', type=int, default=-1, help='Rank for distributed training')
    parser.add_argument('--ngpus_per_node', type=int, default=1, help='Number of GPUs per node for training')
    parser.add_argument('--multiprocessing_distributed', type=bool, default=False, help='Use multi-processing distributed training')

    # Distributed training settings
    parser.add_argument('--distributed', type=bool, default=False, help='Use distributed training')

    # Paths
    parser.add_argument('--data_path', type=str, default='../dataset/kitti', help='Path to dataset directory')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='Directory for saving logs and model checkpoints')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Directory for saving model checkpoints')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume training from a checkpoint')

    # Evaluation settings
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on validation set')
    parser.add_argument('--no_val', action='store_true', help='Disable validation during training')

    # Frequency settings
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency during training')
    parser.add_argument('--tensorboard_freq', type=int, default=100, help='Frequency to log metrics to TensorBoard')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Save a checkpoint every N epochs')

    # Miscellaneous
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory in DataLoader for faster data transfer to GPU')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from, used for resuming')

    configs = parser.parse_args()
    return configs

if __name__ == '__main__':
    configs = parse_train_configs()
    print(configs)