import os
import time

import torch
from tqdm import tqdm
from ultralytics import YOLO  # 使用 YOLOv8
from config.yolo8_train_config import parse_train_configs
from data_process.kitti_dataloader import create_train_dataloader
from utils.misc import AverageMeter, ProgressMeter
import torch.nn.functional as F

def main():
    configs = parse_train_configs()

    # 设置随机种子
    if configs.seed is not None:
        torch.manual_seed(configs.seed)

    # 创建 YOLOv8 模型
    model = YOLO('yolov8n.pt')
    model.to(configs.device)

    # 创建自定义的数据加载器
    train_dataloader, _ = create_train_dataloader(configs)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(configs.num_epochs):
        model.model.train()
        running_loss = 0.0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                                 prefix="Train - Epoch: [{}/{}]".format(epoch + 1, configs.num_epochs))

        start_time = time.time()
        for batch_idx, (_, imgs, targets) in enumerate(tqdm(train_dataloader)):
            data_time.update(time.time() - start_time)

            imgs = imgs.to(configs.device)
            # 将 targets 转换为张量类型，并移动到适当的设备上
            if isinstance(targets, list):
                targets = [
                    torch.tensor(target, device=configs.device) if not isinstance(target, torch.Tensor) else target for
                    target in targets]
                targets = torch.cat(targets, dim=0)  # 拼接目标列表以形成一个张量

            # 前向传播
            preds = model(imgs)

            # 提取预测结果
            pred_boxes = []
            pred_classes = []
            for pred in preds:
                if isinstance(pred, torch.Tensor):
                    pred_boxes.append(pred)
                else:
                    # 提取结果对象中的检测框数据
                    pred_boxes.append(pred.boxes.xyxy)
                    pred_classes.append(pred.boxes.cls)  # 类别预测

            # 将所有预测拼接为一个张量
            if len(pred_boxes) > 0:
                pred_boxes = torch.cat(pred_boxes, dim=0)
                pred_classes = torch.cat(pred_classes, dim=0)

            # 确保预测框和目标框形状匹配
            if pred_boxes.shape[0] != targets.shape[0]:
                continue

            # 计算边界框损失 (使用 MSE 作为占位符，可以根据需要替换为 GIoU 损失)
            bbox_loss = F.mse_loss(pred_boxes[:, :4], targets[:, 1:5])

            # 计算类别损失 (交叉熵损失)
            class_loss = F.cross_entropy(pred_classes, targets[:, 0].long())

            # 总损失
            loss = bbox_loss + class_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.update(loss.item(), imgs.size(0))

            # 记录时间
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if batch_idx % configs.print_freq == 0:
                progress.display(batch_idx)

                # 学习率调度
        lr_scheduler.step()
        print(f"Epoch [{epoch + 1}/{configs.num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

    print("训练完成！")

if __name__ == "__main__":
    main()

