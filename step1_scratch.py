"""第一步：手动搭建 AlexNet，从零训练"""

import torch
import torch.nn as nn
from common import get_loaders, train_and_eval


class AlexNet(nn.Module):
    """精简版 AlexNet，从零训练（无预训练权重）"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 11x11 大窗口，步幅4，快速降采样
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 5x5 卷积，padding=2 保持尺寸
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 三个连续 3x3 卷积
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 全连接层 + Dropout 防过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2),  # 二分类输出
        )

    def forward(self, X):
        return self.net(X)


def init_weight(m):
    """Xavier 均匀初始化"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    batch_size = 16
    lr = 1e-4       # 从零训练，用较小学习率
    epochs = 30
    device = 'cpu'

    net = AlexNet()
    net.apply(init_weight)
    net.to(device)

    train_loader, val_loader = get_loaders("./datasets", batch_size=batch_size)
    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
    print(f"\n【第一步 从零训练 AlexNet】最终准确率: {acc:.4f}")
