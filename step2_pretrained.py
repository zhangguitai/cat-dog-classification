"""第二步：加载 ImageNet 预训练参数，冻结卷积层只训练分类头"""

import torch
import torch.nn as nn
from torchvision import models
from common import get_loaders, train_and_eval


class AlexNetManual(nn.Module):
    """结构与 torchvision.models.alexnet 一致，用于加载预训练权重"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),   # 索引 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # 索引 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # 索引 6
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    batch_size = 16
    lr = 1e-4
    epochs = 10
    device = 'cpu'

    # 1. 先建 1000 类的完整结构，加载预训练权重
    net = AlexNetManual(num_classes=1000)
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net.load_state_dict(official.state_dict())

    # 2. 把分类头换成 2 类（新头自动随机初始化）
    net.classifier[6] = nn.Linear(4096, 2)
    net.to(device)

    # 3. 冻结卷积层，只训练 classifier
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    train_loader, val_loader = get_loaders("./datasets", batch_size=batch_size)
    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
    print(f"\n【第二步 预训练+冻结】最终准确率: {acc:.4f}")
