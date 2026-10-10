"""第四步：调用 torchvision.models 封装好的模型横向对比"""

import torch
import torch.nn as nn
from torchvision import models
from common import get_loaders, train_and_eval


def build_model(name, num_classes=2, freeze=True):
    """统一接口：加载预训练模型并替换分类头"""
    constructors = {
        "resnet18":           lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "shufflenet_v2_x1_0": lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_x1_0_Weights.DEFAULT),
    }
    net = constructors[name]()

    # 找到并替换分类头（不同模型的头名字不一样）
    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()

    if freeze:
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net


if __name__ == '__main__':
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    results = {}
    for name in ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]:
        print(f"\n===== 模型: {name} =====")
        net = build_model(name)
        acc = train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3)
        results[name] = acc
        torch.save(net.state_dict(), f"model_{name}.pth")

    print("\n===== 横向对比结果 =====")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:24s} 准确率: {acc:.4f}")
