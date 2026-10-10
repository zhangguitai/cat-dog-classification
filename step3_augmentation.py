"""第三步：加入数据增强，与预训练模型配合使用"""

from torchvision import transforms, models
import torch.nn as nn
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD
from step2_pretrained import AlexNetManual


if __name__ == '__main__':
    batch_size = 16
    lr = 1e-3
    epochs = 5
    device = 'cpu'

    # 数据增强：只作用于训练集
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),   # 随机裁剪缩放
        transforms.RandomHorizontalFlip(),                      # 随机水平翻转
        transforms.ColorJitter(0.2, 0.2, 0.2),                  # 随机亮度/对比度/饱和度
        transforms.RandomRotation(10),                          # 随机小角度旋转
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader, val_loader = get_loaders(
        "./datasets", batch_size=batch_size, train_transform=train_transform)

    # 使用预训练 AlexNet + 冻结卷积
    net = AlexNetManual(num_classes=1000)
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net.load_state_dict(official.state_dict())
    net.classifier[6] = nn.Linear(4096, 2)
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
    print(f"\n【第三步 预训练+数据增强】最终准确率: {acc:.4f}")
