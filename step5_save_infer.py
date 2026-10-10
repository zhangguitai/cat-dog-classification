"""第五步：训练时保存最佳模型 + 单张图片推理测试"""

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from common import get_loaders, IMAGENET_MEAN, IMAGENET_STD

CLASSES = ["cat 猫", "dog 狗"]


def train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3,
                         save_path="best_model.pth"):
    """训练并在验证集准确率最高时保存模型"""
    device = "cpu"
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()

        # 验证
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                correct += (net(X).argmax(1) == Y).sum().item()
                total += Y.size(0)
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f"  ^ 最佳模型已保存到 {save_path}")
    print(f"最佳准确率: {best_acc:.4f}")
    return best_acc


def load_model(weight_path="best_model.pth"):
    """加载保存的模型权重"""
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    return net


def predict(image_path, net):
    """单张图片推理"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # -> [1,3,224,224]
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item(), probs


if __name__ == '__main__':
    # --- 训练并保存最佳模型 ---
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True

    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3)

    # --- 单张图片推理测试 ---
    print("\n===== 单张图片推理测试 =====")
    model = load_model("best_model.pth")
    test_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/cat.1.jpg"
    label, conf, probs = predict(test_path, model)
    print(f"图片: {test_path}")
    print(f"预测: {label}，置信度 {conf:.2%}")
    print(f"各类概率: 猫 {probs[0]:.2%} | 狗 {probs[1]:.2%}")
