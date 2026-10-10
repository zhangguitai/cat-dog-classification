"""后端推理服务：FastAPI 提供 /predict 接口，同时托管前端页面"""

import io
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

# ---- 模型加载（启动时只做一次） ----
CLASSES = {0: "猫", 1: "狗"}
WEIGHT_PATH = "best_model.pth"

def load_model(weight_path=WEIGHT_PATH):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    return net

# 如果权重文件存在就加载，否则提示先训练
if os.path.exists(WEIGHT_PATH):
    net = load_model()
    print(f"[INFO] 模型已加载: {WEIGHT_PATH}")
else:
    net = None
    print(f"[WARNING] 未找到 {WEIGHT_PATH}，请先运行 python step5_save_infer.py 训练模型")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- FastAPI 应用 ----
app = FastAPI(title="猫狗分类 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html not found</h1>"


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok", "model_loaded": net is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """接收上传图片，返回猫狗分类结果"""
    if net is None:
        return JSONResponse(
            status_code=503,
            content={"error": "模型未加载，请先运行 python step5_save_infer.py"}
        )
    # 读取图片
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 预处理
    x = transform(img).unsqueeze(0)
    # 推理
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    pred = probs.argmax().item()
    return {
        "prediction": CLASSES[pred],
        "confidence": f"{probs[pred].item():.2%}",
        "probabilities": {
            "cat": f"{probs[0].item():.2%}",
            "dog": f"{probs[1].item():.2%}",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
