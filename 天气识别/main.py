# -------------------------- 请加载最满意的模型 ---------------------------
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models

# 模型文件路径 — 请将权重放在 results/ 文件夹下
MODEL_PATH = './results/convnext_tiny_320_best.pth'

# 配置（与训练时保持一致）
IM_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABELS = ['cloudy', 'rainy', 'snowy', 'sunny']

device = torch.device('cpu')


class WeatherClassifier(nn.Module):
    """ConvNeXt-Tiny + classification head（与训练时结构一致）"""
    def __init__(self, backbone, num_classes, dropout):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return self.fc(features)


# 构建 backbone（去掉原始分类头）
backbone = models.convnext_tiny(pretrained=False)
backbone.classifier = nn.Identity()

# 加载模型
model = WeatherClassifier(backbone, num_classes=4, dropout=0.3)
state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

# 处理 wrapped checkpoint 格式
if "model_state_dict" in state:
    state = state["model_state_dict"]

model.load_state_dict(state)
model = model.to(device)
model.eval()


def predict(X):
    """
    模型预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(224,224,3)。
    return：
        y_predict : str, 数据 label，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    # cv2.imread 返回 BGR → 转为 RGB（模型用 PIL/RGB 训练）
    X = X[:, :, ::-1]

    # 缩放到模型输入尺寸
    X = cv2.resize(X, (IM_SIZE, IM_SIZE))

    # 归一化：uint8 → float32 [0,1] → ImageNet 标准化
    X = X.astype(np.float32) / 255.0
    X = (X - MEAN) / STD

    # HWC → CHW → 加 batch 维
    X = np.transpose(X, (2, 0, 1))
    X = torch.from_numpy(X).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X)
        pred_idx = torch.argmax(logits, dim=1).item()

    return LABELS[pred_idx]
