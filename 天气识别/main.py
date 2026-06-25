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


def _build_model():
    backbone = models.convnext_tiny(pretrained=False)
    backbone.classifier = nn.Identity()
    return WeatherClassifier(backbone, num_classes=4, dropout=0.3).to(device).eval()


def _preprocess(X, im_size):
    X = X[:, :, ::-1]
    X = cv2.resize(X, (im_size, im_size))
    X = X.astype(np.float32) / 255.0
    X = (X - MEAN) / STD
    X = np.transpose(X, (2, 0, 1))
    return torch.from_numpy(X).unsqueeze(0).to(device)


# ---- 加载模型 ----
state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

_ensemble_mode = False
_models = []
_weights = []
_image_sizes = []

if isinstance(state, dict) and state.get('type') == 'logits_ensemble':
    # Ensemble 模式：加载多个子模型
    _ensemble_mode = True
    for m in state['members']:
        model = _build_model()
        model.load_state_dict(m['state_dict'])
        _models.append(model)
        _weights.append(m['weight'])
        _image_sizes.append(m.get('image_size', IM_SIZE))
else:
    # 单模型模式
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = _build_model()
    model.load_state_dict(state)
    _models = [model]
    _weights = [1.0]
    _image_sizes = [IM_SIZE]


def predict(X):
    """
    模型预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(H,W,3)。
    return：
        y_predict : str, 数据 label，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    logits_sum = None

    for model, weight, im_size in zip(_models, _weights, _image_sizes):
        x = _preprocess(X, im_size)
        with torch.no_grad():
            logits = model(x)
        if logits_sum is None:
            logits_sum = logits * weight
        else:
            logits_sum += logits * weight

    pred_idx = torch.argmax(logits_sum, dim=1).item()
    return LABELS[pred_idx]
