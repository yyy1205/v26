#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把torchreid的OSNet-AIN(.pth.tar)权重导出成ONNX，方便走ultralytics现成的
AutoBackend加载路径，不需要把torchreid装进跟踪运行的环境里。
"""
import sys
sys.path.insert(0, r"D:\deep-person-reid-master")
import os
import torch
import torchreid

CKPT_PATH = r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-100"
OUT_DIR = r"D:\yolo26-tracking\reid_model"
OUT_PATH = os.path.join(OUT_DIR, "osnet_ain_x1_0_vehicle.onnx")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_H, IMG_W = 256, 128  # torchreid FeatureExtractor 默认 image_size=(256,128)

print("构建模型骨架 (num_classes=751, Market1501 标准)...")
model = torchreid.models.build_model(
    name="osnet_ain_x1_0",
    num_classes=751,
    loss="softmax",
    pretrained=False,
)
print(f"加载权重: {CKPT_PATH}")
torchreid.utils.load_pretrained_weights(model, CKPT_PATH)
model.eval()

# 确认 eval 模式下 forward 直接吐 embedding(不含分类头)，和 FeatureExtractor 实际推理路径一致
dummy_raw = torch.randn(1, 3, IMG_H, IMG_W)
with torch.no_grad():
    out = model(dummy_raw)
print(f"eval模式下模型输出shape: {out.shape} (应为 (1, 特征维度)，不应该是分类logits(1,751))")
assert out.dim() == 2 and out.shape[1] != 751, "看起来输出的是分类头而不是embedding，需要检查forward逻辑"


class NormalizedOSNet(torch.nn.Module):
    """把ImageNet标准化焼进图里：输入是 [0,1] 缩放过的原始RGB(现有tracker pipeline只做了/255，
    没做mean/std标准化)，这里内部自己减均值除方差，这样导出的onnx可以直接接现成的AutoBackend加载路径，
    不用去改共享的预处理代码。
    """

    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


wrapped = NormalizedOSNet(model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
wrapped.eval()

dummy_01 = torch.rand(1, 3, IMG_H, IMG_W)  # 未标准化、[0,1]范围，模拟现有pipeline的/255.0输出
with torch.no_grad():
    out_wrapped = wrapped(dummy_01)
print(f"包装后(带内置标准化)模型输出shape: {out_wrapped.shape}")

print(f"导出ONNX到: {OUT_PATH}")
torch.onnx.export(
    wrapped,
    dummy_01,
    OUT_PATH,
    input_names=["images"],
    output_names=["embedding"],
    dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
)
print("导出完成。")
