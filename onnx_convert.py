import torch
from torchvision import models
from model_Resnet18 import Resnet18

# 模型转换为onnx

# 加载网络
model = Resnet18()
# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device=device, dtype=torch.float32)

# 再加载 权重
# strict 权重与层数是否完整符合
model.load_state_dict(torch.load("sg30.pt"), strict=True)
model = model.eval().to(device)


x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,                   # 要转换的模型
        x,                       # 模型的任意一组输入
        'resnet18_fruit30.onnx', # 导出的 ONNX 文件名
        opset_version=11,        # ONNX 算子集版本
        input_names=['input'],   # 输入 Tensor 的名称（自己起名字）
        output_names=['output']  # 输出 Tensor 的名称（自己起名字）
    )