import onnx

# 读取 ONNX 模型
onnx_model = onnx.load('resnet18_fruit30.onnx')

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

print('无报错，onnx模型载入成功')

# print(onnx.helper.printable_graph(onnx_model.graph))
# 这里就不分2个文件了，自己注释一下吧

import onnxruntime
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

dist = {
    0: "哈密瓜", 1: "圣女果", 2: "山竹", 3: "杨梅", 4: "柚子", 5: "柠檬", 6: "桂圆", 7: "梨", 8: "椰子", 9: "榴莲",
    10: "火龙果", 11: "猕猴桃", 12: "石榴", 13: "砂糖橘", 14: "胡萝卜", 15: "脐橙", 16: "芒果", 17: "苦瓜", 18: "苹果-红",
    19: "苹果-青", 20: "草莓", 21: "荔枝", 22: "菠萝", 23: "葡萄-白", 24: "葡萄-红", 25: "西瓜", 26: "西红柿", 27: "车厘子",
    28: "香蕉", 29: "黄瓜"
}
ort_session = onnxruntime.InferenceSession('resnet18_fruit30.onnx')
x = torch.randn(1, 3, 256, 256).numpy()
# onnx runtime 输入
ort_inputs = {'input': x}

# onnx runtime 输出
ort_output = ort_session.run(['output'], ort_inputs)[0]
print(ort_output.shape)

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ])
img_path = '7.jpeg'

# 用 pillow 载入
img_pil = Image.open(img_path)

input_img = test_transform(img_pil)

print(input_img.shape)
input_tensor = input_img.unsqueeze(0).numpy()
print(input_tensor.shape)

# ONNX Runtime 输入
ort_inputs = {'input': input_tensor}
# ONNX Runtime 输出
pred_logits = ort_session.run(['output'], ort_inputs)[0]
pred_logits = torch.tensor(pred_logits)
print(pred_logits.shape)
pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
print(pred_softmax.shape)
n = 3  # 取前三个最大分数
top_n = torch.topk(pred_softmax, n)
# print(top_n)

pred_ids = top_n.indices.numpy()[0]
print(pred_ids)
print(dist[pred_ids[0]])
# # 预测置信度
# confs = top_n.values.numpy()[0]
# print(confs)
# 载入类别和对应 ID
# idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# for i in range(n):
#     class_name = idx_to_labels[pred_ids[i]] # 获取类别名称
#     confidence = confs[i] * 100             # 获取置信度
#     text = '{:<6} {:>.3f}'.format(class_name, confidence)
#     print(text)
