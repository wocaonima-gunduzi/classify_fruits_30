import os

import torchvision
import torch
from PIL import Image
import time
from model_AlexNet import AlexNet
from model_Resnet18 import Resnet18

# 根据保存方式加载

# 加载方式1  网络与权重 一同保存，才可用此方式加载
# model = torch.load("AlexNet_104.pth", map_location=torch.device('cpu'))

# 加载方式2 只保存权重，才可用此方式加载
# 先加载 网络
# model = AlexNet()
model = Resnet18()

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device=device, dtype=torch.float32)

# 再加载 权重
# strict 权重与层数是否完整符合
model.load_state_dict(torch.load("sg30.pt"), strict=True)

# ['哈密瓜', '圣女果', '山竹', '杨梅', '柚子', '柠檬', '桂圆', '梨', '椰子', '榴莲', '火龙果',
# '猕猴桃', '石榴', '砂糖橘', '胡萝卜', '脐橙', '芒果', '苦瓜', '苹果-红', '苹果-青', '草莓',
# '荔枝', '菠萝', '葡萄-白', '葡萄-红', '西瓜', '西红柿', '车厘子', '香蕉', '黄瓜']

# 指定验证的图像路径
path = r"C:\ce_py_RT\T\Lbw_Cat_Dog\fruit30_split\val\黄瓜"
# 验证的类别名称
# # 定义类别对应字典
dist = {
    0: "哈密瓜", 1: "圣女果", 2: "山竹", 3: "杨梅", 4: "柚子", 5: "柠檬", 6: "桂圆", 7: "梨", 8: "椰子", 9: "榴莲",
    10: "火龙果", 11: "猕猴桃", 12: "石榴", 13: "砂糖橘", 14: "胡萝卜", 15: "脐橙", 16: "芒果", 17: "苦瓜", 18: "苹果-红",
    19: "苹果-青", 20: "草莓", 21: "荔枝", 22: "菠萝", 23: "葡萄-白", 24: "葡萄-红", 25: "西瓜", 26: "西红柿", 27: "车厘子",
    28: "香蕉", 29: "黄瓜"
}
l = "黄瓜"

# ------------------------------------------------------------------------
# 注意更改缩放图像大小、维度转换时的图像大小
imgs = os.listdir(path)
len_imgs = len(imgs)
print(len_imgs)

# 总耗时
mean = 0
# 正确率
acc = 0

for i in imgs:
    # 读取图像
    img = Image.open(os.path.join(path, i))

    # 缩放、格式、归一化
    # transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
    #                                             torchvision.transforms.ToTensor(),
    #                                             torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #                                             ])
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(256),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                ])
    image = transform(img)
    # 注意维度转换，单张图片
    image1 = torch.reshape(image.to(device=device, dtype=torch.float32), (1, 3, 256, 256))

    a = time.time()
    # 测试开关
    model.eval()
    # 节约性能
    with torch.no_grad():
        output = model(image1)
        # print(output)
        # print(output.cuda().cpu())
        # exit()
    # 转numpy格式,列表内取第一个
    a1 = dist[output.cuda().cpu().argmax(1).numpy()[0]]
    if a1 == l:
        acc += 1
    # print(a1, end="    ")
    mean += time.time() - a
    # img.show()

time_mean = mean / len_imgs
print("识别{}张图片，总耗时{}".format(len_imgs, mean))
print("平均耗时：{}".format(time_mean))
print("正确率：{}".format(acc / len_imgs))
