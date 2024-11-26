# 数据加载
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms


class MyData(Dataset):

    def __init__(self, root_dir, label_dir, i):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

        # 定义转换字典
        self.__dict = {}
        self.__dict[label_dir] = i

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)

        img = Image.open(img_item_path)

        # 按照模型需求 统一缩放图像大小
        trans_resize_1 = transforms.Resize((224, 224))
        img_1 = trans_resize_1(img)

        # 对缩放后的图像做数据增强（180，旋转，一半概率）
        trans_ = transforms.RandomVerticalFlip()
        img_2 = trans_(img_1)

        # 数据格式转换
        tensor = transforms.ToTensor()
        # 数据格式转换对象(图像)
        tensor_img = tensor(img_2)
        # 归一化
        trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_norm = trans_norm(tensor_img)

        label = self.label_dir

        # 字典将字符串 找对的数
        label_int = self.__dict[label]

        # 图像--标签
        return img_norm, label_int

    def __len__(self):
        return len(self.img_path)


# 训练集 或 测试集 读取
# "./data/train"
def datasets(path):
    list_dir_train = path
    list_t = os.listdir(list_dir_train)
    list_t_img = MyData(list_dir_train, list_t[0], 0)
    for i in range(len(list_t)):
        if i == 0:
            continue
        list_t_img += MyData(list_dir_train, list_t[(i)], i)
        # 运行测试
        # img1, bb1 = list_t_img[0]
        # img1.show()
    return list_t_img
