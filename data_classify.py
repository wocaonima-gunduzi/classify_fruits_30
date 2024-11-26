import argparse
import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 划分数据集

# './data_name'
def data_list(path, percentage, name):
    # 获取data文件夹下所有文件夹名（即需要分类的类名）
    file_path = path
    flower_class = [cla for cla in os.listdir(file_path)]

    # # 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
    pwd2 = name + "/train"
    mkfile(name)
    for cla in flower_class:
        mkfile(pwd2 + "/" + cla)

    # 创建 验证集val 文件夹，并由类名在其目录下创建子目录
    pwd3 = name + "/val"
    mkfile(name)
    for cla in flower_class:
        mkfile(pwd3 + "/" + cla)

    # 划分比例，训练集 : 验证集 = 9 : 1
    split_rate = percentage

    # 遍历所有类别的全部图像并按比例分成训练集和验证集
    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
        images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
        for index, image in enumerate(images):
            # eval_index 中保存验证集val的图像名称
            if image in eval_index:
                image_path = cla_path + image
                new_path = pwd3 + "/" + cla
                copy(image_path, new_path)  # 将选中的图像复制到新路径

            # 其余的图像保存在训练集train中
            else:
                image_path = cla_path + image
                new_path = pwd2 + "/" + cla
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="开始分离\"训练集\"与\"测试集\"百分比,"
                                                 "默认读取同级目录文件名：data_name,"
                                                 "默认训练集80%，测试集20%"
                                                 "默认保存文件名：data"
                                                 "train-->训练集"
                                                 "val  -->测试集")
    parser.add_argument('--path', type=str, default="./data_name", help='输入目标文件的路径')
    parser.add_argument('--percentage', type=float, default=0.2, help='指定测试集比例，例如:"0.2",训练集80%，测试集20%')
    parser.add_argument('--name', type=str, default="./data", help='另存为命名')
    args = parser.parse_args()
    path, percentage, name = args.path, args.percentage, args.name
    data_list(path, percentage, name)
