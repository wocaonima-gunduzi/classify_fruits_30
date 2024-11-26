import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
from model_AlexNet import AlexNet
from model_Resnet18 import Resnet18
import data_read
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

# 使用GPU 可选则
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 只使用GPU
device = torch.device("cuda:0")


# 程序运行中可以通过 watch -n 0.1 -d nvidia-smi 命令来实时查看GPU占用情况，按Ctrl+c退出
# 通过 nvidia-smi 命令来查看某一时刻的GPU的占用情况
# watch -n 0.1 -d nvidia-smi

# 解决缓存问题
# 进入root用户 使用下面命令直接清除缓存
# echo 3 > /proc/sys/vm/drop_caches


# 查看内存使用情况，注意个人的电脑情况
# free -h

# 旧版 学习使用的加载数据:训练集、测试集
# # 训练集
# train_data = data_read.datasets("./data/train")
# # 测试集
# val_data = data_read.datasets("./data/val")

# 一组包含几张图像
batch_size = 32

# 项目常用加载数据集方式
# 训练集与测试集的路径
ROOT_TRAIN = './fruit30_split/train'
ROOT_TEST = './fruit30_split/val'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    # 随机旋转，-45度到45度之间随机选
    transforms.RandomRotation(45),
    # 从中心开始裁剪
    transforms.CenterCrop(224),
    # 随机水平翻转 选择概率值为 p=0.5
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机垂直翻转
    transforms.RandomVerticalFlip(p=0.5),
    # 参数：亮度、对比度、饱和度、色相
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    # 转为3通道灰度图 R=G=B 概率设定0.025
    transforms.RandomGrayscale(p=0.025),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # 随机旋转，-45度到45度之间随机选
    transforms.RandomRotation(45),
    # 从中心开始裁剪
    transforms.CenterCrop(224),
    # 随机水平翻转 选择概率值为 p=0.5
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机垂直翻转
    transforms.RandomVerticalFlip(p=0.5),
    # 参数：亮度、对比度、饱和度、色相
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    # 转为3通道灰度图 R=G=B 概率设定0.025
    transforms.RandomGrayscale(p=0.025),
    transforms.ToTensor(),
    # 将图像的像素值归一化到【-1， 1】之间
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_data = ImageFolder(ROOT_TEST, transform=val_transform)
# ------------------------------------------------------------------
# 查看各类别名称
class_names = train_data.classes
print(class_names)

# 打包
train_datas = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
val_datas = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

# 导入网络 可选
# model = AlexNet()
model = Resnet18()


# 断点续训 加载以前的权重参数，第一次训练则无需加载
# model.load_state_dict(torch.load("Resnet18_75_0.83.pth"), strict=True)

# 使用GPU
model = model.to(device)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 使用GPU
loss_fn = loss_fn.to(device)

# 优化器
# 学习率
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

# 训练轮次
epoch = 300
# writer = SummaryWriter("logs")
# 准确率 总数据
train_data_len = len(train_data)
val_data_len = len(val_data)

# 模型名字 根据准确率 删除
model_name_acc = ""
# 模型名字 根据轮次 删除
model_name_epoch = ""
# 准确率列表
acc_list = []
if __name__ == '__main__':

    # 训练轮次
    for i in range(epoch):
        optimizer.zero_grad()
        lr_scheduler.step()

        # 训练开关
        model.train(mode=True)
        # 准确个数累计
        train_acc = 0
        # 累计loss
        train_loss = 0
        n = 1
        for data in train_datas:
            imgs, targets = data
            # 使用GPU
            imgs1 = imgs.to(device)
            targets1 = targets.to(device)

            # 数据给模型
            outputs = model(imgs1)
            loss = loss_fn(outputs, targets1)

            # 优化损失   清零、反向传播、优化器启动
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("\r{}".format(loss.item()), end="")
            # 显示
            # 准确率
            train_acc += (outputs.argmax(1) == targets1).sum().item()
            # 累计loss
            train_loss += loss.item()

            print("\r训练次数:{}，Loss:{}, acc：{}".format(n, loss.item(), train_acc / (n * batch_size)), end="")
            n += 1

        print()
        print("Loss:{}, 准确率：{}".format(train_loss, train_acc / train_data_len))
        # 绘图
        # writer.add_scalar("train", train_loss, i)

        # 测试开关
        model.eval()
        # 准确个数累计
        val_acc = 0
        # 累计loss
        val_loss = 0
        n = 1
        with torch.no_grad():
            for data in val_datas:
                imgs, targets = data
                # 使用GPU
                imgs1 = imgs.to(device)
                targets1 = targets.to(device)

                # 数据给模型
                outputs = model(imgs1)

                loss = loss_fn(outputs, targets1)
                # 准确率
                val_acc += (outputs.argmax(1) == targets1).sum().item()

                # 累计loss
                val_loss += loss.item()

                print("\r测试次数:{}，Loss:{}, acc：{}".format(n, loss.item(), val_acc / (n * batch_size)), end="")
                n += 1

        print()
        # 测试集 loss
        acc = val_acc / (n * batch_size)
        print("测试集 Loss:{}, 准确率：{}".format(val_loss, acc))
        # 绘图
        # writer.add_scalar("val", val_loss, i)

        # 第1轮保存模型
        model_name = "Resnet18_{}_{:.2f}.pth".format(i, acc)

        # 保存模型  只保存权重参数，没有保存网络架构
        # 轮次保存
        torch.save(model.state_dict(), model_name)
        print("第{}轮模型已保存".format(i + 1))

        # 放入
        acc_list.append(val_loss)
        # loss 最小保存
        if val_loss == np.min(acc_list):
            acc_max = "Resnet18_{}_{:.2f}_acc_max.pth".format(i, acc)
            torch.save(model.state_dict(), acc_max)

            # 删除上一个准确率
            if model_name_acc != "":
                os.remove(model_name_acc)
            # 将模型名字给外面
            model_name_acc = acc_max

        # 删除上一个轮次保存的模型，来保护缓存
        if i >= 1:
            os.remove(model_name_epoch)

        # 将保存的模型名字给外面
        model_name_epoch = model_name

    # writer.close()
