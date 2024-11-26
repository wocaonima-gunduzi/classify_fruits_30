import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Dropout


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=(4, 4), padding=2),
            ReLU(),
            Conv2d(48, 128, (5, 5), (1, 1), 2),
            ReLU(),
            MaxPool2d(2),
            Conv2d(128, 192, (3, 3), (1, 1), 1),
            ReLU(),
            MaxPool2d(2),
            Conv2d(192, 192, (3, 3), (1, 1), 1),
            ReLU(),
            Conv2d(192, 128, (3, 3), (1, 1), 1),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            Linear(4608, 2048),
            Dropout(p=0.5),
            Linear(2048, 2048),
            Dropout(p=0.5),
            Linear(2048, 1000),
            Dropout(p=0.5),
            # 添加一层 二分类
            Linear(1000, 2),
        )

    # 这个 forward 函数名是固定不可更改的
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = AlexNet()
    x = torch.ones([1, 3, 224, 224])
    y = model(x)
    print(y)
    print(y.shape)
