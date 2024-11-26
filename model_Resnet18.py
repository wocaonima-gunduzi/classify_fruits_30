import torchvision
from torch import nn


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        # 使用自带的resnet18，pretrained=True是加载原权重参数，迁移学习
        self.model = torchvision.models.resnet18(pretrained=True)
        # 添加1层全连接，分30类
        self.f9 = nn.Linear(1000, 30)

    # 这个 forward 函数名是固定不可更改的
    def forward(self, x):
        x = self.model(x)
        x = self.f9(x)
        return x
