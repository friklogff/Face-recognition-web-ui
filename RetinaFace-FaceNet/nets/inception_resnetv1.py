import torch
from torch import nn


# 这段代码定义了InceptionResnetV1模型,我们可以分析如下:
# 1. 最开始的几个卷积层用来进行特征提取,并逐渐增加特征通道数。
# 2. 然后是模型的主体部分,包含重复的InceptionResnetV1模块(Block35、Block17、Block8等)。这些模块使用不同尺寸的卷积核和并联的分支来提取特征,并通过加权和连接,这是Inception模型的典型设计。
# 3. Block35模块包含3个并联分支,分支使用了不同尺寸的卷积核(1x1、3x3、3x3两次)来提取特征。
# 4. Block17模块包含2个并联分支,分支使用了不同形状的卷积核((1x1+1x7+7x1)、(1x1+3x3))来提取特征。
# 5. Block8模块也包含2个并联分支,使用不同形状的卷积核(1x1+1x3+3x1)来提取特征。
# 6. Mixed_6a和Mixed_7a模块使用4个并联分支(不同尺寸的卷积和最大池化),相当于在多个scale上聚合特征。
# 7. 最后通过AdaptiveAvgPool2d将特征映射均值池化到1x1,用于分类或特征向量的提取。
# 8. 该模型使用了Inception模块的并联设计和恒等映射连接,可以有效捕捉图像的空间信息,在图像分类和特征提取上表现很好。
# 定义BasicConv2d类,实现基本的卷积、BN和ReLU操作


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,  # 输入和输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 卷积步长
            padding=padding,  # 填充
            bias=False  # 不使用偏执
        )
        self.bn = nn.BatchNorm2d(  # 批归一化
            out_planes,
            eps=0.001,  # Batch Norm 防止除零
            momentum=0.1,  # Batch Norm 动量
            affine=True  # 仿射变换,如果为False仅进行拉伸
        )
        self.relu = nn.ReLU(inplace=False)  # 激活函数,in-place 避免覆盖输入

    def forward(self, x):
        x = self.conv(x)  # 卷积
        x = self.bn(x)  # BN归一化
        x = self.relu(x)  # ReLU激活
        return x


# Block35,实现35层残差块
class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale  # 缩放因子

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 1x1 和 3x3 卷积
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 1x1 和 两次3x3 卷积
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)  # 融合分支,1x1 卷积
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)  # 分支0
        x1 = self.branch1(x)  # 分支1
        x2 = self.branch2(x)  # 分支2
        out = torch.cat((x0, x1, x2), 1)  # cat融合不同分支
        out = self.conv2d(out)
        out = out * self.scale + x  # 残差连接
        out = self.relu(out)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),  # 1x1 和 1x7 卷积
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))  # 7x1 卷积
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # 1x3 卷积
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))  # 3x1 卷积
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    def __init__(self):
        super(InceptionResnetV1, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        return x
