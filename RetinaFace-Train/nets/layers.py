import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#

def conv_bn(inp, oup, stride=1, leaky=0):
    """
    定义卷积块函数，包括Conv2D、BatchNormalization和LeakyReLU层。

    参数:
        inp (int): 输入通道数。
        oup (int): 输出通道数。
        stride (int): 卷积步长，默认为1。
        leaky (float): LeakyReLU的负斜率，默认为0。

    返回:
        Sequential: 包含Conv2D、BatchNormalization和LeakyReLU层的序列模型。
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    """
    定义卷积块函数，包括1x1的Conv2D、BatchNormalization和LeakyReLU层。

    参数:
        inp (int): 输入通道数。
        oup (int): 输出通道数。
        stride (int): 卷积步长。
        leaky (float): LeakyReLU的负斜率，默认为0。

    返回:
        Sequential: 包含1x1的Conv2D、BatchNormalization和LeakyReLU层的序列模型。
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),   # 3x3卷积操作，输入通道数为inp，输出通道数为oup，步长为stride，填充为1
        nn.BatchNorm2d(oup),                             # 批归一化操作，通道数为oup
        nn.LeakyReLU(negative_slope=leaky, inplace=True)  # LeakyReLU激活函数，负斜率为leaky
    )

# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization
# ---------------------------------------------------#
# 定义卷积块函数，包括1x1的Conv2D、BatchNormalization和LeakyReLU层。
def conv_bn_no_relu(inp, oup, stride):
    """
    定义卷积块函数，包括Conv2D和BatchNormalization层。

    参数:
        inp (int): 输入通道数。
        oup (int): 输出通道数。
        stride (int): 卷积步长。

    返回:
        Sequential: 包含Conv2D和BatchNormalization层的序列模型。
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # 3x3卷积操作，输入通道数为inp，输出通道数为oup，步长为stride，填充为1
        nn.BatchNorm2d(oup)  # 批归一化操作，通道数为oup
    )

# ---------------------------------------------------#
#   多尺度加强感受野
# ---------------------------------------------------#

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0 # 输出通道数必须是4的倍数
        leaky = 0
        if (out_channel <= 64):# 如果输出通道数小于等于64，则设置LeakyReLU的负斜率为0.1；否则为0
            leaky = 0.1

        # 3x3卷积
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        # 利用两个3x3卷积替代5x5卷积
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        # 利用三个3x3卷积替代7x7卷积
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, inputs):
        """
        定义SSH模块的前向传播。

        参数:
            inputs (tensor): 输入特征图张量，形状为(batch_size, in_channel, height, width)。

        返回:
            tensor: 输出特征图张量，形状为(batch_size, out_channel, height, width)。
        """
        conv3X3 = self.conv3X3(inputs)   # 进行3x3卷积操作

        conv5X5_1 = self.conv5X5_1(inputs)    # 进行第一个3x3卷积操作
        conv5X5 = self.conv5X5_2(conv5X5_1)   # 进行第二个3x3卷积操作

        conv7X7_2 = self.conv7X7_2(conv5X5_1)  # 进行第一个3x3卷积操作
        conv7X7 = self.conv7x7_3(conv7X7_2)    # 进行第二个3x3卷积操作

        # 所有结果堆叠起来
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)   # 在通道维度上拼接三个特征图
        out = F.relu(out)   # 进行ReLU激活函数操作
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0.1 if out_channels <= 64 else 0  # 如果输出通道数小于等于64，则设置LeakyReLU的负斜率为0.1；否则为0

        # 输出特征图1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)

        # 输出特征图2
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)

        # 输出特征图3
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        # 特征融合层1
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)

        # 特征融合层2
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, inputs):
        """
        定义FPN模块的前向传播。

        参数:
            inputs (dict): 输入特征图字典，包含三个不同尺度的特征图。
                           字典键为"C3", "C4"和"C5"，值为对应的特征图张量，
                           形状分别为(batch_size, in_channel, height, width)。

        返回:
            list: 输出特征图列表，包含三个不同尺度的特征图。
                  列表元素为对应的特征图张量，形状分别为(batch_size, out_channel, height, width)。
        """
        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        # -------------------------------------------#
        inputs = list(inputs.values())  # 将输入特征图字典转换为列表

        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是output1  80, 80, 64
        #         output2  40, 40, 64
        #         output3  20, 20, 64
        # -------------------------------------------#
        output1 = self.output1(inputs[0])  # 输出特征图1
        output2 = self.output2(inputs[1])  # 输出特征图2
        output3 = self.output3(inputs[2])  # 输出特征图3

        # -------------------------------------------#
        #   output3上采样和output2特征融合
        #   output2  40, 40, 64
        # -------------------------------------------#
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)],
                            mode="nearest")  # 上采样操作，将输出特征图3的尺寸调整为与输出特征图2相同
        output2 = output2 + up3  # 特征融合：输出特征图2和上采样后的输出特征图3相加
        output2 = self.merge2(output2)  # 进行特征融合层2的卷积操作

        # -------------------------------------------#
        #   output2上采样和output1特征融合
        #   output1  80, 80, 64
        # -------------------------------------------#
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)],
                            mode="nearest")  # 上采样操作，将输出特征图2的尺寸调整为与输出特征图1相同
        output1 = output1 + up2  # 特征融合：输出特征图1和上采样后的输出特征图2相加
        output1 = self.merge1(output1)  # 进行特征融合层1的卷积操作

        out = [output1, output2, output3]  # 输出特征图列表
        return out
