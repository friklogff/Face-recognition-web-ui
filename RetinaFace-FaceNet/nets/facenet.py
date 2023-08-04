import torch.nn as nn
from torch.nn import functional as F

from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1


# 1. 首先定义了两种backbone网络:mobilenet和inception_resnetv1。这两个网络用于提取人脸图像的特征向量。
# 2. Facenet类定义了整个FaceNet模型。在__init__方法中,它会选择backbone网络,并删除最后几层(avg和fc层),取而代之的是AdaptiveAvgPool2d层和几个全连接层。
# 3. forward方法先使用选择的backbone网络提取特征,然后使用AdaptiveAvgPool2d平均池化到1x1,flatten成一维,传入全连接层。最后使用BN层和L2归一化产生特征向量。
# 4. forward_feature方法返回特征向量前的BN层输出和归一化后的特征向量。前者可用于训练softmax分类器,后者用于计算特征距离。
# 5. forward_classifier方法将特征向量传入定义的分类器(如softmax)以得到预测类别。
# 6. 所以总体来说,这个FaceNet模型通过backbone网络提取人脸特征,然后使用全连接层将其映射到较低维度空间,并作归一化处理以用于验证人脸身份或训练分类器。
class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()  # 加载mobilenet模型
        del self.model.fc  # 删除最后的全连接层
        del self.model.avg  # 删除平均池化层

    def forward(self, x):
        x = self.model.stage1(x)  # 第1阶段,包含1个depthwise separable convolution
        x = self.model.stage2(x)  # 第2阶段,包含2个depthwise separable convolution
        x = self.model.stage3(x)  # 第3阶段,包含3个depthwise separable convolution
        return x


class inception_resnet(nn.Module):
    def __init__(self):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()  # 加载InceptionResnetV1模型

    def forward(self, x):
        x = self.model.conv2d_1a(x)  # 第1个卷积块
        x = self.model.conv2d_2a(x)  # 第2个卷积块
        x = self.model.conv2d_2b(x)  # 第2个卷积块
        x = self.model.maxpool_3a(x)  # 最大池化
        x = self.model.conv2d_3b(x)  # 第3个卷积块
        x = self.model.conv2d_4a(x)  # 第4个卷积块
        x = self.model.conv2d_4b(x)  # 第4个卷积块
        x = self.model.repeat_1(x)  # 第1组inception模块重复
        x = self.model.mixed_6a(x)  # 第2个inception模块
        x = self.model.repeat_2(x)  # 第2组inception模块重复
        x = self.model.mixed_7a(x)  # 第3个inception模块
        x = self.model.repeat_3(x)  # 第3组inception模块重复
        x = self.model.block8(x)  # 第4个inception模块
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":  # 选择mobilenet作为特征提取网络
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "inception_resnetv1":  # 选择inception_resnetv1作为特征提取网络
            self.backbone = inception_resnet()
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - {}, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))  # 平均池化层
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)  # dropout层
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        # 全连接层,将flatten后的向量映射到embedding_size维
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)  # BN层
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)
            # 可选的全连接分类器

    def forward(self, x):
        x = self.backbone(x)  # 使用选择的backbone网络提取特征
        x = self.avg(x)  # 平均池化
        x = x.view(x.size(0), -1)  # flatten
        x = self.Dropout(x)  # dropout
        x = self.Bottleneck(x)  # 全连接映射
        x = self.last_bn(x)  # BN
        x = F.normalize(x, p=2, dim=1)  # L2归一化
        return x

    def forward_feature(self, x):
        x = self.backbone(x)  # 使用选择的backbone网络提取特征
        x = self.avg(x)  # 平均池化
        x = x.view(x.size(0), -1)  # flatten
        x = self.Dropout(x)  # dropout
        before_normalize = self.Bottleneck(x)  # 全连接映射,但不过BN和归一化
        #           返回BN前的特征向量,用于训练分类器
        x = self.last_bn(before_normalize)  # BN
        x = F.normalize(x, p=2, dim=1)  # L2归一化
        return before_normalize, x  # 返回BN前后的特征向量

    def forward_classifier(self, x):
        x = self.classifier(x)  # 将BN后的特征向量传入分类器
        return x
