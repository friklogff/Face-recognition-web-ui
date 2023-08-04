# -*- coding = utf-8 -*-
"""
# @Time : 2023/8/2 12:36
# @Author : FriK_log_ff 374591069
# @File : enpre.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import cv2
from PIL import Image

from facenet import Facenet


def detect_image(image_1, image_2, model_path, backbone):
    model = Facenet(model_path=model_path, backbone=backbone)
    image1 = Image.open(image_1)
    if image1 is None:
        return 'Open image_1 Error! Try again!'
    image2 = Image.open(image_2)
    if image2 is None:
        return 'Open image_2 Error! Try again!'
    probability,image_path = model.detect_image(image1, image2)
    return image_path


# detect_image("img/1_001.jpg","img/1_002.jpg", 'model_data/facenet_inception_resnetv1.pth', 'inception_resnetv1')
