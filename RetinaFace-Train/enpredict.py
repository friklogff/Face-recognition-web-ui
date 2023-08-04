# -*- coding = utf-8 -*-
"""
# @Time : 2023/8/2 0:02
# @Author : FriK_log_ff 374591069
# @File : enpredict.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import time

import cv2
import numpy as np

from enretinaface import Retinaface

mode = "predict"
video_path = 0
video_save_path = ""
video_fps = 25.0
test_interval = 100
dir_origin_path = "img/"
dir_save_path = "img_out/"


def detect_image(img, model_path, backbone, temp_img_path):
    retinaface = Retinaface(model_path=model_path, backbone=backbone)

    image = cv2.imread(img)
    if image is None:
        print('Open Error! Try again!')
        return
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r_image = retinaface.detect_image(image)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("after", r_image)
        # cv2.waitKey(0)
        if temp_img_path != "":
            # 保存到临时文件
            cv2.imwrite(temp_img_path, r_image)
            print("Save processed img to the path :" + temp_img_path)
            return temp_img_path


# detect_image("img/street.jpg",'model_data/Retinaface_resnet50.pth', 'resnet50', "output/result.jpg")