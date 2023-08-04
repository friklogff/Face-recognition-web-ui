import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import letterbox_image, preprocess_input
from utils.utils_bbox import decode, decode_landm, non_max_suppression, retinaface_correct_boxes


class Retinaface(object):
    _defaults = {
        "model_path": 'model_data/Retinaface_mobilenet0.25.pth',
        "backbone": 'mobilenet',
        "confidence": 0.5,
        "nms_iou": 0.45,
        "input_shape": [1280, 1280, 3],
        "letterbox_image": True,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return f"Unrecognized attribute name '{n}'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()

        self.generate()

    def generate(self):
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        old_image = image.copy()
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return old_image

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            for b in boxes_conf_landms:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(old_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            return old_image

    def get_FPS(self, image, test_interval):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                loc, conf, landms = self.net(image)
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                conf = conf.data.squeeze(0)[:, 1:2]
                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval

        return tact_time

    def get_map_txt(self, image):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return np.array([])

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms,
                                                             np.array([self.input_shape[0], self.input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            return boxes_conf_landms