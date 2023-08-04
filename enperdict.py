# -*- coding = utf-8 -*-
"""
# @Time : 2023/6/30 15:20
# @Author : FriK_log_ff 374591069
# @File : enperdict.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import time

import cv2
import numpy as np

from retinaface import Retinaface
import dlib


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def _largest_face(dets):
    if len(dets) == 1:
        return 0
    face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area:
            largest_index = index
            largest_area = face_areas[index]
    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))
    return largest_index


# 计算眼睛的长宽比：eye aspect ratio (EAR)
def _eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


retinaface = Retinaface()
mode = "video"
temp_img_path = "output/result.jpg"
video_path = "input/1.mp4"
# video_path = 0
video_save_path = "output/1.mp4"
video_fps = 25.0
test_interval = 100
dir_origin_path = "img/"
dir_save_path = "img_out/"


# 对单张图片进行人脸检测
def detect_image(img, temp_img_path):
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

        # 返回临时文件路径


# 对视频流进行人脸检测
# def detect_video(video_path, video_save_path="", video_fps=25.0):
#     capture = cv2.VideoCapture(video_path)  # 0/path
#     if video_save_path != "":
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
#
#     ref, frame = capture.read()
#     if not ref:
#         raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
#
#     fps = 0.0
#     while (True):
#         t1 = time.time()
#         # 读取某一帧
#         ref, frame = capture.read()
#         if not ref:
#             break
#         # 格式转变，BGRtoRGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 进行检测
#         frame = np.array(retinaface.detect_image(frame))
#         # RGBtoBGR满足opencv显示格式
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#
#         fps = (fps + (1. / (time.time() - t1))) / 2
#         print("fps= %.2f" % (fps))
#         frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         cv2.imshow("video", frame)
#         c = cv2.waitKey(1) & 0xff
#         if video_save_path != "":
#             out.write(frame)
#
#         if c == 27:
#             capture.release()
#             break
#     print("Video Detection Done!")
#     capture.release()
#     if video_save_path != "":
#         print("Save processed video to the path :" + video_save_path)
#     out.release()
#     cv2.destroyAllWindows()


# def detect_video(video_path):
#     capture = cv2.VideoCapture(video_path)
#     while True:
#         ref, frame = capture.read()
#         if not ref:
#             break
#         # 格式转变，BGRtoRGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 进行检测
#         frame = np.array(retinaface.detect_image(frame))
#         yield frame

# class VideoDetector:
#     def __init__(self, video_path, video_save_path="", video_fps=25.0):
#         self.capture = cv2.VideoCapture(video_path)
#         self.video_save_path = video_save_path
#         if video_save_path != "":
#             fourcc = cv2.VideoWriter_fourcc(*'XVID')
#             size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#             self.out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
#         ref, frame = self.capture.read()
#         if not ref:
#             raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
#         self.fps = 0.0
#
#     def process_frame(self):
#         t1 = time.time()
#         # 读取某一帧
#         ref, frame = self.capture.read()
#         if not ref:
#             return None
#         # 格式转变，BGRtoRGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 进行检测
#         frame = np.array(retinaface.detect_image(frame))
#         # RGBtoBGR满足opencv显示格式
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         self.fps = (self.fps + (1. / (time.time() - t1))) / 2
#         print("fps= %.2f" % (self.fps))
#         frame = cv2.putText(frame, "fps= %.2f" % (self.fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         if self.video_save_path != "":
#             self.out.write(frame)
#         return frame
#
#     def release(self):
#         print("Video Detection Done!")
#         self.capture.release()
#         if self.video_save_path != "":
#             print("Save processed video to the path :" + self.video_save_path)
#             self.out.release()
class LiveVideoDetector:
    def __init__(self, video_path, video_save_path="", video_fps=25.0, use_camera=False):
        if use_camera:
            self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.VideoCapture(video_path)
        self.video_save_path = video_save_path
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        ref, frame = self.capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        self.fps = 0.0
        self.frame_counter = 0
        self.blink_counter = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.flag = 0
        self.fname = None

    def process_frame(self):

        t1 = time.time()
        ref, frame = self.capture.read()
        if not ref:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        # 集成眨眼检测
        if len(faces) != 0:
            largest_index = _largest_face(faces)
            face_rectangle = faces[largest_index]
            landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, face_rectangle).parts()])
            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            EAR_left = _eye_aspect_ratio(left_eye)
            EAR_right = _eye_aspect_ratio(right_eye)
            ear = (EAR_left + EAR_right) / 2.0
            if ear < 0.21:
                self.frame_counter += 1
                status = "Blinking"
            else:
                if self.frame_counter >= 3:
                    self.blink_counter += 1
                self.frame_counter = 0
                status = "Open"
            cv2.putText(frame, "Blinks: {}".format(self.blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Status: {}".format(status), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if self.blink_counter >= 2:  # If blinks are more than the threshold, perform face recognition
                self.flag = 1
                cv2.putText(frame, "Liveness: Yes", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.flag = 0
                cv2.putText(frame, "Liveness: No", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 进行检测
        old_image, self.fname = retinaface.live_detect_image(frame, self.flag)
        frame = np.array(old_image)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 计算fps
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (self.fps))
        frame = cv2.putText(frame, "fps= %.2f" % (self.fps), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.video_save_path != "":
            self.out.write(frame)
        return frame

    def release(self):
        print("Video Detection Done!")
        self.capture.release()
        if self.video_save_path != "":
            print("Save processed video to the path:" + self.video_save_path)
            self.out.release()

    def get_blink_counter(self):
        return self.blink_counter

    def get_fname(self):
        return self.fname



class VideoDetector:
    def __init__(self, video_path, video_save_path="", video_fps=25.0, use_camera=False):
        if use_camera:
            self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.VideoCapture(video_path)
        self.video_save_path = video_save_path
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        ref, frame = self.capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        self.fps = 0.0

    def process_frame(self):
        t1 = time.time()
        # 读取某一帧
        ref, frame = self.capture.read()
        if not ref:
            return None
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 进行检测
        frame = np.array(retinaface.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.fps = (self.fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (self.fps))
        frame = cv2.putText(frame, "fps= %.2f" % (self.fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.video_save_path != "":
            self.out.write(frame)

        return frame

    def release(self):
        print("Video Detection Done!")
        self.capture.release()
        if self.video_save_path != "":
            print("Save processed video to the path :" + self.video_save_path)
            self.out.release()



# 对视频流进行人脸检测
# def detect_video(video_path, video_save_path="", video_fps=25.0):
#     capture = cv2.VideoCapture(video_path)  # 0/path
#     if video_save_path != "":
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
#
#     ref, frame = capture.read()
#     if not ref:
#         raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
#
#     fps = 0.0
#     while (True):
#         t1 = time.time()
#         # 读取某一帧
#         ref, frame = capture.read()
#         if not ref:
#             break
#         # 格式转变，BGRtoRGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 进行检测
#         frame = np.array(retinaface.detect_image(frame))
#         # RGBtoBGR满足opencv显示格式
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#
#         fps = (fps + (1. / (time.time() - t1))) / 2
#         print("fps= %.2f" % (fps))
#         frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         cv2.imshow("video", frame)
#         c = cv2.waitKey(1) & 0xff
#         if video_save_path != "":
#             out.write(frame)
#
#         if c == 27:
#             capture.release()
#             break
#     print("Video Detection Done!")
#     capture.release()
#     if video_save_path != "":
#         print("Save processed video to the path :" + video_save_path)
#     out.release()
#     cv2.destroyAllWindows()

# 测试指定图片的FPS
def get_FPS(img_path, test_interval=100):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tact_time = retinaface.get_FPS(img, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')


# 对图片文件夹进行批量检测
def detect_dir(dir_origin_path, dir_save_path):
    import os

    from tqdm import tqdm
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            r_image = retinaface.detect_image(image)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)


if __name__ == "__main__":
    # mode = "predict"
    # temp_img_path = "output/result.jpg"
    # video_path = 0
    # video_save_path = ""
    # video_fps = 25.0
    # test_interval = 100
    # dir_origin_path = "img/"
    # dir_save_path = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            detect_image(img)
        # if temp_img_path != "":
        #     os.remove(temp_img_path)


    elif mode == "video":
        # detect_video(video_path, video_save_path, video_fps)
        detector = VideoDetector(video_path, video_save_path, video_fps)
        while True:
            frame = detector.process_frame()
            if frame is None:
                break
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        detector.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img_path = 'img/obama.jpg'
        get_FPS(img_path, test_interval)

    elif mode == "dir_predict":
        detect_dir(dir_origin_path, dir_save_path)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
