# -*- coding = utf-8 -*-
"""
# @Time : 2023/7/3 9:18
# @Author : FriK_log_ff 374591069
# @File : newui.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import os
from retinaface import Retinaface
import gradio as gr
import cv2
from enperdict import VideoDetector, detect_image, LiveVideoDetector


def detect_upload(video_path, video_save_path='output/result.avi', video_fps=25.0):
    video_path = video_path.name
    # 上传视频文件并进行人脸识别
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
    print(f"Returning video path: {video_save_path}")
    return video_save_path


def detect_realtime(video_path=None, video_save_path='output/result.mp4', video_fps=25.0):
    # 开启摄像头实时进行人脸识别
    video_path = 0
    detector = VideoDetector(video_path, "", video_fps)
    while True:
        frame = detector.process_frame()
        if frame is None:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    detector.release()
    cv2.destroyAllWindows()


def live_detect_realtime(video_path=None, video_save_path='output/result.mp4', video_fps=25.0):
    # 开启摄像头实时进行人脸识别
    video_path = 0
    detector = LiveVideoDetector(video_path, "", video_fps)
    while True:
        flag = detector.get_blink_counter()
        frame = detector.process_frame()
        if frame is None:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if flag == 2:
            cv2.imwrite("last_frame.png", frame)
            break
    detector.release()
    cv2.destroyAllWindows()
    return "last_frame.png"


def detect_image_change(image=None):
    image_path = image.name
    temp_img_path = "output/result.jpg"
    result = detect_image(image_path, temp_img_path)
    return result


num = 0


def encode_faces():
    '''
    在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
    '''
    retinaface = Retinaface(1)

    list_dir = os.listdir("face_dataset")
    image_paths = []
    names = []
    for name in list_dir:
        image_paths.append("face_dataset/" + name)
        names.append(name.split("_")[0])

    retinaface.encode_face_dataset(image_paths, names)
    return "Encoding complete!"


webcam = gr.Image(label="Webcam")


def capture_photo(name, img):
    """

    :param name:
    :param img:
    :return:
    """
    if name == "":
        return "Name cannot be empty!"
    if img is None:
        return "img cannot be empty"
    if webcam is gr.Image(label="Webcam"):
        return "Please click the 'Start Webcam' button first!"
    else:
        global num
        num += 1
        cv2.imwrite("face_dataset/" + name + "_" + str(num) + ".jpg", img)
        return "success to save" + name + "_" + str(num) + ".jpg"


def start_webcam():
    global webcam
    webcam = gr.Image(source="webcam", label="Webcam")
    webcam.show()

# live_detect_realtime()

with gr.Blocks() as demo:
    with gr.Tab("图片人脸识别（可测试图片在img_test）"):
        image_input = gr.File(label="Image")
        image_output = gr.Image(label="Output Image")
        image_button = gr.Button("Detect")
        image_button.click(detect_image_change, inputs=image_input, outputs=image_output)
    with gr.Tab("照相"):
        image_input = [gr.components.Textbox(label="Name(格式为name_数字.jpg，连拍数字会递增，存在face_dataset)"),
                       gr.components.Image(source="webcam", label="Webcam"),
                       ]
        image_output = gr.components.Textbox(label="output")
        image_button = gr.Button("提交")
        image_button.click(capture_photo, inputs=image_input, outputs=image_output)

    with gr.Tab("数据库更新"):
        encode_button = gr.Button("Encode")
        encode_output = gr.Textbox(label="Output")
        encode_button.click(encode_faces, outputs=encode_output)
    with gr.Tab("视频上传人脸识别（点弹出的视频框英文输入法按q可提前退出，保存在output）"):
        video_input = gr.File(label="video_path")
        # video_output = gr.Video(label="Output Video")
        video_output = gr.File(label="Output Video")
        upload_button = gr.Button("Upload")
        upload_button.click(detect_upload, inputs=video_input, outputs=video_output)
    with gr.Tab("实时人脸识别（使用前先禁用浏览器摄像头权限，避免摄像头冲突，点弹出的视频框英文输入法按q退出）"):
        realtime_button = gr.Button("Start")
        # realtime_output = gr.Video(label="Output Video")
        realtime_button.click(detect_realtime)
    with gr.Tab("实时人脸识别plus（使用前先禁用浏览器摄像头权限，避免摄像头冲突，点弹出的视频框英文输入法按q退出）"):
        realtime_button = gr.Button("Start")
        # realtime_output = gr.Video(label="Output Video")
        live_output = [
                       gr.Image(label="Output Image")
                       ]
        realtime_button.click(live_detect_realtime, outputs=live_output)

demo.launch()
