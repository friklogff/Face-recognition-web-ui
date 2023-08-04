# -*- coding = utf-8 -*-
"""
# @Time : 2023/8/1 22:07
# @Author : FriK_log_ff 374591069
# @File : ReTrainUI.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import gradio as gr
from retinaface_trainer import RetinaFaceTrainer
from enpredict import detect_image
from eneva import test


def train_retinaface(Cuda=True, training_dataset_path='./data/widerface/train/label.txt',
                     backbone="mobilenet", pretrained=False, model_path='',
                     Freeze_Train=True, num_workers=4):
    trainer = RetinaFaceTrainer(Cuda, training_dataset_path, backbone, pretrained, model_path, Freeze_Train,
                                int(num_workers))
    trainer.freeze_train()

    return "训练结束，具体情况请在控制台中查看"


def detect_image_change(image, model_path, backbone):
    image_path = image.name
    model_path = model_path.name
    temp_img_path = "output/result.jpg"
    result = detect_image(image_path, model_path, backbone, temp_img_path)
    return result


def evaluation_test(model_path, backbone):
    model_path = model_path.name
    img_path_1, img_path_2 = test(model_path, backbone)
    print(img_path_1, img_path_2)
    return img_path_1, img_path_2

if __name__ == "__main__":

    with gr.Blocks() as demo:
        # 顶部文字
        gr.Markdown("""
        # Retinaface模型
        项目运行具体情况请在控制台中查看
        ### 1. 模态一：模型训练
        点击example即可自动填充预设训练方式
        如果设置了model_path，则主干的权值无需加载，pretrained的值无意义,默认pretrained = False，Freeze_Train = True。
        如果不设置model_path，pretrained = True，此时仅加载主干开始训练。默认pretrained = True，Freeze_Train = True。
        如果不设置model_path，pretrained = False，Freeze_Train = False，此时从0开始训练，且没有冻结主干的过程。
    
        ### 2. 模态二：模型试用
        点击example即可自动填充预设模型和测试用图
        ### 3. 模态三：模型评估
        点击example即可自动填充预设模型
        """)

        with gr.Tabs():
            with gr.TabItem("模型训练"):
                # 一行 两列 左边一列是输入 右边一列是输出
                with gr.Row():
                    with gr.Column():  # 左边一列是输入
                        use_cuda = gr.Checkbox(label="Use CUDA")
                        dataset_path = gr.Textbox(label="Training Dataset Path")
                        backbone = gr.Dropdown(['mobilenet', 'resnet50'], label="Backbone")
                        use_pretrained = gr.Checkbox(label="Use Pretrained")
                        model_path = gr.Textbox(label="Model Path (if available)")
                        freeze_training = gr.Checkbox(label="Freeze Training")
                        num_workers = gr.Number(label="Number of Workers")
                        with gr.Row():
                            train_button = gr.Button("开始训练")

                    with gr.Column():
                        x_output = gr.Textbox(label="Training Log")
                        gr.Examples(
                            examples=[[True,
                                       './data/widerface/train/label.txt',
                                       "mobilenet",
                                       False,
                                       'model_data/Retinaface_mobilenet0.25.pth',
                                       True,
                                       4],
                                      [True,
                                       './data/widerface/train/label.txt',
                                       "resnet50",
                                       False,
                                       'model_data/Retinaface_resnet50.pth',
                                       True,
                                       4],
                                      [True,
                                       './data/widerface/train/label.txt',
                                       "mobilenet",
                                       True,
                                       '',
                                       True,
                                       4],
                                      [True,
                                       './data/widerface/train/label.txt',
                                       "mobilenet",
                                       False,
                                       'model_data/Retinaface_mobilenet0.25.pth',
                                       False,
                                       4],
                                      ],
                            inputs=[use_cuda, dataset_path, backbone, use_pretrained,
                                    model_path, freeze_training, num_workers])

            train_button.click(fn=train_retinaface,
                               inputs=[use_cuda, dataset_path, backbone, use_pretrained,
                                       model_path, freeze_training, num_workers],
                               outputs=x_output)
            with gr.TabItem("模型试用"):
                # 一行 两列 左边一列是输入 右边一列是输出
                with gr.Row():
                    with gr.Column():  # 左边一列是输入
                        image_input = gr.File(label="Image")
                        model_input = gr.File(label="model_path")
                        bone_input = gr.Dropdown(['mobilenet', 'resnet50'], label="Backbone")

                        # 生成、重置按钮（row：行）
                        with gr.Row():
                            image_button = gr.Button("生成")
                    with gr.Column():  # 右边一列是输出
                        # 输出框
                        image_output = gr.Image(label="Output Image")

                        # 样例框
                        gr.Examples(
                            examples=[
                                ["model_data/Retinaface_mobilenet0.25.pth", "mobilenet"],
                                ["model_data/Retinaface_resnet50.pth", 'resnet50'],
                                ["logs/Epoch1-Total_Loss7.8133.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss16.7059.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss18.3385.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss22.6031.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss28.5903.pth", "mobilenet"],
                            ],
                            inputs=[model_input, bone_input]
                        )

                        # 样例框
                        gr.Examples(
                            examples=[
                                "img/street.jpg",
                                "img/timg.jpg"
                            ],
                            inputs=[image_input]
                        )
            image_button.click(fn=detect_image_change,
                               inputs=[image_input, model_input, bone_input],
                               outputs=image_output),

            with gr.TabItem("模型评估"):
                # 一行 两列 左边一列是输入 右边一列是输出
                with gr.Row():
                    with gr.Column():  # 左边一列是输入
                        test_model_input = gr.File(label="model_path")
                        test_bone_input = gr.Dropdown(['mobilenet', 'resnet50'], label="Backbone")

                        # 生成、重置按钮（row：行）
                        with gr.Row():
                            eva_button = gr.Button("测试")
                    with gr.Column():  # 右边一列是输出
                        # 输出框
                        test_image_output_1 = gr.Image(label="Output Image")
                        test_image_output_2 = gr.Image(label="Output Image")
                        # 样例框
                        gr.Examples(
                            examples=[
                                ["model_data/Retinaface_mobilenet0.25.pth", "mobilenet"],
                                ["model_data/Retinaface_resnet50.pth", 'resnet50'],
                                ["logs/Epoch1-Total_Loss7.8133.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss16.7059.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss18.3385.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss22.6031.pth", "mobilenet"],
                                ["logs/Epoch1-Total_Loss28.5903.pth", "mobilenet"],
                            ],
                            inputs=[test_model_input, test_bone_input]
                        )
            eva_button.click(fn=evaluation_test,
                             inputs=[test_model_input, test_bone_input],
                             outputs=[test_image_output_1, test_image_output_2]),
    demo.launch()
