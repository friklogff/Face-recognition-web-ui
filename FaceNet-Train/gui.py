import gradio as gr
from simpleentra import train
from eval_en import evatest

def train_face_net(Cuda=True,
                   annotation_path="cls_train.txt",
                   input_shape=[160, 160, 3],
                   backbone="mobilenet",
                   pretrained=False,
                   model_path="model_data/facenet_mobilenet.pth",
                   Freeze_Train=True,
                   num_workers=4,
                   lfw_eval_flag=False,
                   lfw_dir_path="lfw",
          lfw_pairs_path="model_data/lfw_pair.txt"):
    train(
        Cuda=Cuda,
        annotation_path=annotation_path,
        input_shape=eval(input_shape),
        backbone=backbone,
        pretrained=pretrained,
        model_path=model_path,
        Freeze_Train=Freeze_Train,
        num_workers=int(num_workers),
        lfw_eval_flag=lfw_eval_flag,
        lfw_dir_path=lfw_dir_path,
        lfw_pairs_path=lfw_pairs_path)
    return "训练结束，具体情况请在控制台中查看"


from enpre import detect_image
#
#
def detect_image_change(image_1, image_2, model_path, backbone):
    image_path1 = image_1.name
    image_path2 = image_2.name
    model_path = model_path.name
    result = detect_image(image_path1, image_path2, model_path, backbone)
    return result


def eval_test(model_path, backbone):
    model_path = model_path.name
    result = evatest(model_path, backbone)
    return result


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("""
        # FaceNet模型
        项目运行具体情况请在控制台中查看
        ## 1. 模态一：模型训练
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
            with gr.TabItem("训练配置"):
                with gr.Row():
                    with gr.Column():  # 左边一列是输
                        cuda = gr.Checkbox(label="使用GPU")
                        annotation_path = gr.Textbox(lines=2, label="训练集标注文件")
                        input_shape = gr.Textbox(label="输入图像尺寸")
                        backbone = gr.Dropdown(["mobilenet", "resnet50"], label="网络backbone")
                        pretrained = gr.Checkbox(label="使用预训练权重")
                        model_path = gr.Textbox(label="模型保存路径")
                        freeze_train = gr.Checkbox(label="冻结前置网络训练")
                        num_workers = gr.Number(label="数据加载线程数")
                        with gr.Row():
                            lfw_eval = gr.Checkbox(label="LFW评估")
                            lfw_dir = gr.Textbox(label="LFW数据集目录")
                            lfw_pairs = gr.Textbox(label="LFW数据对文件")
                        train_button = gr.Button("开始训练")
                    with gr.Column():  # 右边一列是输出
                        log_box = gr.Textbox(label="日志输出")
                        gr.Examples(
                            examples=[
                                [True,
                                 "cls_train.txt",
                                 "[160, 160, 3]",
                                 "mobilenet",
                                 False,
                                 "model_data/facenet_mobilenet.pth",
                                 True,
                                 4,
                                 False,
                                 "lfw",
                                 "model_data/lfw_pair.txt"],
                                [True,
                                 "cls_train.txt",
                                 "[160, 160, 3]",
                                 "inception_resnetv1",
                                 False,
                                 "model_data/facenet_inception_resnetv1.pth",
                                 True,
                                 4,
                                 False,
                                 "lfw",
                                 "model_data/lfw_pair.txt"],
                                [True,
                                 "cls_train.txt",
                                 "[160, 160, 3]",
                                 "mobilenet",
                                 True,
                                 "",
                                 True,
                                 4,
                                 False,
                                 "lfw",
                                 "model_data/lfw_pair.txt"],
                                [True,
                                 "cls_train.txt",
                                 "[160, 160, 3]",
                                 "mobilenet",
                                 False,
                                 "",
                                 False,
                                 4,
                                 False,
                                 "lfw",
                                 "model_data/lfw_pair.txt"]
                            ],
                            inputs=[cuda, annotation_path, input_shape, backbone, pretrained, model_path, freeze_train,
                                    num_workers, lfw_eval, lfw_dir, lfw_pairs]
                        )
            train_button.click(fn=train_face_net,
                               inputs=[cuda, annotation_path, input_shape, backbone, pretrained, model_path,
                                       freeze_train,
                                       num_workers, lfw_eval, lfw_dir, lfw_pairs],
                               outputs=log_box)
            with gr.TabItem("模型试用"):
                # 一行 两列 左边一列是输入 右边一列是输出
                with gr.Row():
                    with gr.Column():  # 左边一列是输入
                        image_input1 = gr.File(label="Image")
                        image_input2 = gr.File(label="Image")
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
                                ["model_data/facenet_mobilenet.pth", "mobilenet"],
                                ['model_data/facenet_inception_resnetv1.pth', 'inception_resnetv1'],
                                ['logs/Epoch1-Total_Loss0.5115.pth-Val_Loss1.3756.pth', 'mobilenet'],
                                ['logs/Epoch1-Total_Loss7.6118.pth-Val_Loss7.3420.pth', 'mobilenet']
                            ],
                            inputs=[model_input, bone_input]
                        )

                        # 样例框
                        gr.Examples(
                            examples=[
                                ["img/1_001.jpg", "img/1_002.jpg"],
                                ["img/1_002.jpg", 'img/2_001.jpg'],
                                ["img/1_001.jpg", 'img/2_001.jpg']
                            ],
                            inputs=[image_input1, image_input2]
                        )
            image_button.click(fn=detect_image_change,
                               inputs=[image_input1, image_input2, model_input, bone_input],
                               outputs=image_output),
            with gr.TabItem("模型评估"):
                # 一行 两列 左边一列是输入 右边一列是输出
                with gr.Row():
                    with gr.Column():  # 左边一列是输入
                        eva_model_input = gr.File(label="model_path")
                        eva_bone_input = gr.Dropdown(['mobilenet', 'resnet50'], label="Backbone")
                        # 生成、重置按钮（row：行）
                        with gr.Row():
                            image_button = gr.Button("生成")
                    with gr.Column():  # 右边一列是输出
                        # 输出框
                        eva_image_output = gr.Image(label="Output Image")

                        # 样例框
                        gr.Examples(
                            examples=[
                                ["model_data/facenet_mobilenet.pth", "mobilenet"],
                                ['model_data/facenet_inception_resnetv1.pth', 'inception_resnetv1'],
                                ['logs/Epoch1-Total_Loss0.5115.pth-Val_Loss1.3756.pth', 'mobilenet'],
                                ['logs/Epoch1-Total_Loss7.6118.pth-Val_Loss7.3420.pth', 'mobilenet']
                            ],
                            inputs=[eva_model_input, eva_bone_input]
                        )
            image_button.click(fn=eval_test,
                               inputs=[eva_model_input, eva_bone_input],
                               outputs=eva_image_output),

    demo.launch()
