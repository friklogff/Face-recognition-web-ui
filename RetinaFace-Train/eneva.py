import os
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from enretinaface import Retinaface
from utils.enutils_map import evaluation
def plot_precision_recall_curve(precisions, recalls):
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')  # Save the plot to local file
    plt.close()
    # plt.show()

# Function to plot the AP bar chart
def plot_ap_bar_chart(aps):
    labels = ['Easy', 'Medium', 'Hard']
    x = np.arange(len(labels))
    plt.bar(x, aps)
    plt.xlabel('Difficulty Setting')
    plt.ylabel('AP')
    plt.title('Average Precision (AP) by Difficulty Setting')
    plt.xticks(x, labels)
    plt.savefig('ap_bar_chart.png')  # Save the plot to local file
    plt.close()
    # plt.show()

def test(model_path, backbone):
    mAP_retinaface = Retinaface(model_path=model_path, backbone=backbone, confidence=0.01, nms_iou=0.45)
    save_folder = './widerface_evaluate/widerface_txt/'
    gt_dir = "./widerface_evaluate/ground_truth/"
    imgs_folder = './data/widerface/val/images/'
    sub_folders = os.listdir(imgs_folder)

    test_dataset = []
    for sub_folder in sub_folders:
        image_names = os.listdir(os.path.join(imgs_folder, sub_folder))
        for image_name in image_names:
            test_dataset.append(os.path.join(sub_folder, image_name))

    num_images = len(test_dataset)

    # 存储精确率和召回率数据
    precisions = []
    recalls = []
    aps = []

    for img_name in tqdm.tqdm(test_dataset):
        image = cv2.imread(os.path.join(imgs_folder, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mAP_retinaface.get_map_txt(image)

        save_name = save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with open(save_name, "w") as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(results)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in results:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

    # 计算精确率和召回率数据
    precision, recall, ap = evaluation(save_folder, gt_dir)
    precisions.append(precision)
    recalls.append(recall)
    aps.append(ap)

    # 处理精确率和召回率数据
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    precisions = np.mean(precisions, axis=0)
    recalls = np.mean(recalls, axis=0)
    precisions = np.squeeze(precisions)
    recalls = np.squeeze(recalls)

    np.savetxt('precisions.txt', precisions)
    np.savetxt('recalls.txt', recalls)
    np.savetxt('aps.txt', aps)
    # Plot Precision-Recall Curve
    # Load the data from the local files
    precisions = np.loadtxt('precisions.txt')
    recalls = np.loadtxt('recalls.txt')
    aps = np.loadtxt('aps.txt')
    # Plot Precision-Recall Curve
    plot_precision_recall_curve(precisions, recalls)

    # Plot AP Bar Chart
    plot_ap_bar_chart(aps)
    return "precision_recall_curve.png","ap_bar_chart.png"


if __name__ == '__main__':
    test("model_data/Retinaface_resnet50.pth", 'resnet50')
    # plot_precision_recall_curve()
    # plot_ap_bar_chart()

# # 保存数据到本地
# save_data_to_file("precisions.csv", precisions)
# save_data_to_file("recalls.csv", recalls)
# save_data_to_file("aps.csv", aps)
#
# # 绘制精确率-召回率曲线
# plt.plot(recalls, precisions)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()
# # 绘制精确率-召回率曲线并保存到本地
# save_plot_to_file("precision_recall_curve.png", recalls, precisions, 'Recall', 'Precision',
#                   'Precision-Recall Curve')
#
# # 绘制 AP 值柱状图
# settings = ['easy', 'medium', 'hard']
# x_pos = [i for i, _ in enumerate(settings)]
# aps = [float(ap) for ap in aps]  # 将 aps 列表中的元素转换为浮点数
# plt.bar(x_pos, aps, color='green')
# plt.xlabel("Setting")
# plt.ylabel("AP")
# plt.title("Average Precision (AP) for each Setting")
# plt.xticks(x_pos, settings)
# plt.show()
#
# # 绘制精确率-召回率曲线并保存到本地
# save_plot_to_file("precision_recall_curve.png", recalls, precisions, 'Recall', 'Precision',
#                   'Precision-Recall Curve')
#
# # 绘制 AP 值柱状图并保存到本地
# settings = ['easy', 'medium', 'hard']
# x_pos = [i for i, _ in enumerate(settings)]
# save_plot_to_file("ap_barchart.png", x_pos, aps, 'Setting', 'AP', 'Average Precision (AP) for each Setting')
#
# # 打印 AP 值
# print("==================== Results ====================")
# print("Easy   Val AP: {}".format(aps[0]))
# print("Medium Val AP: {}".format(aps[1]))
# print("Hard   Val AP: {}".format(aps[2]))
# print("=================================================")
