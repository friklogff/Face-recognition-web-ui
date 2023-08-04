# -*- coding = utf-8 -*-
"""
# @Time : 2023/8/2 19:07
# @Author : FriK_log_ff 374591069
# @File : eval_en.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from nets.facenet import Facenet
from utils.dataloader import LFWDataset
from utils.utils_metrics import evaluate
from car import roc


def test(test_loader, model, cuda, log_interval, batch_size):
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))

        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    return tpr, fpr, accuracy, val, val_std, far, best_thresholds, distances, labels


def evatest(model_path, backbone):
    cuda = True
    # backbone = "mobilenet"
    input_shape = [160, 160, 3]
    # model_path = "model_data/facenet_mobilenet.pth"
    lfw_dir_path = "lfw"
    lfw_pairs_path = "model_data/lfw_pair.txt"
    batch_size = 256
    log_interval = 1

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size,
        shuffle=False)

    model = Facenet(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    # 多轮测试
    num_tests = 1
    tpr_list, fpr_list, accuracy_list, val_list, val_std_list, far_list, best_thresholds_list = [], [], [], [], [], [], []
    distances_list, labels_list = [], []

    for i in range(num_tests):
        print('Test number:', i + 1)
        tpr, fpr, accuracy, val, val_std, far, best_thresholds, distances, labels = test(test_loader, model, cuda,
                                                                                         log_interval, batch_size)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)
        val_list.append(val)
        val_std_list.append(val_std)
        far_list.append(far)
        best_thresholds_list.append(best_thresholds)
        distances_list.append(distances)
        labels_list.append(labels)

    # 保存多轮测试的性能指标
    np.savez("test_results.npz",
             tpr=tpr_list, fpr=fpr_list,
             accuracy=accuracy_list, val=val_list, val_std=val_std_list,
             far=far_list, best_thresholds=best_thresholds_list,
             distances=distances_list, labels=labels_list)
    roc()
    return "model_data/roc_test.png"

if __name__ == "__main__":
    evatest("model_data/facenet_mobilenet.pth", "mobilenet")
# roc()