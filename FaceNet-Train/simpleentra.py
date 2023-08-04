# -*- coding = utf-8 -*-
"""
# @Time : 2023/8/2 17:24
# @Author : FriK_log_ff 374591069
# @File : simpleentra.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import LossHistory, triplet_loss, weights_init
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils_fit import fit_one_epoch


# ------------------------------------------------#
#   计算一共有多少个人，用于利用交叉熵辅助收敛
# ------------------------------------------------#
def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


def train(Cuda=True, annotation_path="cls_train.txt", input_shape=[160, 160, 3],
          backbone="mobilenet", pretrained=False, model_path="model_data/facenet_mobilenet.pth",
          Freeze_Train=True, num_workers=4, lfw_eval_flag=True, lfw_dir_path="lfw",
          lfw_pairs_path="model_data/lfw_pair.txt"):
    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss = triplet_loss()
    loss_history = LossHistory("logs")
    # ---------------------------------#
    #   LFW估计
    # ---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32,
        shuffle=False) if lfw_eval_flag else None

    # -------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    # -------------------------------------------------------#
    val_split = 0.05
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if True:
        lr = 1e-3
        Batch_size = 64
        Init_Epoch = 0
        Interval_Epoch = 2

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes)

        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen,
                          gen_val, Interval_Epoch, Cuda, LFW_loader, Batch_size, lfw_eval_flag)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 32
        Interval_Epoch = 2
        Epoch = 4

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes)

        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen,
                          gen_val, Epoch, Cuda, LFW_loader, Batch_size, lfw_eval_flag)
            lr_scheduler.step()


if __name__ == "__main__":
    train()
