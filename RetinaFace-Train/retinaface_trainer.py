import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.retinaface import RetinaFace
from nets.retinaface_training import MultiBoxLoss, weights_init
from utils.anchors import Anchors
from utils.callbacks import LossHistory
from utils.config import cfg_mnet, cfg_re50
from utils.dataloader import DataGenerator, detection_collate
from utils.utils_fit import fit_one_epoch


class RetinaFaceTrainer:
    def __init__(self, Cuda=True, training_dataset_path='./data/widerface/train/label.txt',
                 backbone="mobilenet", pretrained=True, model_path='',
                 Freeze_Train=True, num_workers=4):
        self.Cuda = Cuda  # 是否使用GPU
        self.training_dataset_path = training_dataset_path  # 人脸标注文件的路径
        self.backbone = backbone  # 选择mobilenet或resnet50为特征提取网络
        self.pretrained = pretrained  # 是否使用预训练权重
        self.model_path = model_path  # 模型权重地址
        self.Freeze_Train = Freeze_Train  # 是否进行冻结训练
        self.num_workers = num_workers  # 使用4个workers线程从DataLoader中读取数据

    def load_model(self):
        if self.backbone == "mobilenet":
            cfg = cfg_mnet
        elif self.backbone == "resnet50":
            cfg = cfg_re50
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(self.backbone))

        model = RetinaFace(cfg=cfg, pretrained=self.pretrained)
        if not self.pretrained:
            weights_init(model)
        if self.model_path != '':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(self.model_path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        model_train = model.train()
        if self.Cuda:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

        anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
        if self.Cuda:
            anchors = anchors.cuda()

        criterion = MultiBoxLoss(2, 0.35, 7, cfg['variance'], self.Cuda)
        loss_history = LossHistory("logs/")

        return model,model_train, anchors, criterion, loss_history

    def freeze_train(self):
        model,model_train, anchors, criterion, loss_history = self.load_model()

        lr = 1e-3
        # 学习率大收敛快
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 2

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DataGenerator(self.training_dataset_path, cfg_mnet['train_image_size'])
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=self.num_workers,
                         pin_memory=True, drop_last=True, collate_fn=detection_collate)
        epoch_step = train_dataset.get_len() // Batch_size

        if self.Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen,
                          Freeze_Epoch, anchors, cfg_mnet, self.Cuda)
            lr_scheduler.step()

        lr = 1e-4
        # 学习率小防止震荡
        Batch_size = 4
        Freeze_Epoch = 2
        Unfreeze_Epoch = 4

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DataGenerator(self.training_dataset_path, cfg_mnet['train_image_size'])
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=self.num_workers,
                         pin_memory=True, drop_last=True, collate_fn=detection_collate)
        epoch_step = train_dataset.get_len() // Batch_size

        if self.Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen,
                          Unfreeze_Epoch, anchors, cfg_mnet, self.Cuda)
            lr_scheduler.step()


# if __name__ == "__main__":
#     trainer = RetinaFaceTrainer(Cuda=True, training_dataset_path='./data/widerface/train/label.txt',
#                  backbone="mobilenet", pretrained=True, model_path='',
#                  Freeze_Train=True, num_workers=4)
#     trainer.freeze_train()
