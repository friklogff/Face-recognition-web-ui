# -*- coding = utf-8 -*-
"""
# @Time : 2023/8/2 19:05
# @Author : FriK_log_ff 374591069
# @File : car.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def plot_roc(fpr, tpr, figure_name="roc.png"):
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)


def roc():
    # 加载多轮测试的性能指标
    test_results = np.load("test_results.npz")
    tpr_list = test_results['tpr']
    fpr_list = test_results['fpr']

    # 绘制ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for i in range(len(tpr_list)):
        fpr, tpr = fpr_list[i], tpr_list[i]
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='darkorange',
             lw=2, label='Mean ROC curve (area = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("./model_data/roc_test.png", dpi=300)
    plt.close()


# if __name__ == "__main__":
#     roc()