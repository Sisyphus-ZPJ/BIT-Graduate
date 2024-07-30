import os
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torchvision.transforms import *

from model.deepten import DeepTen
from datasets.dtd import DTDDataset
from metric import AverageMeter, accuracy

labels = ['knitted', 'cracked', 'porous', 'chequered', 'dotted', 'honeycombed', 
          'stratified', 'interlaced', 'crystalline', 'matted', 'veined', 
          'cobwebbed', 'braided', 'waffled', 'spiralled', 'gauzy', 'blotchy', 
          'zigzagged', 'crosshatched', 'sprinkled', 'swirly', 'frilly', 'studded', 
          'woven', 'lined', 'perforated', 'flecked', 'grid', 'fibrous', 'lacelike', 
          'bumpy', 'banded', 'scaly', 'pitted', 'stained', 'meshed', 'smeared', 'polka-dotted', 
          'grooved', 'bubbly', 'pleated', 'freckled', 'paisley', 'marbled', 'striped', 
          'potholed', 'wrinkled']

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConfusionMatrix(object):
    """
    注意版本问题,使用numpy来进行数值计算的
    """
    def __init__(self, num_classes: int, labels: list):
            self.matrix = np.zeros((num_classes, num_classes))
            self.num_classes = num_classes
            self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1
        # 行代表预测标签 列表示真实标签

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("acc is", acc)

        # precision, recall, specificity
        table = PrettyTable(["Label", "Precission", "Recall", "Specificity"])
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            pre = round(TP / (TP + FP), 3)    # round 保留三位小数
            recall = round(TP / (TP + FN), 3)
            spec = round(TN / (TN + FP), 3)
            table.add_row([self.labels[i], pre, recall, spec])
        print(table)

    def plot(self):
        matrix = self.matrix
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print(matrix)
        fig = plt.gcf()
        fig.set_size_inches(32.0, 24.0)
        plt.imshow(matrix, cmap=plt.cm.Blues)  # 颜色变化从白色到蓝色

        # 设置 x  轴坐标 label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 将原来的 x 轴的数字替换成我们想要的信息 self.num_classes  x 轴旋转45度
        # 设置 y  轴坐标 label
        plt.yticks(range(self.num_classes), self.labels)

        # 显示 color bar  可以通过颜色的密度看出数值的分布
        plt.colorbar()
        plt.xlabel("true_label")
        plt.ylabel("Predicted_label")
        plt.title("ConfusionMatrix")

        # 在图中标注数量 概率信息
        thresh = matrix.max() / 2
        # 设定阈值来设定数值文本的颜色 开始遍历图像的时候一般是图像的左上角
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 这里矩阵的行列交换，因为遍历的方向 第y行 第x列
                info = format(matrix[y, x], '.2f') # int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if matrix[y, x] > thresh else "black")
        # plt.tight_layout()
        # 图形显示更加的紧凑
        plt.show()
        plt.savefig('./fusionmatrix.png')


def main():
    # Load Model
    n_class = 47
    model = DeepTen(n_class, pretrained=False).cuda()
    model.eval()

    # Load Checkpoint
    checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/checkpoints/3_dtd/119.tar")['model_state_dict']
    # checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/model/deepten_resnet50_minc-1225f149.pth")
    model.load_state_dict(checkpoint, strict=False)

    # Load Data
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([# Resize(base_size),
                                         CenterCrop(352),
                                         ToTensor(),
                                         normalize])
    dataset = DTDDataset(root='/data/zhangpengjie/zhangpengjie/Workspace/Datasets', split='test', transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Loop
    confusion = ConfusionMatrix(num_classes=n_class, labels=labels)
    tbar = tqdm(dataloader, desc='\r')
    for batch_idx, (data, target) in enumerate(tbar):
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data.float())

        _, pred = output.topk(1, 1, True, True)
        confusion.update(pred.squeeze().cpu().numpy(), target.cpu().numpy())

    confusion.plot()
    # 绘制混淆矩阵
    confusion.summary()

    return


if __name__ == '__main__':
    main()
