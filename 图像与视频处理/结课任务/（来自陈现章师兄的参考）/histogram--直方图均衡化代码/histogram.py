import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


# 对单个维度进行直方图均衡
def channel(img: np.ndarray):
    h, w = img.shape
    output = np.zeros_like(img)
    hist = np.zeros(256)
    hist_sum = np.zeros_like(hist)
    # 计算直方图
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    hist /= (h * w) #获取概率
    # 计算小于某一值的概率和
    for i in range(1, 256):
        hist_sum[i] = hist_sum[i - 1] + hist[i]
    hist_sum *= 255
    # 对每个点进行重新映射
    for i in range(h):
        for j in range(w):
            output[i][j] = hist_sum[img[i][j]]
    return output


# 直方图均衡，适配多通道、单通道情况
def histogram(img: np.ndarray):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    output = np.zeros_like(img)
    # 对每个通道分别进行直方图均衡化
    for i in range(img.shape[-1]):
        output[:, :, i] = channel(img=img[:, :, i])
    return output.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 通过参数选择要处理的图片
    parser.add_argument('--img', type=str, default='testImages\lena_gray_256.png')
    parser = parser.parse_args()

    img_path = parser.img
    img = cv2.imread(img_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    
    # 调用直方图均衡化函数生成结果
    output = histogram(img=img)
    results = [img, output]

    # 将结果保存
    plt.figure()
    for i in range(len(results)):
        plt.subplot(1, 2, i + 1)
        plt.imshow(results[i])

    # 均衡化后的图片存为histogram.jpg
    plt.savefig('histogram.jpg')
    plt.show()