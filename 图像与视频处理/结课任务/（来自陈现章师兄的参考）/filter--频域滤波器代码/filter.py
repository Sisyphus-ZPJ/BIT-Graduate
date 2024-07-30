import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# cupy为numpy的cuda版本，在这里使用只是为了调用GPU加快计算
import cupy as cp

# 傅里叶变换DFT
def DFT(img):
    img = cp.array(img)
    h, w = img.shape
    output = cp.zeros((h, w), dtype=cp.complex64)
    x, y = cp.mgrid[0:h, 0:w]

    # 根据公式计算每个点的频域值
    for i in range(h):
        for j in range(w):
            t = -1j * 2 * cp.pi * (i * x / h + j * y / w)
            output[i, j] = cp.sum(img * cp.exp(t))

    # 调整中心位置
    output = cp.roll(output, h // 2, axis=0)
    output = cp.roll(output, w // 2, axis=1)
    return cp.asnumpy(output)


# 傅里叶逆变换IDFT
def IDFT(img):
    img = cp.array(img)
    h, w = img.shape
    output = cp.zeros((h, w))
    x, y = cp.mgrid[0:h, 0:w]

    # 先将位置调整回来
    img = cp.roll(img, h // 2, axis=0)
    img = cp.roll(img, w // 2, axis=1)

    # 根据公式计算原图像
    for i in range(h):
        for j in range(w):
            t = 2j * cp.pi * (i * x / h + j * y / w)
            output[i, j] = abs(cp.sum(img * cp.exp(t)))
    return cp.asnumpy(1 / (h * w) * output)


# 低通滤波器
def low_filter(img, r=50):
    h, w = img.shape
    hc, wc = h // 2, w // 2
    x, y = np.ogrid[:h, :w]
    # 将距离大于半径的过滤掉
    mask = (x - hc)**2 + (y - wc)**2 <= r * r
    return img * mask


# 高通滤波器
def high_filter(img, r=50):
    h, w = img.shape
    hc, wc = h // 2, w // 2
    x, y = np.ogrid[:h, :w]
    # 将距离小于半径的过滤掉
    mask = (x - hc)**2 + (y - wc)**2 > r * r
    return img * mask

# 高斯滤波器
def gauss_filter(img, r=50):
    h, w = img.shape
    hc, wc = h // 2, w // 2
    x, y = np.ogrid[:h, :w]
    dist = (x - hc)**2 + (y - wc)**2
    mask = np.exp(-dist / (r * r) / 2)
    return img * mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 通过参数选择要处理的图片
    parser.add_argument('--img', type=str, default='testImages\lena_gray_256.png')
    parser = parser.parse_args()

    img_path = parser.img
    img = cv2.imread(img_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算DFT和IDFT
    uv = DFT(img=img)
    iuv = IDFT(uv)

    # 进行三种不同的滤波
    filter = [low_filter(uv), high_filter(uv), gauss_filter(uv)]
    # 对滤波结果进行IDFT
    ifilter = [IDFT(x) for x in filter]
    filter = [np.log(1 + abs(x)) for x in filter]

    # 将生成的9张图片排列并保存
    results = [img, np.log(1 + abs(uv)), iuv] + filter + ifilter
    # 每张图片的标签
    title = ['GRAY', 'DFT', 'IDFT', 'Low Filter', 'High Filter', 'Gauss Filter', 'Low IDFT', 'High IDFT', 'Gauss IDFT']
    plt.figure()
    for i in range(len(results)):
        ax = plt.subplot(3, 3, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(results[i], 'gray')
        plt.title(title[i])

    # 图片保存在filter.jpg中
    plt.savefig('filter.jpg')
    plt.show()