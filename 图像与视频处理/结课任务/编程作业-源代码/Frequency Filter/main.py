import cv2
import numpy as np

# 快速离散傅里叶变换
def FFT(img):
    img = np.array(img)
    h, w = img.shape

    # 转变为两次一维的FFT
    mid = np.zeros((h, w), dtype=np.complex64)
    for i in range(h):
        mid[i, ...] = FFT_1D(img[i, ...])
    output = np.zeros((h, w), dtype=np.complex64)
    for j in range(w):
        output[..., j] = FFT_1D(mid[..., j])

    # 调整中心位置
    output = np.roll(output, h // 2, axis=0)
    output = np.roll(output, w // 2, axis=1)
    return output

# 快速离散傅里叶反变换
def IFFT(img):
    img = np.array(img)
    h, w = img.shape

    # 先将位置调整回来
    img = np.roll(img, w // 2, axis=1)
    img = np.roll(img, h // 2, axis=0)
    
    # 转变为两次一维的IFFT
    mid = np.zeros((h, w), dtype=np.complex64)
    for i in range(h):
        mid[i, ...] = IFFT_1D(img[i, ...]) / w
    output = np.zeros((h, w), dtype=np.complex64)
    for j in range(w):
        output[..., j] = IFFT_1D(mid[..., j]) / h

    return output

# 快速离散傅里叶变换（1维）
def FFT_1D(f):
    M = len(f)
    K = int(M/2)
    F = np.zeros(M, dtype=np.complex64)

    if M == 1:
        return np.array(f)
    
    u = np.array([i for i in range(K)])

    F_even = FFT_1D([f[index] for index in range(M) if index % 2 == 0])
    F_odd = FFT_1D([f[index] for index in range(M) if index % 2 == 1])
    _F_odd = F_odd * np.exp(-1j * 2 * np.pi * u / M)
    F[0:K] = F_even + _F_odd
    F[K:M] = F_even - _F_odd

    return F

# 快速离散傅里叶反变换（1维）
def IFFT_1D(f):
    M = len(f)
    K = int(M/2)
    F = np.zeros(M, dtype=np.complex64)

    if M == 1:
        return np.array(f)
    
    u = np.array([i for i in range(K)])

    F_even = IFFT_1D([f[index] for index in range(M) if index % 2 == 0])
    F_odd = IFFT_1D([f[index] for index in range(M) if index % 2 == 1])
    _F_odd = F_odd * np.exp(1j * 2 * np.pi * u / M)
    F[0:K] = F_even + _F_odd
    F[K:M] = F_even - _F_odd

    return F

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

def visual_uv(uv):
    uv_visual = np.log(1 + abs(uv))
    uv_visual = (uv_visual - np.min(uv_visual))/(np.max(uv_visual) - np.min(uv_visual)) * 255
    return uv_visual

if __name__ == '__main__':
    # Load Image
    img_path = '.\\Frequency Filter\\lena.png'
    image = cv2.imread(img_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('.\\Frequency Filter\\lena_grey.png', grey_image)


    # 计算DFT
    uv = FFT(grey_image)
    cv2.imwrite('.\\Frequency Filter\\lena_FFT.png', visual_uv(uv))

    # 计算IDFT
    iuv = IFFT(uv)
    cv2.imwrite('.\\Frequency Filter\\lena_IFFT.png', abs(iuv))

    # 低通滤波
    uv_lowFiltered = low_filter(uv)
    cv2.imwrite('.\\Frequency Filter\\lena_lowFiltered_uv.png', visual_uv(uv_lowFiltered))
    xy_lowFiltered = IFFT(uv_lowFiltered)
    cv2.imwrite('.\\Frequency Filter\\lena_lowFiltered_xy.png', abs(xy_lowFiltered))

    # 高通滤波
    uv_highFiltered = high_filter(uv)
    cv2.imwrite('.\\Frequency Filter\\lena_highFiltered_uv.png', visual_uv(uv_highFiltered))
    xy_highFiltered = IFFT(uv_highFiltered)
    cv2.imwrite('.\\Frequency Filter\\lena_highFiltered_xy.png', abs(xy_highFiltered))
