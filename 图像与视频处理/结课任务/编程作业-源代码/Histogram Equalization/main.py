import numpy as np
import cv2

def histogramEqualization(grey_image, K):
    # 0. Initialize
    histogram = np.zeros(K)
    height = grey_image.shape[0]
    width = grey_image.shape[1]
   
    # 1. Histogram
    for i in range(height):
        for j in range(width):
            histogram[grey_image[i][j]] += 1
    
    # 2. Normalized Histogram
    histogram_normalized = histogram/(height*width)

    # 3. Cumulative Histogram
    histogram_cumulative = np.zeros(K)
    histogram_cumulative[0] = histogram_normalized[0]
    for i in range(1,K):
        histogram_cumulative[i] = histogram_cumulative[i-1] + histogram_normalized[i]

    # 4. FSHS
    map = (K-1)*(histogram_cumulative-histogram_cumulative[0]) / (1-histogram_cumulative[0])
    map = (map+0.5).astype(np.int32)

    # 5. New Image
    new_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            new_image[i][j] = map[grey_image[i][j]]

    return new_image.astype(np.uint8)
    


if __name__ == '__main__':
    # 1. MiniTest
    matrix = [[0,2,2,2],
              [0,3,3,2],
              [1,1,4,6],
              [1,1,5,5]]
    grey_image = np.array(matrix)
    new_image = histogramEqualization(grey_image, 8)
    print(new_image)

    # 2. RealTest
    img_path = '.\\Histogram Equalization\\TooBright.jpg'
    image = cv2.imread(img_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = histogramEqualization(grey_image, 256)
    cv2.imshow('Initial', grey_image)
    cv2.imshow('Processed', new_image)
    cv2.waitKey(0)
    cv2.imwrite('.\\Histogram Equalization\\TooBright_hisEqualization.jpg', new_image)

