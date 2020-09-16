import numpy as np
import cv2
from matplotlib import pyplot as plt


def grid(N, M, Im):
    for x in range(0, N, M+1):
        Im = cv2.line(Im, (x, 0), (x, N), 255)
    for y in range(0, N, M+1):
        Im = cv2.line(Im, (0, y), (N, y), 255)
    return Im


def interpolado(N, f, Im, numb):
    IN = cv2.resize(Im, (int(N*f), int(N*f)), interpolation=cv2.INTER_NEAREST)
    IL = cv2.resize(Im, (int(N*f), int(N*f)), interpolation=cv2.INTER_LINEAR)
    IC = cv2.resize(Im, (int(N*f), int(N*f)), interpolation=cv2.INTER_CUBIC)
    IA = cv2.resize(Im, (int(N*f), int(N*f)), interpolation=cv2.INTER_AREA)
    ILA = cv2.resize(Im, (int(N*f), int(N*f)),
                     interpolation=cv2.INTER_LANCZOS4)
    ILX = cv2.resize(Im, (int(N*f), int(N*f)),
                     interpolation=cv2.INTER_LINEAR_EXACT)
    cv2.imwrite("I"+numb+" INTER_NEAREST.png", IN)
    cv2.imwrite("I"+numb+" INTER_LINEAR.png", IL)
    cv2.imwrite("I"+numb+" INTER_CUBIC.png", IC)
    cv2.imwrite("I"+numb+" INTER_AREA.png", IA)
    cv2.imwrite("I"+numb+" INTER_LANCZOS4.png", ILA)
    cv2.imwrite("I"+numb+" INTER_LINEAR_EXACT.png", ILX)
    return


N_1, M_1 = 250, 1
N_2, M_2 = 1000, 1
N_3, M_3 = 250, 2
N_4, M_4 = 1000, 2

I1 = grid(N_1, M_1, np.zeros((N_1, N_1, 1), np.uint8))
I2 = grid(N_2, M_2, np.zeros((N_2, N_2, 1), np.uint8))
I3 = grid(N_3, M_3, np.zeros((N_3, N_3, 1), np.uint8))
I4 = grid(N_4, M_4, np.zeros((N_4, N_4, 1), np.uint8))
cv2.imwrite("I4.png", I4)
cv2.imwrite("I1.png", I1)
cv2.imwrite("I2.png", I2)
cv2.imwrite("I3.png", I3)

I1_copy = I1.copy()
I2_copy = I2.copy()
I3_copy = I3.copy()
I4_copy = I4.copy()
interpolado(N_1, 2, I1_copy, str(1))
interpolado(N_2, 0.5, I2_copy, str(2))
interpolado(N_3, 2, I3_copy, str(3))
interpolado(N_4, 0.5, I4_copy, str(4))


# Punto 4:
img = cv2.imread('photo4.jpeg', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO',
          'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


img = cv2.imread('photo4.jpeg', 0)
img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('photo4.jpeg', 0)
# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
