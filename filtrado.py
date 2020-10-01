import cv2
import numpy as np


def us_mask(original, kernel, amount):
    blurred = cv2.GaussianBlur(original, kernel, 0)
    result = original + (original-blurred)*amount
    #result = cv2.convertScaleAbs(result)
    im1 = cv2.addWeighted(original, 1.0 + amount, blurred, -amount, 0)
    # im1 = im + 3.0*(im - im_blurred)
    return result, im1


"""
img = cv2.imread('gausiano_aditivo.jpg')
img = cv2.imread('lena_b_n.jpg')
img = cv2.imread('Ruido_sal_y_pimienta.jpg')
img = cv2.imread('comprimida.jpg')
img = cv2.imread('ruido_multiplicativo.jpg')
img = cv2.imread('gausiano_aditivo.jpg')
"""
img = cv2.imread('punto1.png')

if img is not None:
    # Punto 1:

    # Ventanas de Sobel
    sobel1 = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, 3))
    sobel2 = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, 3))
    sobel3 = np.array([1, -2, 1, 2, -4, 2, 1, -2, 1]).reshape((3, 3))
    a1 = cv2.filter2D(img, -1, sobel1)
    cv2.imwrite("Sobel_1.png", a1)
    a2 = cv2.filter2D(img, -1, sobel2)
    cv2.imwrite("Sobel_2.png", a2)
    a3 = cv2.filter2D(img, -1, sobel3)
    cv2.imwrite("Sobel_3.png", a3)

    # Ventanas de laplace
    laplace1 = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
    laplace2 = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape((3, 3))
    laplace3 = np.array([1, 1, 1, 1, -8, 1, 1, 1, 1]).reshape((3, 3))
    b1 = cv2.filter2D(img, -1, laplace1)
    cv2.imwrite("Laplace 1.png", b1)
    b2 = cv2.filter2D(img, -1, laplace2)
    cv2.imwrite("Laplace 2.png", b2)
    b3 = cv2.filter2D(img, -1, laplace3)
    cv2.imwrite("Laplace 3.png", b3)

    # Ventanas de Scharr
    scharr1 = np.array([-3, 0, 3, -10, 0, 10, -3, 0, 3]).reshape((3, 3))
    scharr2 = np.array([-3, -10, -3, 0, 0, 0, 3, 10, 3]).reshape((3, 3))
    c1 = cv2.filter2D(img, -1, scharr1)
    cv2.imwrite("Scharr_1.png", c1)
    c2 = cv2.filter2D(img, -1, scharr2)
    cv2.imwrite("Scharr_2.png", c2)

    # Punto 2a
    # kernel = np.ones((5, 5), np.float32)/25
    # dst1 = cv2.filter2D(img, -1, kernel)
    # cv2.imshow("Original", img)
    # cv2.imshow("filtrada kernel 5x5 usando filter2D", dst1)
    # cv2.waitkey(0)

    dst2 = img.copy()
    cv2.blur(img, (5, 5), dst2, (2, 2), cv2.BORDER_REFLECT)
    # cv2.imshow("Original", img)
    # cv2.imshow("filtrada kernel 15x15 unos usando blur", dst2)
    # cv2.waitkey(0)
    cv2.imwrite("Filtro_media_k5.png", dst2)

    dst4 = cv2.medianBlur(img, 5)
    cv2.imwrite("Filtro_mediana.png", dst4)

    dst3 = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("original", img)
    # cv2.imshow("filtrada kernel 7x7 usando  GaussianBlur por defecto", dst3)
    # cv2.waitkey(0)
    cv2.imwrite("Filtro_gaussian_blur_k5.png", dst3)

    blur = cv2.bilateralFilter(img, 5, 100, 80)
    # cv2.imshow("Original", img)
    # cv2.imshow("filtrada bilateral 9x9 sigmas de color y espacial en 100",
    # dst3)
    cv2.imwrite("Filtro_bilateral.png", blur)
    # cv2.waitkey(0)
    # cv2.destroyAllWindows()
    # cv2.waitkey(1)

    # Punto 2b
    res1, res2 = us_mask(dst3, (5, 5), 10)
    cv2.imwrite("unsharpen mask k5 a10.png", res1)
    cv2.imwrite("unsharpen mask weighted k5 a10.png", res2)

    p2cfilter = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape((3, 3))
    p2c = cv2.filter2D(img, -1, p2cfilter)
    cv2.imwrite("sharpen w filter.png", p2c)

else:
    print('La imagen no fue cargada correctamente')

print('Saliendo')
# cv2.destroyAllWindows()
