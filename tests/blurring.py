# i want to compare about gaussian blurring using
# cv.GaussianBlur and convolution result using self created kernel
# the result is similar

import numpy as np
import cv2 as cv
import os


def create_gaussian_filter(ksize:int, sigma:int):
    """ replace matlab function fspecial('gaussian', ksize, sigma)\\

        ksize   : int   kernel size\\
        sigma   : int   level of gaussian blurring\\
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    e = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return e / e.sum()


if __name__ == '__main__':
    im = cv.imread(os.getcwd() + './imgs/granite.png', 0)
    psf = create_gaussian_filter(ksize=7, sigma=3)
    im_blur = cv.filter2D(im, ddepth=cv.CV_8U, kernel=psf)
    im_gaus = cv.GaussianBlur(im, (7,7), 3)
    im_diff = cv.absdiff(im_blur, im_gaus)
    print(psf)

    cv.imshow('original', im)
    cv.imshow('im_blur', im_blur)
    cv.imshow('im_gaus', im_gaus)
    cv.imshow('im_diff', im_diff)

    cv.waitKey(0)
    cv.destroyAllWindows()