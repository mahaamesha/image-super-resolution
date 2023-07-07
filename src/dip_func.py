import os
import cv2 as cv
import numpy as np


def add_salt_and_pepper_noise(image, salt_prob:float=0.01, pepper_prob:float=0.01):
    noisy_image = np.copy(image)
    h, w = noisy_image.shape[:2]
    # generate random noise mask
    salt_mask = np.random.random(size=(h, w)) < salt_prob
    pepper_mask = np.random.random(size=(h, w)) < pepper_prob
    noisy_image[salt_mask] = 255    # salt noise
    noisy_image[pepper_mask] = 0    # pepper noise
    return noisy_image


def pyramid_upscaling(im, dsize:tuple=(None,None)):
    im2 = cv.pyrUp(im)      # if the dstsize set to default, it will upsize twice
    im2_resized = cv.resize(im2, (dsize[0],dsize[1]))
    return im2, im2_resized


def display_im(im, fname:str='im', isShow:bool=1, isSave:bool=0):
    if isShow: cv.imshow(fname, im)
    if isSave: cv.imwrite(os.getcwd() + './tests/im_pyramid_' + fname + '.jpg', im)