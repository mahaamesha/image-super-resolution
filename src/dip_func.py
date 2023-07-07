import os
import cv2 as cv
import numpy as np


def add_salt_and_pepper_noise(im, salt_prob:float=0.01, pepper_prob:float=0.01):
    im_noisy = np.copy(im)
    h, w = im_noisy.shape[:2]
    # generate random noise mask
    salt_mask = np.random.random(size=(h, w)) < salt_prob
    pepper_mask = np.random.random(size=(h, w)) < pepper_prob
    im_noisy[salt_mask] = 255    # salt noise
    im_noisy[pepper_mask] = 0    # pepper noise
    return im_noisy


def deblur_out_of_focus(im, ksize:int=7, sigma:int=2):
    psf = cv.getGaussianKernel(ksize, sigma)
    psf = np.outer(psf, psf)
    psf /= np.sum(psf)
    im_deblur = cv.filter2D(im, -1, psf)
    return im_deblur


def pyramid_upscaling(im, dsize:tuple=(None,None)):
    im2 = cv.pyrUp(im)      # if the dstsize set to default, it will upsize twice
    im2_resized = cv.resize(im2, (dsize[0],dsize[1]))
    return im2, im2_resized


def display_im(im, fname:str='im', isShow:bool=1, isSave:bool=0):
    if isShow: cv.imshow(fname, im)
    if isSave: cv.imwrite(os.getcwd() + './tests/im_pyramid_' + fname + '.jpg', im)