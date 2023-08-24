import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# from ./tests/blurring.py
def create_gaussian_filter(ksize:int, sigma:int):
    """ replace matlab function fspecial('gaussian', ksize, sigma)\\

        ksize   : int   kernel size\\
        sigma   : int   level of gaussian blurring\\
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    e = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return e / e.sum()


def add_salt_and_pepper_noise(im, salt_prob:float=0.01, pepper_prob:float=0.01):
    im_noisy = np.copy(im)
    h, w = im_noisy.shape[:2]
    # generate random noise mask
    salt_mask = np.random.random(size=(h, w)) < salt_prob
    pepper_mask = np.random.random(size=(h, w)) < pepper_prob
    im_noisy[salt_mask] = 255    # salt noise
    im_noisy[pepper_mask] = 0    # pepper noise
    return im_noisy


# from ./tests/compare_fft.py
def get_fft_image(im):
    """ transform image into freq domain using FFT\n

        im  : image in spatial domain\n
    """
    im_fft = np.fft.fftshift( np.fft.fft2(im) )     # get the fft 2D, shift the zero into center
    im_fft_amplitude = np.log10(1 + np.abs(im_fft))
    im_fft_phase = np.angle(im_fft)
    return im_fft, im_fft_amplitude, im_fft_phase


def add_padding_kernel(kernel, size_like):
    """ multiplication in freq domain conducted in same size\n

        kernel  : the kernel\n
        size_like   : image whose desired size. Recomended to be the same w source image\n
    """
    delta_rows = size_like.shape[0] - kernel.shape[0]
    delta_cols = size_like.shape[1] - kernel.shape[1]
    # Pad Array A with zeros to match the size of Array B
    pad_top = delta_rows // 2
    pad_bottom = delta_rows - pad_top
    pad_left = delta_cols // 2
    pad_right = delta_cols - pad_left
    # Pad Array A from the center to match the size of Array B
    paded_kernel = np.pad(kernel, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return paded_kernel


def get_inverse_filter(psf, K:float):
    """ get the inverse filter (Wiener Transfer Filter)\n
        Ideal   : G(u,v) = 1 / H(u,v)\n
        WTF     : G(u,v) = H*(u,v) / (|H(u,v)|^2 + K), where K = (noise_power/signal_power)^2\n

        psf : filter h(x,y) in spatial domain\n
    """
    # get OTF (optical transfer function): FFT of the PSF. In book it is H(u,v)
    otf = np.fft.fftshift( np.fft.fft2(psf) )       # get the fft 2D, shift the zero into center
    wtf = np.conj(otf) / (np.abs(otf)**2 + K)       # wiener transfer filter
    return wtf      # the ideal inverse filter


def deblurring_image_fdomain(Yuv, Guv):
    """ Deblur an image in the frequency domain.\n
        
        Yuv : Blurred image in the frequency domain.\n
        Guv : Inverse filter G(u,v). The ideal inverse filter G(u,v) = 1 / H(u,v).\n
        
        Returns:\n
        Xxy : Deblurred image in the spatial domain.\n
    """    
    Xuv = np.multiply(Yuv, Guv)
    Xxy = np.fft.ifft2(np.fft.ifftshift(Xuv))
    Xxy = np.abs(Xxy).astype(np.uint8)  # normalize in 0-255 scale

    return Xxy


def image_mirroring_quadrant(im):
    """ the result is mirrored, I<>III, II<>IV \n"""
    h,w = im.shape[:2]
    im2 = np.zeros_like(im)
    im2[:h//2, :w//2] = im[h//2:, w//2:]    # II in im2 = IV in im
    im2[:h//2, w//2:] = im[h//2:, :w//2]    # I
    im2[h//2:, :w//2] = im[:h//2, w//2:]    # III
    im2[h//2:, w//2:] = im[:h//2, :w//2]    # IV
    return im2


def pyramid_upscaling(im, dsize:tuple=(None,None)):
    im2 = cv.pyrUp(im)      # if the dstsize set to default, it will upsize twice
    im2_resized = cv.resize(im2, (dsize[0],dsize[1]))
    return im2, im2_resized
   

def display_n_save(fname:str, im):
    fig, ax = plt.subplots()
    plt.imshow(im, cmap='gray')
    fig.savefig(os.getcwd() + f'./imgs/process/{fname}.jpg')
    
    im_normalized = cv.normalize(im, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite(os.getcwd() + f'./imgs/process/_{fname}.jpg', im_normalized)