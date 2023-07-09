# I want to compare image in frequency domain which is acquired from different approaches.
# Use numpy and opencv
# the im_fft_amplitude n amplitude_cv is similar.
# the im_fft_magnitude has larger value.

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    im = cv.imread(os.getcwd() + './imgs/building.png', 0)
    

    im_fft = np.fft.fftshift( np.fft.fft2(im) )     # get the fft 2D, shift the zero into center
    im_fft_amplitude = np.log10(1 + np.abs(im_fft))
    im_fft_phase = np.angle(im_fft)

    im_fft_magnitude = 20 * np.log10(np.abs(im_fft))


    dft_cv = cv.dft(np.float32(im), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_cv_shifted = np.fft.fftshift(dft_cv)
    magnitude_cv = cv.magnitude(dft_cv_shifted[:, :, 0], dft_cv_shifted[:, :, 1])
    amplitude_cv = np.log10(1 + magnitude_cv)
    
    plt.figure(1); plt.title('FFT - amplitude spectrum (log scale)')
    plt.imshow(im_fft_amplitude, cmap='gray')
    plt.figure(2); plt.title('FFT - phase spectrum')
    plt.imshow(im_fft_phase, cmap='gray')
    plt.figure(3); plt.title('FFT - magnitude')
    plt.imshow(im_fft_magnitude, cmap='gray')
    plt.figure(4); plt.title('FFT - CV')
    plt.imshow(amplitude_cv, cmap='gray')
    plt.figure(5); plt.title('FFT NP - FFT CV')
    plt.imshow(abs(im_fft_amplitude-amplitude_cv), cmap='gray')

    plt.show()