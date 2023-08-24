import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img = cv2.imread(os.getcwd() + './imgs/building.png',0)

# fft to convert the image to freq domain 
f = np.fft.fft2(img)

# shift the center
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow,ccol = rows//2 , cols//2

# remove the low frequencies by masking with a rectangular window of size 60x60
# High Pass Filter (HPF)
# fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# shift back (we shifted the center before)
f_ishift = np.fft.ifftshift(fshift)

# inverse fft to get the image back 
img_back = np.fft.ifft2(f_ishift)

img_back = np.abs(img_back).astype(np.uint8)        # grayscale


img_dif = cv2.absdiff(img, img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(142),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(img_back, cmap='gray')
plt.title('Final Result'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(img_dif, cmap='gray')
plt.title('|Input - Final|'), plt.xticks([]), plt.yticks([])

plt.show()