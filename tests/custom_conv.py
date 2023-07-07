# i want to do convolution with custom kernel

import numpy as np
import cv2 as cv


def convolution(image, kernel):
    # Get dimensions of the input image and kernel
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding sizes for "same" convolution
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2
    
    # Create a padded version of the input image for each channel
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    
    # Create an empty output image
    output_image = np.zeros_like(image)
    
    # Perform the convolution operation for each channel
    for ch in range(num_channels):
        for i in range(image_height):
            for j in range(image_width):
                patch = padded_image[i:i+kernel_height, j:j+kernel_width, ch]
                output_image[i, j, ch] = np.sum(patch * kernel)
    
    return output_image



if __name__ == '__main__':
    im = cv.imread('./imgs/mandrill.tif')
    k = np.array([  [0,1,0],
                    [2,8,2],
                    [0,1,0]])
    im2 = convolution(im, k/14)
    cv.imshow('im', im)
    cv.imshow('im2', im2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    