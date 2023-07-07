import cv2 as cv
from src.dip_func import *
from src.eval_func import *

if __name__ == '__main__':
    im0 = cv.imread('./imgs/voxel.jpg', cv.IMREAD_GRAYSCALE)
    im = cv.GaussianBlur(im0, (7,7), 3)  # add some blur
    im = add_salt_and_pepper_noise(im)  # add noise
    h,w = im.shape[:2]

    # upscaling
    imgs, imgs_resized = [im, im], [im]     # store resized upscaled image
    for i in range(5):
        print(i)
        im_tmp, im_resized_tmp = pyramid_upscaling(imgs[-1], (w,h))
        
        display_im(im_tmp, f'imfsize{i}', isShow=0, isSave=1)
        display_im(im_resized_tmp, f'imresized{i}', isShow=0, isSave=1)
        
        imgs[1] = im_tmp; imgs_resized.append(im_resized_tmp)

    # evaluation
    for i, img in enumerate(imgs_resized):
        print(f'original vs im{i}')
        print(f'MSE\t= {calculate_mse(im0, img)}')
        # print(f'SNR\t= {calculate_snr(img)}')
        print(f'PSNR\t= {calculate_psnr(im0, img)}')
        # print(f'MSE\t= {calculate_mse(imgs_resized[0], img)}')
        # print(f'PSNR\t= {calculate_psnr(imgs_resized[0], img)}')
        # print(f'SSIM\t= {calculate_ssim(imgs[0], im)}')
        # print(f'VGG Loss\t= {calculate_vgg_loss(imgs[0], im)}')
    cv.waitKey(0)
    cv.destroyAllWindows()