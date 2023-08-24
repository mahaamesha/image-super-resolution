import cv2 as cv
from src.dip_func import *
from src.eval_func import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    im_path = './imgs/granite.png'
    ksize, sigma = 21, 7
    salt_prob, pepper_prob = 0, 0
    
    dK = 0.0001
    Kloops = 10000
    
    im0 = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
    h,w = im0.shape[:2]
    
    # add blur n noise
    psf = create_gaussian_filter(ksize, sigma)
    psf_paded = add_padding_kernel(psf, size_like=im0)  # used in freq domain
    im_blur = cv.filter2D(im0, ddepth=cv.CV_8U, kernel=psf)   # add blur

    # denoising
    if salt_prob != 0 and pepper_prob != 0:
        im_noisy = add_salt_and_pepper_noise(im_blur, salt_prob, pepper_prob)   # add noise
        im_denoised = cv.medianBlur(im_noisy, ksize=ksize)
    else: im_denoised = im_blur.copy()
    im_denoised_fft,_,_ = get_fft_image(im_denoised)
    
    # deblurring
    arr_K = []
    arr_mse = []
    arr_ssim = []
    for i in range(0, Kloops):
        K = i * dK
        Guv = get_inverse_filter(psf_paded, K)
        im_deblur = deblurring_image_fdomain(im_denoised_fft, Guv)
        im_deblur = image_mirroring_quadrant(im_deblur)     # because quadrant I<>III, II<>IV
        
        mse_tmp = calculate_mse(im_deblur, im0)
        ssim_tmp = calculate_ssim(im_deblur, im0)
        
        arr_K.append(K)
        arr_mse.append(mse_tmp)
        arr_ssim.append(ssim_tmp)
        print('i={}, K={:.4f}, mse={:.4f}, ssim={:.4f}'.format(i, K, mse_tmp, ssim_tmp))
    
    # deblur using best K value
    Kidx = np.argmin(arr_mse)
    # Kidx = np.argmax(arr_ssim)
    print(f'BEST -> K={arr_K[Kidx]}, mse={arr_mse[Kidx]}, ssim={arr_ssim[Kidx]}')
    Guv = get_inverse_filter(psf_paded, K=arr_K[Kidx])
    im_deblur = deblurring_image_fdomain(im_denoised_fft, Guv)
    im_deblur = image_mirroring_quadrant(im_deblur)


    # plot mse vs K
    plt.plot(arr_K, arr_mse, label='mse')
    plt.plot(arr_K, arr_ssim, label='ssim')
    plt.legend()
    plt.show()
    
    display_n_save('01 original', im0)
    # display_n_save('02 blur + noise', im_noisy)
    display_n_save('03 blur denoised', im_denoised)
    display_n_save('04 FFT blur denoised', np.log10(1+np.abs(im_denoised_fft)))
    display_n_save('05 psf', psf)
    display_n_save('06 otf', np.log10(1+np.abs(Guv)))
    display_n_save('07 deblur', im_deblur)
    plt.show()
    
    cv.waitKey(0)
    cv.destroyAllWindows()