import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


def add_salt_and_pepper_noise(im, salt_prob:float=0.01, pepper_prob:float=0.01):
    im_noisy = np.copy(im)
    h, w = im_noisy.shape[:2]
    # generate random noise mask
    salt_mask = np.random.random(size=(h, w)) < salt_prob
    pepper_mask = np.random.random(size=(h, w)) < pepper_prob
    im_noisy[salt_mask] = 255    # salt noise
    im_noisy[pepper_mask] = 0    # pepper noise
    return im_noisy


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
    """ Deblur an image in the frequency domain.
        
        Yuv : Blurred image in the frequency domain.
        Guv : Inverse filter G(u,v). The ideal inverse filter G(u,v) = 1 / H(u,v).
        
        Returns:
        Xxy : Deblurred image in the spatial domain.
    """
    # if Yuv.shape != Guv.shape: Guv = add_padding_kernel(Guv, size_like=Yuv)
    # Xuv = cv.mulSpectrums(Yuv, Guv, flags=cv.DFT_COMPLEX_OUTPUT)
    Xuv = np.multiply(Yuv, Guv)
    Xxy = np.fft.ifft2(np.fft.ifftshift(Xuv))
    Xxy = np.abs(Xxy).astype(np.uint8)

    return Xxy


def calculate_mse(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    return mse


def image_mirroring_quadrant(im):
    """ the result is mirrored, I<>III, II<>IV \n"""
    h,w = im.shape[:2]
    im2 = np.zeros_like(im)
    im2[:h//2, :w//2] = im[h//2:, w//2:]    # II in im2 = IV in im
    im2[:h//2, w//2:] = im[h//2:, :w//2]    # I
    im2[h//2:, :w//2] = im[:h//2, w//2:]    # III
    im2[h//2:, w//2:] = im[:h//2, :w//2]    # IV
    return im2


def display_n_save(title, im, isShow:bool=0):
    fig = plt.figure()
    plt.title(title); plt.imshow(im, cmap='gray')
    fig.savefig(os.path.join( os.getcwd(), f'./tests/weiner/{title}.jpg'))
    plt.tight_layout(); 
    if isShow: plt.show()


def test():
    im = cv.imread(os.getcwd() + './imgs/granite.png', cv.IMREAD_GRAYSCALE)
    im_fft, im_fft_amplitude, im_fft_phase = get_fft_image(im)
    im_ifft = np.fft.ifft2(np.fft.ifftshift(im_fft))
    im_ifft = np.abs(im_ifft).astype(np.uint8)  # normalize in 0-255 scale
    plt.figure(1), plt.title('im'), plt.imshow(im, cmap='gray')
    plt.figure(2), plt.title('im_fft_amplitude'), plt.imshow(im_fft_amplitude, cmap='gray')
    plt.figure(3), plt.title('im_fft_phase'), plt.imshow(im_fft_phase, cmap='gray')
    plt.figure(4), plt.title('im_ifft'), plt.imshow(im_ifft, cmap='gray')
    plt.figure(5), plt.title('abs diff'), plt.imshow(cv.absdiff(im, im_ifft), cmap='gray'), plt.colorbar()
    plt.show()



if __name__ == '__main__':
    im = cv.imread(os.getcwd() + './imgs/granite.png')
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # blur the image
    psf = create_gaussian_filter(ksize=7, sigma=3)
    psf_paded = add_padding_kernel(psf, size_like=im_gray)
    im_blur = cv.filter2D(im_gray, ddepth=cv.CV_8U, kernel=psf_paded)
    # add noise
    im_noisy = add_salt_and_pepper_noise(im_blur, salt_prob=0.01, pepper_prob=0.01)

    # preprocessing
    im_denoised = cv.medianBlur(im_noisy, 7)
    im_denoised_fft,_,_ = get_fft_image(im_denoised)

    # deblurring
    arr_K, arr_mse = [], []
    # for i in range(0,1):
    #     K = 0.0039000000000000003
    for i in range(0,1):
        K = i * 0.00001
        K = 0.01
        Guv = get_inverse_filter(psf_paded, K=K)
        im_deblur = deblurring_image_fdomain(im_denoised_fft, Guv)
        im_deblur = image_mirroring_quadrant(im_deblur)
        mse_tmp = calculate_mse(im_gray, im_deblur)
        arr_K.append(K); arr_mse.append(mse_tmp)
        print(i, K, mse_tmp)
    # deblur using best value of K
    idx = np.argmin(arr_mse)
    Guv = get_inverse_filter(psf_paded, K=arr_K[idx])
    im_deblur = deblurring_image_fdomain(im_denoised_fft, Guv)
    im_deblur = image_mirroring_quadrant(im_deblur)

    im_diff = cv.absdiff(im_denoised, im_deblur).astype(np.uint8)

    print('im_gray\n', im_gray)
    print('psf\n', psf_paded)
    print('im_blur_fft\n', im_denoised_fft)
    print('Guv\n', Guv)
    print(f'K={arr_K[idx]}, MSE={arr_mse[idx]}')

    display_n_save('original', im)
    display_n_save('blur + noise', im_noisy)
    display_n_save('blur denoised', im_denoised)
    display_n_save('FFT blur denoised', np.log10(1+np.abs(im_denoised_fft)))
    display_n_save('psf', psf_paded)
    display_n_save('otf', np.log10(1+np.abs(Guv)))
    display_n_save('blur denoised', im_denoised)
    display_n_save('deblur', im_deblur, isShow=1)
    # display_n_save('|blur-deblur|', im_diff, isShow=1)

    # cv.waitKey(0)
    # cv.destroyAllWindows()