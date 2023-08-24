import numpy as np
# from skimage.metrics import structural_similarity as ssim
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model


def calculate_mse(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    return mse

def calculate_snr(im):
    signal_power = np.mean(im ** 2)
    noise_power = np.mean((im - np.mean(im)) ** 2)
    if noise_power == 0: return np.inf
    snr = 20 * np.log10(signal_power / noise_power)
    return snr

def calculate_psnr(im1, im2):
    mse = calculate_mse(im1, im2)
    if mse == 0: return np.inf
    psnr = 20 * np.log10(255 / np.sqrt(mse))  # in dB
    return psnr

# def calculate_ssim(im1, im2):
#     ssim_value = ssim(im1, im2, multichannel=True)
#     return ssim_value

# def create_vgg_model():
#     base_model = VGG16(weights='imagenet', include_top=False)
#     vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_conv3').output)
#     return vgg_model

# def calculate_vgg_loss(im1, im2):
#     vgg_model = create_vgg_model()
#     features1 = vgg_model.predict(im1)
#     features2 = vgg_model.predict(im2)
#     vgg_loss = np.mean((features1 - features2) ** 2)
#     return vgg_loss


def calculate_ssim(im1, im2, k1=0.01, k2=0.03, L=255):
    """ compute Structural Similarity Index (SSIM) between two images.\\
        1 indicating a perfect match and -1 indicating a complete mismatch.\\

        im1  : Any   deblured images using wiener filter with certain psf\\
        im2  : Any   original image without blur\\
        k1, k2  : float constants used for stability control.\\
        L   : int   max value of pixel
    """
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mean1 = np.mean(im1)
    mean2 = np.mean(im2)
    var1 = np.var(im1)
    var2 = np.var(im2)
    cov12 = np.cov(im1.flat, im2.flat)[0,1]
    numerator = (2*mean1*mean2 + c1) * (2*cov12 + c2)
    denominator = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
    return numerator / denominator