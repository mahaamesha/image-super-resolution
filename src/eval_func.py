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