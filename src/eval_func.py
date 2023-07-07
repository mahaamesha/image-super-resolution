import numpy as np
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


def calculate_mse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse

def calculate_snr(image):
    signal_power = np.mean(image ** 2)
    noise_power = np.mean((image - np.mean(image)) ** 2)
    if noise_power == 0: return np.inf
    snr = 20 * np.log10(signal_power / noise_power)
    return snr

def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0: return np.inf
    psnr = 20 * np.log10(255 / np.sqrt(mse))  # in dB
    return psnr

def calculate_ssim(image1, image2):
    ssim_value = ssim(image1, image2, multichannel=True)
    return ssim_value

def create_vgg_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_conv3').output)
    return vgg_model

def calculate_vgg_loss(image1, image2):
    vgg_model = create_vgg_model()
    features1 = vgg_model.predict(image1)
    features2 = vgg_model.predict(image2)
    vgg_loss = np.mean((features1 - features2) ** 2)
    return vgg_loss