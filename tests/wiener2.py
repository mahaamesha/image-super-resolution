import cv2
import numpy as np
import os

def calc_psf(output_img, filter_size, R):
    h = np.zeros(filter_size, dtype=np.float32)
    point = (filter_size[0] // 2, filter_size[1] // 2)
    cv2.circle(h, point, R, 255, -1, 8)
    summa = np.sum(h)
    output_img[:] = h / summa

def fft_shift(input_img, output_img):
    output_img[:] = np.fft.fftshift(input_img)

def filter_2D_freq(input_img, output_img, H):
    input_img = input_img.astype(np.float32)  # Convert to float32 data type

    complexI = cv2.dft(input_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    complexI = complexI[:, :, 0] + complexI[:, :, 1] * 1j
    
    complexIH = complexI * np.conj(H)
    complexIH = cv2.idft(complexIH, flags=cv2.DFT_SCALE)
    output_img[:] = complexIH.real

def calc_wnr_filter(input_h_PSF, output_G, nsr):
    h_PSF_shifted = np.fft.fftshift(input_h_PSF)
    complexI = cv2.dft(h_PSF_shifted, flags=cv2.DFT_COMPLEX_OUTPUT)
    denom = np.abs(complexI[:, :, 0]) ** 2 + nsr
    output_G[:] = complexI[:, :, 0] / denom

def deblur_image(image_path, R, snr):
    img_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_in is None:
        print("ERROR: Image cannot be loaded!")
        return

    img_out = np.empty_like(img_in)
    roi = (0, 0, img_in.shape[1] & -2, img_in.shape[0] & -2)

    # Calculate PSF
    psf = np.empty(roi[2:], dtype=np.float32)
    calc_psf(psf, roi[2:], R)

    # Calculate Wiener filter
    Hw = np.empty_like(psf)
    calc_wnr_filter(psf, Hw, 1.0 / snr)

    # Apply filtering
    filter_2D_freq(img_in[roi[1]:roi[3], roi[0]:roi[2]], img_out, Hw)

    # Normalize and convert output image
    img_out = cv2.normalize(img_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_out

def main():
    str_in_file_name = os.getcwd() + "./imgs/building.png"
    R = 5
    snr = 100

    img_deblur = deblur_image(str_in_file_name, R, snr)

    cv2.imshow("Original", cv2.imread(str_in_file_name, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("Deblurring", img_deblur)
    cv2.imwrite("result.jpg", img_deblur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
