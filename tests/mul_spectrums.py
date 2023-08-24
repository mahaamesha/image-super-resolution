import cv2
import numpy as np

# Generate two complex-valued arrays for multiplication
array1 = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
array2 = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
# Compute the shape differences between A and B
delta_rows = array2.shape[0] - array1.shape[0]
delta_cols = array2.shape[1] - array1.shape[1]
# Pad Array A with zeros to match the size of Array B
pad_top = delta_rows // 2
pad_bottom = delta_rows - pad_top
pad_left = delta_cols // 2
pad_right = delta_cols - pad_left
# Pad Array A from the center to match the size of Array B
array1_paded = np.pad(array1, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

print('a1\n', array1_paded)
print('a2\n', array2)

# Perform DFT on the input arrays
fft1 = np.fft.fftshift( np.fft.fft2(array1_paded) )#.astype(np.float32)
fft2 = np.fft.fftshift( np.fft.fft2(array2) )#.astype(np.float32)
print('fft1\n', fft1)
print('fft2\n', fft2)

# fft1_cv = np.fft.fftshift( cv2.dft(array1, flags=cv2.DFT_COMPLEX_OUTPUT) )        # same
# fft2_cv = np.fft.fftshift( cv2.dft(array2, flags=cv2.DFT_COMPLEX_OUTPUT) )
# print('fft1 cv\n', fft1_cv)
# print('fft2 cv\n', fft2_cv)

# Perform complex multiplication using cv.mulSpectrums
# product = cv2.mulSpectrums(fft1, fft2, flags=0)
product = np.multiply(fft1, fft2)
print('product\n', product)

# Apply inverse DFT to obtain the result
result = np.fft.ifft2( np.fft.ifftshift(product) )
# result = cv2.idft(product, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
print('result\n', result)
result_real = np.abs(result).astype(np.uint8)  # Take the magnitude or absolute value. In grayscale

# Display the result
cv2.imshow("A", array1)
cv2.imshow("A_paded", array1_paded)
cv2.imshow("B", array2)
# cv2.imshow("Product", product)
cv2.imshow("Result", result_real)
cv2.waitKey(0)
cv2.destroyAllWindows()