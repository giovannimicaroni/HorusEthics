import numpy as np
from PIL import Image
import random
import pywt
from skimage.restoration import denoise_wavelet

# Function to extract a certain number of patches of a certain size from an image
def extract_subimages(image_path, subimage_size = [100, 100], N=15):
    # Open the image and convert to a NumPy array
    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width, _ = img_array.shape
    
    subimages = []
    subimages_array = []
    
    for _ in range(N):
        # Generate random coordinates for the top-left corner of the subimage
        x = random.randint(0, img_width - subimage_size[0])
        y = random.randint(0, img_height - subimage_size[1])
        
        # Extract the subimage using NumPy slicing
        subimage_array = img_array[y:y + subimage_size[1], x:x + subimage_size[0]]
        subimages_array.append(subimage_array)
        
        # Convert the subimage array back to a Pillow Image
        subimage = Image.fromarray(subimage_array)
        
        subimages.append(subimage)
    
    return subimages, subimages_array

# Function to perform wavelet denoising on a single channel
def wavelet_denoising(channel, wavelet='db1', level=1, threshold=0.04):
    # Perform wavelet transform
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    
    # Thresholding the coefficients
    def thresholding(data, thresh):
        return pywt.threshold(data, thresh, mode='soft')
    
    # Apply the thresholding to detail coefficients
    coeffs_thresh = [coeffs[0]] + [tuple(thresholding(subband, threshold) for subband in detail_coeff) for detail_coeff in coeffs[1:]]
    
    # Reconstruct the image using the thresholded coefficients
    denoised_channel = pywt.waverec2(coeffs_thresh, wavelet)
    
    return denoised_channel

def denoise_rgb(image_array):
        R, G, B = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]

        # Apply wavelet denoising to each channel
        R_denoised = wavelet_denoising(R, wavelet='db1', level=2, threshold=20)
        G_denoised = wavelet_denoising(G, wavelet='db1', level=2, threshold=20)
        B_denoised = wavelet_denoising(B, wavelet='db1', level=2, threshold=20)

        # Clip the values to be in the valid range [0, 255] and convert to uint8
        R_denoised = np.uint8(np.clip(R_denoised, 0, 255))
        G_denoised = np.uint8(np.clip(G_denoised, 0, 255))
        B_denoised = np.uint8(np.clip(B_denoised, 0, 255))

        # Merge the denoised channels back into an RGB image
        denoised_image_array = np.stack((R_denoised, G_denoised, B_denoised), axis=-1)

        return denoised_image_array

if __name__ == '__main__':
    image_path = 'm.010lz5_0001.jpg'
    subimages, subimages_array = extract_subimages(image_path)
    noise_matrix = []
    for count, image_array in enumerate(subimages_array):
        denoised_image = denoise_rgb(image_array)
        print(denoised_image.shape)
        denoised_image_jpg = Image.fromarray(denoised_image)
        denoised_image_jpg.show()
        image_array_jpg = Image.fromarray(image_array)
        image_array_jpg.show()
        noise = image_array - denoised_image
        noise_matrix.append(noise)
    noise_matrix = np.array(noise_matrix) #matrix with shape (15, 100, 100, 3)
    avg_noise_matrix = np.mean(noise_matrix, axis=0) #matrix with the average noise of the 15 
    # img = Image.fromarray(avg_noise_matrix)
    # img.show()
    print(denoised_image.shape)

    print(avg_noise_matrix.shape)


    
    
        
