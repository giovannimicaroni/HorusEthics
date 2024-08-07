import numpy as np
from PIL import Image
import random
import pywt

from IMG_Dataset import IMG_Dataset
import torch

PATCH_SIZE = 100
N_PATCHES = 50

# Function to extract a certain number of patches of a certain size from an image
def extract_patches(img_array, patch_size = [PATCH_SIZE, PATCH_SIZE], N=N_PATCHES):
    # Open the image and convert to a NumPy array

    img_height, img_width, _ = img_array.shape
    
    patches = []
    patches_array = []
    
    for _ in range(N):
        # Generate random coordinates for the top-left corner of the subimage
        x = random.randint(0, img_width - patch_size[0])
        y = random.randint(0, img_height - patch_size[1])
        
        # Extract the subimage using NumPy slicing
        patch_array = img_array[y:y + patch_size[1], x:x + patch_size[0]]
        patches_array.append(patch_array)
        
        # Convert the subimage array back to a Pillow Image
        patch = Image.fromarray(patch_array)
        
        patches.append(patch)
    
    return patches, patches_array

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
    dataset = IMG_Dataset(csv_file='5camera.csv')
    noise_vectors_list = []

    for i in range(0, len(dataset)):
        image, ethnicity = dataset[i]
        width, height = image.size
        cropped_image = image.crop(((width - 1024) // 2, (height - 1024) // 2, (width + 1024) // 2, (height + 1024) // 2))
        image_array = np.array(cropped_image)
        denoise_image_array = denoise_rgb(image_array)
        noise = image_array - denoise_image_array
        patches, patches_array = extract_patches(noise)

        avg_noise_matrix = np.mean(patches_array, axis=0) #matrix with the average noise of the 15 patches
        #matriz 100x100x3 da media dos 15 patches
        noise_vector = []
        for j in range(PATCH_SIZE):
            noise_vector.append(0.299*avg_noise_matrix[j][j][0] + 0.587*avg_noise_matrix[j][j][1] + 0.114*avg_noise_matrix[j][j][2]) 
        #TODO testar fazer para todos os pixeis, vai resultar em uma matriz 100x100, e depois fazer um pooling por coluna
        noise_vector = np.array(noise_vector) # vetor de 100 elementos
        noise_vectors_list.append(noise_vector)
        print(f'{i+1} images done')
        
    noise_vectors_array = np.array(noise_vectors_list)
    np.save('5camera_noises.npy', noise_vectors_array)
    print('Array saved as 5camera_noises.npy')
    
    
    
        
