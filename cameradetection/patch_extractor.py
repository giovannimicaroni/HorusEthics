import random
import os
from PIL import Image
import numpy as np
from IMG_Dataset import IMG_Dataset

PATCH_SIZE = 256
N_PATCHES = 10

# Function to extract a certain number of patches of a certain size from an image
def extract_patches(img_array, img_number, dest_folder, patch_size = [PATCH_SIZE, PATCH_SIZE], N=N_PATCHES):
    # Open the image and convert to a NumPy array

    img_height, img_width, _ = img_array.shape
    
    patches = []
    patches_array = []
    
    for i in range(N):
        # Generate random coordinates for the top-left corner of the subimage
        x = random.randint(0, img_width - patch_size[0])
        y = random.randint(0, img_height - patch_size[1])
        
        # Extract the subimage using NumPy slicing
        patch_array = img_array[y:y + patch_size[1], x:x + patch_size[0]]
        patches_array.append(patch_array)
        
        # Convert the subimage array back to a Pillow Image
        patch = Image.fromarray(patch_array)
        patch.save(f'{dest_folder+str(img_number).zfill(6)+'.jpg'}')
        
        patches.append(patch)
    
    return patches, patches_array

if __name__ == '__main__':
    # dir = '5camerasdata'
    # for filename in os.listdir(dir):
    #     file_path = os.path.join(dir, filename)
    #     current_img = Image.open(file_path)
    #     current_img_array = np.array(current_img)
    #     patches, patches_array = extract_patches(current_img_array, filename)
    dataset = IMG_Dataset('KANFace.csv')
    i = 0
    for image, ethnicity in dataset:
        i += 1
        image_array = np.array(image)
        print(f'Extracting patches of image {i}...')
        image_dimensions = image_array.shape
        if image_dimensions[0] > PATCH_SIZE and image_dimensions[1] > PATCH_SIZE:
            patches, patches_array = extract_patches(image_array, i, 'C:/Users/g2317/OneDrive/Documentos/Unicamp/IC/Horus/cameradetection/patchesKANFace/')
