import random
import os
from PIL import Image
import numpy as np
from IMG_Dataset import IMG_Dataset

PATCH_SIZE = 256
N_PATCHES = 10
DATASET = 'RFW'

# Function to extract a certain number of patches of a certain size from an image
def extract_patches(img, img_number, img_path, dest_folder, patch_size = [PATCH_SIZE, PATCH_SIZE], N=N_PATCHES):
    # Open the image and convert to a NumPy array
    img_array = np.array(img)
    img_height, img_width = img_array.shape
    
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
        dest = os.path.join(dest_folder, f'{str(i).zfill(2)}.npy')
        np.save(dest, patch_array)
        # patch_name = f'{str(i).zfill(2)}.jpg'
        # patch.save(f'{dest_folder+patch_name}')
        
        patches.append(patch)
    
    return patches, patches_array

if __name__ == '__main__':
    # dir = '5camerasdata'
    # for filename in os.listdir(dir):
    #     file_path = os.path.join(dir, filename)
    #     current_img = Image.open(file_path)
    #     current_img_array = np.array(current_img)
    #     patches, patches_array = extract_patches(current_img_array, filename)
    
    current_dir = os.getcwd()
    noiseprint_dir = os.path.join(current_dir, f'C:/IC/outputRFW2/outputRFW2')
    patches_dir = os.path.join(current_dir, f'C:/IC/patchesRFW2')
 
    os.makedirs(patches_dir, exist_ok=True)
    i = 0

    for npz in os.listdir(noiseprint_dir):
        current_path = os.path.join(noiseprint_dir, npz)
        current_npz = np.load(current_path)
        current_noiseprint = current_npz['np++']
        noiseprint_dimensions = current_noiseprint.shape
        
        if noiseprint_dimensions[0] > PATCH_SIZE and noiseprint_dimensions[1] > PATCH_SIZE: 
            i += 1
            print(f'Extracting patches of image {i}...')
            patches_dest_dir = os.path.join(patches_dir, npz[:-8])
            os.makedirs(patches_dest_dir, exist_ok=True)
            patches, patches_array = extract_patches(current_noiseprint, i, current_path, patches_dest_dir)
