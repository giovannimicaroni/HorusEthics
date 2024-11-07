import random
import os
from PIL import Image
import numpy as np
import torch

PATCH_SIZE = 256
N_PATCHES = 10
DATASET = 'WSD'

# Function to extract a certain number of patches of a certain size from an image
def extract_patches(img_array, img_number, img_path, dest_folder, patch_size = [PATCH_SIZE, PATCH_SIZE], N=N_PATCHES):
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
        patch_np = patch_array.numpy()
        patch = Image.fromarray(patch_np)
        patch_name = img_path
        patch.save(os.path.join(dest_folder, f'{patch_name[:-4]+'_'+str(i)}.jpg'))
        
        patches.append(patch)
    
    return patches, patches_array

if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    current_dir = os.getcwd()
    imgs_directory = os.path.join(current_dir, 'WSD_Dataset/train')
    dest_dir =  os.path.join(current_dir, f'patches{DATASET}')
    os.makedirs(dest_dir, exist_ok=True)
    i = 0

    for directory in os.listdir(imgs_directory):
        current_dir = os.path.join(imgs_directory, directory)
        for file in os.listdir(current_dir):
            i += 1
            selfie_path = os.path.join(current_dir, file)
            current_img = Image.open(selfie_path)
            current_img_array = np.array(current_img)
            current_img_tensor = torch.from_numpy(current_img_array).to(device)
            patches, patches_array = extract_patches(current_img_tensor, i, file, dest_dir)
            print(i)
