#!/usr/bin/env python3

import numpy as np
from PIL import Image
import os
import pandas as pd
import torch
import re

IMG_SIZE = 1024
DATASET = 'RFW'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == '__main__':
    # data = np.load('output5cameras/croppedipad00.jpg.npz')
    # print(data.files)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    feature_vectors_list = []
    current_dir = os.getcwd()
    dir_patches = os.path.join(current_dir, f'output{DATASET}')
    dir_images = os.path.join(current_dir, f'images/patches{DATASET}')
    i = 0
    current_noises_list = []
    ids_matrix = []
    files = os.listdir(dir_patches)
    files.sort(key=natural_sort_key)

    for filename in files:
        
        current_patch_path = os.path.join(dir_patches, filename)
        current_image_patch_path = os.path.join(dir_images, filename[:-4]) #remove the .npz from filename
        current_npz = np.load(current_patch_path) #opens the .npy file that contains the noiseprint++ of a patch
        current_patch = current_npz['np++']
        current_patch = torch.from_numpy(current_patch).to(device)
        current_image_patch = Image.open(current_image_patch_path)
        image_patch_array = np.array(current_image_patch)
        image_patch_array = torch.from_numpy(image_patch_array).to(device)
        grayscale_image_patch =  0.299*image_patch_array[:,:,0] + 0.587*image_patch_array[:,:,1] + 0.114*image_patch_array[:,:,2] 
        noise = grayscale_image_patch - current_patch
        current_noises_list.append(noise)
                
        if(filename[-9] == '9'): #last patch of an image
            
            i+=1
            patches_matrix = np.array(current_noises_list)
            patches_matrix = torch.from_numpy(patches_matrix).to(device)
            patches_mean_matrix = patches_matrix.mean(dim=0)
            feature_vector = patches_mean_matrix.mean(dim=1)
            feature_vectors_list.append(feature_vector)
            ids_matrix.append(filename[:-10])
            current_noises_list = []
            print(f'{i} images done')

        

    ids_df = pd.DataFrame(ids_matrix)
    ids_df.to_csv('identitiesmatrix.csv')
    feature_vectors_matrix = torch.stack(feature_vectors_list)
    torch.save(feature_vectors_matrix, f'featurematrix{DATASET}.pt')
    print(f'Array saved as featurematrix{DATASET}.pt')
    

    # array_noises = np.array(lista_noises) #array_noises stores the values of the noises of all images
    # avg_noise_array = np.mean(array_noises, axis=0) #calculates the mean of the noises of all images
    # pooled_vector = np.mean(avg_noise_array, axis=1) #applies an average pooling to the mean of the noises of all images to obtain a vector
    # print(pooled_vector.shape)
        # img = Image.open(f'patches{DATASET}/'+filename[:-4]) #open the image related do the .npz
        # img_array = np.array(img)
        # current_noiseprint = current_npz['np++'] #extracts the noiseprint++ from the .npz
        # img_vector = np.zeros((IMG_SIZE,IMG_SIZE))
        # img_vector = 0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2] #transforms the 1024x1024x3 image into a 1024x1024
        # noise = img_vector - current_noiseprint #calculates the noise, which is the difference between the original image and the np++
        # lista_noises.append(noise)