import numpy as np
from PIL import Image
import os
import torch
import pandas as pd

IMG_SIZE = 1024
DATASET = '5camerasdata'


if __name__ == '__main__':
    # data = np.load('output5cameras/croppedipad00.jpg.npz')
    # print(data.files)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feature_vectors_list = []
    dir_patches = f'C:/Users/g2317/OneDrive/Documentos/Unicamp/IC/Horus/cameradetection/outputp5cameras/outputp5cameras'
    dir_images = f'C:/Users/g2317/OneDrive/Documentos/Unicamp/IC/Horus/cameradetection/patches5cameras'
    images = os.listdir(dir_images)
    ids_matrix = []
    i = 0
    for directory in os.listdir(dir_patches):

        dir_path = os.path.join(dir_patches, directory)
        current_patches_list = []
        if os.path.isdir(dir_path):
            for patch in os.listdir(dir_path):
                patch_path = os.path.join(dir_path, patch)
                current_npz = np.load(patch_path)
                current_image = Image.open(os.path.join(dir_images, patch[:-4]))
                current_patch = current_npz['np++']
                current_patch = torch.from_numpy(current_patch).to(device)
                print(patch[:-4])
                print(patch)
                current_image_array = np.array(current_image)
                current_image_array = torch.from_numpy(current_image_array).to(device)
                current_image_grayscale = 0.299*current_image_array[:,:,0] + 0.587*current_image_array[:,:,1] + 0.114*current_image_array[:,:,2]
                noise = current_image_grayscale - current_patch
                current_patches_list.append(noise)

            patches_matrix = np.array(current_patches_list)
            patches_matrix = torch.from_numpy(patches_matrix).to(device)
            patches_mean_matrix = patches_matrix.mean(dim=0)
            feature_vector = patches_mean_matrix.mean(dim=1)
            ids_matrix.append(directory)
            feature_vectors_list.append(feature_vector)        

        i+=1
        print(f'{i} images done')

    ids_df = pd.DataFrame(ids_matrix)
    ids_df.to_csv('identitiesmatrix.csv')
    feature_vectors_matrix = torch.stack(feature_vectors_list)
    torch.save(feature_vectors_matrix, f'featurematrix{DATASET}.pt' )
    print(f'Array saved as featurematrix{DATASET}.pt and identities as identitiesmatrix.csv')
    

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