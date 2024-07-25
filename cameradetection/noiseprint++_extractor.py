import numpy as np
from PIL import Image
import os

IMG_SIZE = 1024

if __name__ == '__main__':
    # data = np.load('output5cameras/croppedipad00.jpg.npz')
    # print(data.files)
    lista_noises = []
    dir = 'output5cameras'
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        current_npz = np.load(file_path) #opens the .npz file that contains the noiseprint++
        img = Image.open('croppedimages/'+filename[:-4]) #open the image related do the .npz
        img_array = np.array(img)
        current_noiseprint = current_npz['np++'] #extracts the noiseprint++ from the .npz
        img_vector = np.zeros((IMG_SIZE,IMG_SIZE))
        img_vector = 0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2] #transforms the 1024x1024x3 image into a 1024x1024
        noise = img_vector - current_noiseprint #calculates the noise, which is the difference between the original image and the np++
        lista_noises.append(noise)

    array_noises = np.array(lista_noises) #array_noises stores the values of the noises of all images
    avg_noise_array = np.mean(array_noises, axis=0) #calculates the mean of the noises of all images
    pooled_vector = np.mean(avg_noise_array, axis=1) #applies an average pooling to the mean of the noises of all images to obtain a vector
    np.save('np++5cameras.npy', avg_noise_array)
    print('Array saved as np++5cameras.npy')