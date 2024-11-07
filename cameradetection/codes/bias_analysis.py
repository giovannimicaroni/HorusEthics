import pandas as pd
import os
import numpy as np
from PIL import Image, ImageFilter, ImageStat

DATASET = 'WSD'
IMAGES_DIR = '/home/gimicaroni/Documents/Datasets/WSD_Dataset/train'
N_CLUSTERS = 20

def calculate_contrast(image):
    # Convert the image to grayscale
    grayscale = image.convert('L')
    
    # Get image statistics
    stat = ImageStat.Stat(grayscale)
    
    # Contrast is the difference between the maximum and minimum pixel values
    contrast = stat.extrema[0][1] - stat.extrema[0][0]
    
    return contrast

def calculate_sharpness(image):
    # Apply an edge detection filter
    edges = image.filter(ImageFilter.FIND_EDGES)
    
    # Convert to grayscale for easier calculation
    edges = edges.convert('L')
    
    # Calculate the variance of the edge pixels (higher variance = sharper image)
    sharpness = np.var(np.array(edges))
    
    return sharpness

def calculate_brightness(image):
    # Convert the image to grayscale
    grayscale = image.convert('L')
    
    # Calculate the average pixel value (brightness)
    brightness = np.mean(np.array(grayscale))
    
    return brightness

def calculate_noise(image):

    
    # Convert image to grayscale (if it isn't already)
    image = image.convert('L')
    
    # Convert image to NumPy array
    image_array = np.array(image)
    
    # Calculate the standard deviation of pixel values
    noise = np.std(image_array)
    
    return noise

if __name__ == '__main__':
    current_dir = os.getcwd()
    images_dir = IMAGES_DIR
    images_path = os.path.join(current_dir, images_dir)

    df = pd.read_csv(f'~/Documents/Unicamp/IC/HorusProjeto/HorusEthics/cameradetection/docs/clustering/reduced_clusterized{DATASET}{N_CLUSTERS}.csv')
    columns = ['kmeans', 'ids']
    simplified_df = df.filter(items=columns)
    simplified_df["identity"] = simplified_df["ids"].str.slice(0, 5)
    grouped_df = simplified_df.groupby('kmeans')

    for kmeans_value, group in grouped_df:
        image_feature = []
        # print(f'kmeans = {kmeans_value}: {len(group)}')
        for idx, (index, row) in enumerate(group.iterrows()):
            identity_folder = os.path.join(IMAGES_DIR, row["identity"])
            image_path = os.path.join(identity_folder, row["ids"])
            current_image = Image.open(f'{image_path}.jpg')
            image_feature.append(calculate_noise(current_image))
        image_feature_array = np.array(image_feature)
        image_feature_mean = np.mean(image_feature_array)
        print(f'kmeans = {kmeans_value}: {image_feature_mean}')
            


#não eh o total de linhas no cluster
#não eh o numero de identidades diferentes
#não eh o tamanho das imagens
#não eh sharpness
#não eh brightness
#não eh contraste
#não eh um noise simples

#perguntar o que analisar no cluster, ele ja esta agrupado pelo resultado do noiseprint++, mas que fatores eu deveria analisar para procurar um vies?


#conclusões:
#cluster 16, que obteve maior erros no true negative, possui muitas imagens com filtros, como preto e branco, ou onde as pessoas estão de oculos
#muitas imagens borradas ou com qualidade baixa nesse cluster tambem. Normalmente, em uma mesma identidade no mesmo cluster, o algoritmo costuma errar a validação nas mesmas imagens
#os filtros aumentam a porcentagem de erro e, se as duas imagens sendo comparadas tiverem filtros, a probabilidade de erro é bem maior. o filtro preto e branco afeta a probabilidade mas afeta menos do que outros filtros mais complexos. Oculos tambem afeta
#fotos parecidas (com o mesmo fundo, mesma posição, mas com expressoes faciais diferentes) costumam ter o mesmo resultado, se uma é caracterizada errada muito provavelmente as outras também vao