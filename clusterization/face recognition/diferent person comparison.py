import os
import numpy as np
import random
import pandas as pd
from itertools import combinations
from pathlib import Path
import face_recognition

parent_folder = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"

# limiar de similaridade
similarity_threshold = 0.6  # menor valor = maior similaridade, nesse caso

# obter o encoding de uma imagem
def get_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]  # Retorna o encoding do primeiro rosto encontrado
    return None

# comparar duas imagens
def compare_faces(img1_path, img2_path):
    enc1 = get_encoding(img1_path)
    enc2 = get_encoding(img2_path)
    
    if enc1 is None or enc2 is None:
        return None
    
    # distância euclidiana entre os encodings
    distance = np.linalg.norm(enc1 - enc2)
    return distance

# processar imagens e gerar o CSV
def process_images(base_directory, total_comparisons, csv_filename):
    num_clusters = 15
    comparisons_per_cluster = total_comparisons // num_clusters
    results = []
    comparison_count = 0

    try:
        for cluster_id in range(num_clusters):
            cluster_directory = os.path.join(base_directory, f'cluster_{cluster_id}')
            image_paths = list(Path(cluster_directory).glob('*.jpg'))
            groups = {}

            # agrupar imagens por prefixo de 7 dígitos
            # imagens com os primeiros 7 digitos iguais são da mesma pessoa
            for image_path in image_paths:
                prefix = image_path.stem[:7]
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(image_path)

            all_image_paths = [path for paths in groups.values() for path in paths]

            # com muitas imagens, amostrar uma quantidade menor
            if len(all_image_paths) > 1000:
                all_image_paths = random.sample(all_image_paths, 1000)

            # comparar imagens apenas imagens de 7 dígitos diferentes
            while len(results) < comparisons_per_cluster * (cluster_id + 1):
                if len(all_image_paths) < 2:
                    break

                # duas imagens aleatórias de diferentes grupos
                img1_group, img2_group = random.sample(list(groups.keys()), 2)
                img1 = random.choice(groups[img1_group])
                img2 = random.choice(groups[img2_group])

                distance = compare_faces(img1, img2)
                if distance is not None:
                    same_person = distance < similarity_threshold
                    results.append([img1.name, img2.name, distance, same_person, cluster_id])

                # verificar se alcançou o número máximo de comparações por cluster
                if len(results) >= comparisons_per_cluster * num_clusters:
                    break

            if len(results) >= comparisons_per_cluster * num_clusters:
                break

    except KeyboardInterrupt:
        print("Processo interrompido manualmente. Salvando os dados acumulados...")

    # se o número total de comparações for menor que o esperado, ajustar
    if len(results) < total_comparisons:
        print(f"Realizou {len(results)} comparações, que é menos que o total esperado de {total_comparisons}.")

    # salvar em CSV
    output_directory = os.path.dirname(csv_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.DataFrame(results, columns=['image A', 'image B', 'distance', 'same person', 'cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Resultados salvos em '{csv_filename}'")

# exemplo
if __name__ == "__main__":
    base_directory = 'C:/Users/emily/Downloads/RFW_dataset/separacao/clusters'
    csv_filename = r'C:\Users\emily\OneDrive\Documents\IC\resultados\face recognition\clusterizacaoFalse_fr.csv'
    total_comparisons = 6000  # número total de comparações

    # salvar resultados
    process_images(base_directory, total_comparisons, csv_filename)
