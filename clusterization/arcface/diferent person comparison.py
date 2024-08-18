import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from itertools import combinations
import random
from pathlib import Path

# aplicação de análise de faces com ArcFace iResNet100
app = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' inclui ArcFace iResNet100
app.prepare(ctx_id=0, det_thresh=0.5)  # ctx_id=0 para CPU, ctx_id=1 para GPU
parent_folder = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"

# limiar de similaridade
similarity_threshold = 0.5 

# obter o embedding de uma imagem
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) > 0:
        return faces[0].embedding  # Retorna o embedding do primeiro rosto encontrado
    return None

# comparar duas imagens
def compare_faces(img1_path, img2_path):
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)
    
    if emb1 is None or emb2 is None:
        return None
    
    # calcular a distância coseno entre os embeddings
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# processar imagens e gerar o CSV
def process_images(base_directory, total_comparisons, csv_filename):
    num_clusters = 15
    comparisons_per_cluster = total_comparisons // num_clusters
    results = []
    comparison_count = 0

    for cluster_id in range(num_clusters):
        cluster_directory = os.path.join(base_directory, f'cluster_{cluster_id}')
        image_paths = list(Path(cluster_directory).glob('*.jpg'))

        # se houver muitas imagens, amostrar uma quantidade menor
        if len(image_paths) > 1000:
            image_paths = random.sample(image_paths, 1000)

        # selecionar aleatoriamente pares de imagens dentro do mesmo cluster
        if len(image_paths) > 1:
            sampled_images = random.sample(image_paths, min(len(image_paths), 1000))
            sampled_pairs = random.sample(list(combinations(sampled_images, 2)), min(comparisons_per_cluster, len(sampled_images)*(len(sampled_images)-1)//2))

            for img1, img2 in sampled_pairs:
                similarity = compare_faces(img1, img2)
                if similarity is not None:
                    same_person = similarity > similarity_threshold
                    results.append([img1.name, img2.name, similarity, same_person, cluster_id])
                    
                # parar se o número máximo de comparações for atingido
                if len(results) >= total_comparisons:
                    break
        if len(results) >= total_comparisons:
            break

    # Mensagem indicando o número total de comparações realizadas
    if len(results) < total_comparisons:
        print(f"Realizou {len(results)} comparações, que é menos que o total esperado de {total_comparisons}.")

    # salvar em CSV
    df = pd.DataFrame(results, columns=['image A', 'image B', 'similarity', 'same person', 'cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Resultados salvos em {csv_filename}")

# exemplo
if __name__ == "__main__":
    base_directory = 'C:/Users/emily/Downloads/RFW_dataset/separacao/clusters'
    csv_filename = r"C:\Users\emily\OneDrive\Documents\IC\resultados\arcface\clusterizacaoFalse_arcface.csv"
    total_comparisons = 6000  # número total de comparações

    #salvar resultados
    process_images(base_directory, total_comparisons, csv_filename)
