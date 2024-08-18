import os
from itertools import combinations
from pathlib import Path
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import random

# pré-processamento da imagem
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Adicionar batch dimension

# obter o embedding da imagem
def get_embedding(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        model.eval()
        embedding = model(image_tensor)
    return embedding

# comparar duas imagens
def compare_images(image_path1, image_path2, model):
    embedding1 = get_embedding(image_path1, model)
    embedding2 = get_embedding(image_path2, model)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# processar imagens em uma pasta e gerar o CSV
def process_images(base_directory, model, csv_filename, total_comparisons):
    num_clusters = 15
    comparisons_per_cluster = total_comparisons // num_clusters
    results = []
    comparison_count = 0
    message_threshold = 100  # Número de comparações após o qual uma mensagem é exibida

    # Processar cada pasta de cluster
    for cluster_id in range(num_clusters):
        cluster_directory = os.path.join(base_directory, f'cluster_{cluster_id}')
        image_paths = list(Path(cluster_directory).glob('*.jpg'))

        # agrupar imagens por prefixo de 7 dígitos
        groups = {}
        for image_path in image_paths:
            prefix = image_path.stem[:7]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(image_path)

        all_image_paths = [path for paths in groups.values() for path in paths]

        # se houver muitas imagens, amostrar uma quantidade menor
        if len(all_image_paths) > 1000:  # Ajuste o número conforme necessário
            all_image_paths = random.sample(all_image_paths, 1000)

        # pares aleatórios de imagens para comparação
        sampled_pairs = random.sample(list(combinations(all_image_paths, 2)), min(comparisons_per_cluster, len(all_image_paths)*(len(all_image_paths)-1)//2))

        # comparar imagens que têm 7 dígitos diferentes
        for img1, img2 in sampled_pairs:
            if img1.stem[:7] != img2.stem[:7]:  # Comparar apenas se os 7 primeiros dígitos forem diferentes
                distance = compare_images(img1, img2, model)
                results.append([img1.name, img2.name, distance, 0, cluster_id])

                #pParar se o número máximo de comparações for atingido
                if len(results) >= total_comparisons:
                    break
        if len(results) >= total_comparisons:
            break

    # mensagem indicando o número total de comparações realizadas
    if len(results) < total_comparisons:
        print(f"Realizou {len(results)} comparações, que é menos que o total esperado de {total_comparisons}.")

    # salvar em CSV
    df = pd.DataFrame(results, columns=['image A', 'image B', 'distance', 'same person', 'cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Resultados salvos em {csv_filename}")

# exemplo
if __name__ == "__main__":
    model = iresnet100(pretrained=False)
    model.load_state_dict(torch.load('C:/Users/emily/Downloads/295672backbone.pth', map_location='cpu')) # modelo baixado do github elasticface

    base_directory = 'C:/Users/emily/Downloads/RFW_dataset/separacao/clusters'
    csv_filename = 'C:/Users/emily/OneDrive/Documents/IC/resultados/elasticface/clusterizacaoFalse_elasticface.csv'
    total_comparisons = 6000  # número total de comparações

    # salvar resultados
    process_images(base_directory, model, csv_filename, total_comparisons)
