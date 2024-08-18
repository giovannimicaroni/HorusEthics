# o uso do cvl é mais complicado, precisa seguir uma estrutura especifica, então a cada cluster:
# eu renomeava a pasta para "image" (necessário), rodava esse código e depois o cvl, trocava o nome da pasta para o cluster de novo
# renomeava o csv que saia (ex: resultado do cluster 3 era result3), e fazia a mesma coisa com o próximo, depois apenas juntei os csv

import os
import random
import pandas as pd
from pathlib import Path
from itertools import combinations

# processar imagens e gerar o CSV no formato desejado
def generate_pairs_csv(cluster_directory, csv_filename, total_pairs=400):
    image_paths = list(Path(cluster_directory).glob('*.jpg'))

    # todas as combinações possíveis de pares de imagens
    possible_pairs = list(combinations(image_paths, 2))

    # filtrar pares com os primeiros 7 dígitos iguais
    filtered_pairs = [
        (img1, img2) for img1, img2 in possible_pairs 
        if img1.stem[:7] != img2.stem[:7]
    ]

    # se o número de pares filtrados for menor que o número total de pares desejado, ajustar
    if len(filtered_pairs) < total_pairs:
        print(f"Há apenas {len(filtered_pairs)} pares disponíveis após o filtro. Usando todos eles.")
        selected_pairs = filtered_pairs
    else:
        # selecionar aleatoriamente os pares desejados
        selected_pairs = random.sample(filtered_pairs, total_pairs)

    # criar DataFrame
    data = []
    for idx, (img1, img2) in enumerate(selected_pairs):
        data.append([idx, img1.name, img2.name])

    df = pd.DataFrame(data, columns=['Index', 'A', 'B'])
    df.to_csv(csv_filename, index=False)
    print(f"CSV gerado em {csv_filename}")

# exemplo
if __name__ == "__main__":
    # Definir caminho do cluster e nome do CSV
    cluster_directory = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters\images"  # escolha o cluster desejado aqui
    csv_filename = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters\pairs.csv"

    generate_pairs_csv(cluster_directory, csv_filename, total_pairs=400)
