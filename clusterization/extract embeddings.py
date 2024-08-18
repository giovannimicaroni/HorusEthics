import os
from deepface import DeepFace
import numpy as np

# carregar imagens
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, filename)
            images.append(img_path)
            filenames.append(filename)
    return images, filenames

# extrair
def extract_embeddings(images):
    embeddings = []
    for idx, img in enumerate(images):
        try:
            representation = DeepFace.represent(img_path=img, model_name="Facenet", enforce_detection=False)
            embedding = representation[0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Erro ao processar {img}: {e}")
            continue
    return embeddings

# diretório onde estão as imagens
folder_path = r"C:\Users\emily\Downloads\RFW_dataset\separacao\total"

# carregar as imagens da pasta
images, filenames = load_images_from_folder(folder_path)

# extrair embeddings
all_embeddings = extract_embeddings(images)

# converter embeddings para um array
all_embeddings = np.array(all_embeddings)

print("Extração completa. Total de embeddings extraídos:", len(all_embeddings))
