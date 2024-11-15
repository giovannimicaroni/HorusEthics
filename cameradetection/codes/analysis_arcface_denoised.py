import pandas as pd
import os
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np

DATASET = 'WSD'
IMAGES_DIR = '/home/gimicaroni/Documents/Datasets/WSD_Dataset/train'
NOISEPRINT_DIR = '/work/giovanni.micaroni/TruFor/test_docker/outputWSD'
N_CLUSTERS = 20
PROGRESS = 0
SIMILARITY_THRESHOLD = 0.5
MAX_COMPARISONS_PER_IMAGE = 25

def get_denoised_embedding(image_path, noiseprint_path):
    img = cv2.imread(image_path)
    img_array = np.array(img)
    npz = np.load(noiseprint_path) # imagens do openCV são arrays numpy, mas devem ser inteiros de 8 bits
    noise = npz['np++']
    denoised_img_array = img_array - noise
    denoised_img = np.uint8(denoised_img_array)
    faces = app.get(denoised_img)
    if len(faces) > 0:
        return faces[0].embedding  # Retorna o embedding do primeiro rosto encontrado
    return None

# comparar duas imagens
def compare_faces(emb1, emb2):

    if emb1 is None or emb2 is None:
        return None
    
    # calcular a distância coseno entre os embeddings
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if similarity > SIMILARITY_THRESHOLD:
        return True
    return False

def face_comparisons(grouped_df, app, current_dir, images_path, noiseprints_path):

    tp_percentages = []
    tn_percentages = []
    print(grouped_df['kmeans'].value_counts())

    try:
        for kmeans_value, group in grouped_df:
            if kmeans_value >= PROGRESS:
                print(f'starting cluster {kmeans_value}...')
                total_p = 0
                total_n = 0
                true_positives = 0
                true_negatives = 0

                for idx, (index, row) in enumerate(group.iterrows()):
                    current_identity = f'/{row['ids'][:5]}'
                    current_image_encodings = get_denoised_embedding(images_dir + current_identity + f'/{row['ids']}.jpg',
                                                             noiseprints_path + current_identity + f'/{row['ids']}.jpg.npz')
                    img_comparisons = 0
                    
                    if not current_image_encodings is None:
                        sampled_group = group.sample(MAX_COMPARISONS_PER_IMAGE)
                        for next_index, next_row in sampled_group.iterrows():

                            next_identity = f'/{next_row['ids'][:5]}'
                            next_image_encodings = get_denoised_embedding(images_dir + next_identity + f'/{next_row['ids']}.jpg',
                                                                 noiseprints_path + next_identity + f'/{next_row['ids']}.jpg.npz')

                            if not next_image_encodings is None:
                                comparison_result = compare_faces(current_image_encodings, next_image_encodings)
                                print(f'{current_identity} {next_identity}: {comparison_result}')
                                
                                if current_identity == next_identity:
                                    total_p += 1
                                    if comparison_result == True:
                                        true_positives += 1
                                else:
                                    total_n += 1
                                    if comparison_result == False:
                                        true_negatives += 1
                                img_comparisons += 1
                                if img_comparisons > MAX_COMPARISONS_PER_IMAGE:
                                    print(f'{MAX_COMPARISONS_PER_IMAGE} comparisons done for this image, continuing..')
                                    break
                            

                tp_percentage = true_positives / total_p
                tn_percentage = true_negatives / total_n
                tp_percentages.append(tp_percentage)
                tn_percentages.append(tn_percentage)
                print(f'cluster {kmeans_value} done {tp_percentage}, {tn_percentage}')

    except KeyboardInterrupt:
        print(f"Process interrupted, saving progress of cluster{kmeans_value}...")
        tn_percentages_array = np.array(tn_percentages)  
        tp_percentages_array = np.array(tp_percentages)
        np.save(f'tps{DATASET}{PROGRESS}', tp_percentages_array)             
        np.save(f'tns{DATASET}{PROGRESS}', tn_percentages_array)
        return 0
    
    tn_percentages_array = np.array(tn_percentages)  
    tp_percentages_array = np.array(tp_percentages)
    np.save(f'tps{DATASET}{PROGRESS}', tp_percentages_array)             
    np.save(f'tns{DATASET}{PROGRESS}', tn_percentages_array)
    print('Process completed successfully')
    return 1


if __name__ == '__main__':
    app = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' inclui ArcFace iResNet100
    app.prepare(ctx_id=0, det_thresh=SIMILARITY_THRESHOLD)  # ctx_id=0 para CPU, ctx_id=1 para GPU
    current_dir = os.getcwd()
    images_dir = IMAGES_DIR
    noiseprints_dir = NOISEPRINT_DIR
    images_path = os.path.join(current_dir, images_dir)
    noiseprints_path = os.path.join(current_dir, noiseprints_dir)

    df = pd.read_csv(f'~/Documents/Unicamp/IC/HorusProjeto/HorusEthics/cameradetection/docs/clustering/reduced_clusterized{DATASET}{N_CLUSTERS}.csv')
    columns = ['kmeans', 'ids']
    simplified_df = df.filter(items=columns)
    grouped_df = simplified_df.groupby('kmeans')
    
    comparison = face_comparisons(grouped_df, app, current_dir, images_path, noiseprints_path)




        

