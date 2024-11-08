import pandas as pd
import os
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np

DATASET = 'WSD'
IMAGES_DIR = '/home/gimicaroni/Documents/Datasets/WSD_Dataset/train'
N_CLUSTERS = 20
PROGRESS = 14
SIMILARITY_THRESHOLD = 0.5

def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
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

def face_comparisons(grouped_df, app, current_dir, images_path):
    tp_percentages = []
    tn_percentages = []
    print(grouped_df['kmeans'].value_counts())

    try:
        for kmeans_value, group in grouped_df:
            print(f'starting cluster {kmeans_value}...')
            total_p = 0
            total_n = 0
            true_positives = 0
            true_negatives = 0

            for idx, (index, row) in enumerate(group.iterrows()):
                current_identity = f'/{row['ids'][:5]}'
                current_image_encodings = get_embedding(images_dir + current_identity + f'/{row['ids']}.jpg')
                img_comparisons = 0
                
                for next_index, next_row in group.iloc[idx+1:].iterrows():

                    next_identity = f'/{next_row['ids'][:5]}'
                    next_image_encodings = get_embedding(images_dir + next_identity + f'/{next_row['ids']}.jpg')

                    if current_image_encodings != None and next_image_encodings != None:
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
                        

            tp_percentage = true_positives / total_p
            tn_percentage = true_negatives / total_n
            tp_percentages.append(tp_percentage)
            tn_percentages.append(tn_percentage)
            print(f'cluster {kmeans_value} done {tp_percentage}, {tn_percentage}')

    except KeyboardInterrupt:
        print("Process interrupted, saving progress...")
        tn_percentages_array = np.array(tn_percentages)  
        tp_percentages_array = np.array(tp_percentages)
        return 0
        # np.save(f'tps{DATASET}{PROGRESS}', tp_percentages_array)             
        # np.save(f'tns{DATASET}{PROGRESS}', tn_percentages_array)
    
    return 1


if __name__ == '__main__':
    app = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' inclui ArcFace iResNet100
    app.prepare(ctx_id=0, det_thresh=0.5)  # ctx_id=0 para CPU, ctx_id=1 para GPU
    current_dir = os.getcwd()
    images_dir = IMAGES_DIR
    images_path = os.path.join(current_dir, images_dir)

    df = pd.read_csv(f'~/Documents/Unicamp/IC/HorusProjeto/HorusEthics/cameradetection/docs/clustering/reduced_clusterized{DATASET}{N_CLUSTERS}.csv')
    columns = ['kmeans', 'ids']
    simplified_df = df.filter(items=columns)
    grouped_df = simplified_df.groupby('kmeans')
    
    comparison = face_comparisons(grouped_df, app, current_dir, images_path)




        

