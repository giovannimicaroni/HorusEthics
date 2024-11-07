import pandas as pd
import os
import face_recognition
import numpy as np

DATASET = 'WSD'
IMAGES_DIR = '/home/gimicaroni/Documents/Datasets/WSD_Dataset/train'
N_CLUSTERS = 20
PROGRESS = 14

if __name__ == '__main__':
    current_dir = os.getcwd()
    images_dir = IMAGES_DIR
    images_path = os.path.join(current_dir, images_dir)

    df = pd.read_csv(f'~/Documents/Unicamp/IC/HorusProjeto/HorusEthics/cameradetection/docs/clustering/reduced_clusterized{DATASET}{N_CLUSTERS}.csv')
    columns = ['kmeans', 'ids']
    simplified_df = df.filter(items=columns)
    grouped_df = simplified_df.groupby('kmeans')
    tp_percentages = []
    tn_percentages = []
    print(grouped_df['kmeans'].value_counts())

    try:
        for kmeans_value, group in grouped_df:
            if kmeans_value == 16:
                print(f'starting cluster {kmeans_value}...')
                total_p = 0
                total_n = 0
                true_positives = 0
                true_negatives = 0

                for idx, (index, row) in enumerate(group.iterrows()):
                    current_identity = f'/{row['ids'][:5]}'
                    current_image = face_recognition.load_image_file(images_dir + current_identity + f'/{row['ids']}.jpg')
                    current_image_encodings = face_recognition.face_encodings(current_image)
                    img_comparisons = 0
                    
                    for next_index, next_row in group.iloc[idx+1:].iterrows():

                        next_identity = f'/{next_row['ids'][:5]}'
                        next_image = face_recognition.load_image_file(images_dir + next_identity + f'/{next_row['ids']}.jpg')
                        next_image_encodings = face_recognition.face_encodings(next_image)

                        if current_image_encodings and next_image_encodings:
                            comparison_result = face_recognition.compare_faces([current_image_encodings[0]], next_image_encodings[0])
                            print(f'{current_identity} {next_identity}: {comparison_result}')
                            
                            if current_identity == next_identity:
                                total_p += 1
                                if comparison_result[0] == True:
                                    true_positives += 1
                            else:
                                total_n += 1
                                if comparison_result[0] == False:
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
        # np.save(f'tps{DATASET}{PROGRESS}', tp_percentages_array)             
        # np.save(f'tns{DATASET}{PROGRESS}', tn_percentages_array)




        

