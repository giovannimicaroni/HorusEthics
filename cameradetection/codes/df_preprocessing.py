import os
import numpy as np
import pandas as pd  # GPU-accelerated DataFrame

DATASET = 'WSD'
IMAGES_DIR = 'C:/IC/WSD_Dataset/train'
MAX_ID_PERCLUSTER = 25
N_CLUSTERS = 20

def compute_distance(row, cluster_center):
    return np.sqrt(np.sum((row[:'ids'] - cluster_center) ** 2))


if __name__ == '__main__':
    current_dir = os.getcwd()
    images_dir = IMAGES_DIR
    images_path = os.path.join(current_dir, images_dir)

    df = pd.read_csv(f'clusterized{DATASET}{N_CLUSTERS}.csv')
    df_clustercenters = pd.read_csv(f'clustercenters{DATASET}{N_CLUSTERS}.csv')
    df[['ID', 'Image_Number']] = df['ids'].str.split('_', expand=True)
    grouped_df = df.groupby('kmeans')
    print(f'before: {df["ID"].value_counts()}')
 
    updated_grouped = pd.DataFrame()
    for kmeans_value, group in grouped_df:
        n_images_per_id = group['ID'].value_counts()
        # print(f'before: {n_images_per_id}')
        for identity, frequency in n_images_per_id.items():
            if frequency > MAX_ID_PERCLUSTER:

                rows_with_identity = group[group['ID'] == identity]
                cluster_center = df_clustercenters.iloc[kmeans_value, :].values
                data_points = rows_with_identity.iloc[:, :-4].values
                distances = np.linalg.norm(data_points - cluster_center, axis=1)
                smallest_distance_indices = np.argsort(distances)[:10]
                closest_rows = rows_with_identity.iloc[smallest_distance_indices]
                group = group[~((group['ID'] == identity) & (~group.index.isin(closest_rows.index)))]

        n_images_per_id = group['ID'].value_counts()
        updated_grouped = pd.concat([updated_grouped, group])
        # print(f'after: {n_images_per_id}')

            # print(f'{identity} {frequency}')

    print(f'after: {updated_grouped["ID"].value_counts()}')
    print(updated_grouped.head())
    updated_grouped.to_csv(f'reduced_clusterized{DATASET}{N_CLUSTERS}.csv')
            
            
