from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
import torch

DATASET = 'WSD'
N_CLUSTERS = 20

def optimize_kmeans(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    fig=plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    noise_matrix = torch.load(f'featurematrix{DATASET}.pt')
    print(noise_matrix.shape)
    ids_matrix = pd.read_csv(f'identitiesmatrix{DATASET}.csv')
    df = pd.DataFrame(noise_matrix)
    df['ids'] = ids_matrix.iloc[:, -1]
    # optimize_kmeans(noise_matrix[:-1], 50)
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    kmeans.fit(df.iloc[:, :-1])
    df['kmeans'] = kmeans.labels_
    df.to_csv(f'clusterized{DATASET}{N_CLUSTERS}.csv')
    print(df.head())
    cluster_centers = kmeans.cluster_centers_

    # Convert the centers to a DataFrame for easier viewing if needed
    centers_df = pd.DataFrame(cluster_centers, columns=df.columns[:-2])
    centers_df.to_csv(f'clustercenters{DATASET}{N_CLUSTERS}.csv')

    # features = df.iloc[:, :-2]
    # umap_2d = UMAP(n_components=2)
    # proj_2d = umap_2d.fit_transform(features)
    # plt.scatter(x=proj_2d[:, 0], y=proj_2d[:, 1], c=df.iloc[:, -1], cmap='Spectral')
    # plt.title('2D UMAP Projection')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.show()
    


