from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

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
    noise_matrix = np.load('5camera_noises.npy')
    print(noise_matrix.shape)
    df = pd.DataFrame(noise_matrix)
    # optimize_kmeans(noise_matrix, 50)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df)
    df['kmeans'] = kmeans.labels_
    print(df.head())

    features = df.iloc[:, :-1]
    umap_2d = UMAP(n_components=2)
    proj_2d = umap_2d.fit_transform(features)
    plt.scatter(x=proj_2d[:, 0], y=proj_2d[:, 1], c=df.iloc[:, -1], cmap='Spectral')
    plt.title('2D UMAP Projection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
    


