from sklearn.cluster import KMeans
import numpy as np

# já ter extraido os embeddings
# all_embeddings = np.array(all_embeddings) 

# número de clusters 
n_clusters = 15 

# clusterização
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(all_embeddings)

# resultados
print("Clusters atribuídos para cada embedding:", clusters)
