import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

if __name__ == "__main__":
    true_positives = np.load('docs/TP_WSD.npy')
    true_negatives = np.load('docs/TN_WSD.npy')
    cluster_pos = np.arange(len(true_negatives))

    style.use('ggplot')
    barWidth = 0.35
    plt.bar(cluster_pos, true_positives, color='royalblue', width=barWidth, label='TP')
    plt.bar(cluster_pos+0.35, true_negatives, color='red', width=barWidth, label='TN')
    plt.xlabel('Clusters')
    plt.ylabel('Percentage')
    plt.title('Dataset WSD - Face Recognition')
    plt.xticks(cluster_pos + barWidth / 2, [f'{i+1}' for i in cluster_pos])
    plt.show()