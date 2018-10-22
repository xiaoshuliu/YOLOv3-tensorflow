import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from config import path, Input_shape
import numpy as np
from PIL import Image

annotation_path_train = path + '/model/bdd_train.txt'

def test_of_k_means():
    X = []
    with open(annotation_path_train) as f:
        GG = f.readlines()
        count = 0
        print(len(GG))
        for line in (GG):
            line = line.split(' ')
            filename = line[0]

            if filename[-1] == '\n':
                filename = filename[:-1]
            image = Image.open(filename)
            image_w, image_h = image.size
            if len(line)==1:
                continue
            for i, box in enumerate(line[1:]):
                box = list(map(int, box.split(',')))
                X.append([(box[2] - box[0])*Input_shape/image_w, (box[3] - box[1])*Input_shape/image_h])

    X = np.array(X)
    print(X.shape)
    kmeans = KMeans(n_clusters=9)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # 颜色范围viridis: https://matplotlib.org/examples/color/colormaps_reference.html
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=4, cmap='viridis')

    centers = kmeans.cluster_centers_  # 聚类的中心
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=40, alpha=0.5)  # 中心点为黑色
    print(centers)
    plt.show()  # 展示


if __name__ == '__main__':
    test_of_k_means()
