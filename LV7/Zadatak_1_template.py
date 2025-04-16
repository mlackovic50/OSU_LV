import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers=4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

plt.figure(figsize=(15, 10))
for i in range(1, 6):
    X = generate_data(500, i)
    plt.subplot(2, 3, i)
    plt.scatter(X[:,0], X[:,1], s=10)
    plt.title(f'flagc={i}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
plt.suptitle('Generirani podaci s različitim flagc vrijednostima', y=1.02)
plt.tight_layout()
plt.show()

X = generate_data(500, 1)
k_values = [2, 3, 4, 5, 6]

plt.figure(figsize=(15, 8))
for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    
    plt.subplot(2, 3, i+1)
    plt.scatter(X[:,0], X[:,1], c=labels, s=10)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
               c='red', marker='x', s=100)
    plt.title(f'K={k}, Silhouette={silhouette:.2f}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
plt.suptitle('KMeans s različitim K vrijednostima (flagc=1)', y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 12))
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    
    optimal_k = 3 if flagc in [1,2] else 4 if flagc == 3 else 2
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    
    plt.subplot(2, 3, flagc)
    plt.scatter(X[:,0], X[:,1], c=labels, s=10)
    if flagc in [1,2,3]:
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                   c='red', marker='x', s=100)
    plt.title(f'flagc={flagc}, K={optimal_k}\nSilhouette={silhouette:.2f}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
plt.suptitle('KMeans s optimalnim K za različite načine generiranja podataka', y=1.02)
plt.tight_layout()
plt.show()