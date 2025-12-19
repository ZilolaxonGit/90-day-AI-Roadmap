import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data: x = hours, y = scores
X = np.array([[1, 2],
              [1, 4],
              [1, 0],
              [10, 2],
              [10, 4],
              [10, 0]])

# Create KMeans model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Get cluster labels for each point
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Labels:", labels)
print("Centroids:", centroids)

# Visualize
plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
# plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='black')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
