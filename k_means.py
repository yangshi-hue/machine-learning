import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [20, 200],
    [22, 220],
    [25, 250],
    [40, 700],
    [42, 720],
    [45, 750],
    [60, 1200],
    [62, 1250],
    [65, 1300]
])

model = KMeans(n_clusters=3, random_state=42)

model.fit(X)

labels = model.labels_

print(labels)
print(model.cluster_centers_) #finding center of cluster

#visualizing the data
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Age")
plt.ylabel("Spending")
plt.title("Customer Clusters")

plt.show()