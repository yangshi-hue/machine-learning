import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv("customer_data.csv")
X = data[["annual_income","spending_score"]]
#clustering the data set
model = KMeans(n_clusters = 3, random_state = 42)
#training the dataset
model.fit(X)
#get the clusters label
labels = model.labels_
print(labels)
#visualizing
plt.scatter(
    X["annual_income"],
    X["spending_score"],
    c = labels
)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("customer segmentation")

plt.show()