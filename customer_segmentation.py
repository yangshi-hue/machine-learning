import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv("customer_data.csv")
X = data[["annual_income","spending_score"]]
#clustering the data set
model = KMeans(n_clusters = 3, random_state = 42)
#training the dataset
model.fit(X)
#get the clusters label
labels = model.labels_
print(labels)