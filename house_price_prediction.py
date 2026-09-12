import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#loading the csv file
data = pd.read_csv("housing.csv")
X = data[["size"]]#house size
y = data["price"]#house price

#training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state= 42
)
#training a model 
model = LinearRegression()
model.fit(X_train, y_train) #learn the relationship between house and price

#make a prediction
prediction = model.predict(X_test)

#finding the 2r2_Score
r2 = r2_score(y_test, prediction)
print("R2 score : " , r2)
#visualizing the data
plt.scatter(X, y, marker = "*", color = "red")
plt.plot(X, model.predict(X), color="blue")
plt.xlabel("house size")
plt.ylabel("house price")
plt.title("House size vs house price")
plt.show()

