import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([[1],[3],[5],[9],[10],[7]])
y = np.array([3,2,5,8,9,2]) #creating a sample data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state= 42
) #training data

model = LinearRegression ()#creating a mode;
#train the model
model.fit(X_train, y_train)

#making predictions
predictions = model.predict(X_test)

#evalaute the model
mse = mean_squared_error(y_test, predictions)
print("Predictions :", predictions)
print("Mean Sqaured Error : ", mse)

#visualize
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()