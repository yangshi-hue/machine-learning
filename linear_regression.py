import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1],[3],[5],[9],[10],[7],[20],[19],[45],[54]])
y = np.array([3,2,5,8,9,2,10,23,1,9]) #creating a sample data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state= 42
) #training data

model = LinearRegression ()#creating a mode;
#train the model
model.fit(X_train, y_train)
# Display the regression equation values
slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

#making predictions
predictions = model.predict(X_test)
#evalaute the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Predictions :", predictions)
print("Mean Sqaured Error : ", mse)
print("R2_score : ", r2)
#compare actual and predicted values
print("\nActual vs Predicted Vaues:")

for actual, predicted in zip(y_test, predictions):
    print(f"Actual :{actual} | Predicted:{predicted:.2f}")


#visualize
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()