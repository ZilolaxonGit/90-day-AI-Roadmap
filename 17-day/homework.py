import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# House sizes (in sqft) and prices (in thousands of dollars)
sizes = np.array([[500], [700], [800], [1000], [1200], [1500], [1800], [2000]])
prices = np.array([100, 150, 170, 200, 240, 300, 350, 400])


# Train & test

x_train, x_test, y_train, y_test = train_test_split(
    sizes, prices, train_size=0.3, random_state=42
)

# Create a model

model = LinearRegression()
model.fit(x_train, y_train)

# predict
print("Predicting for 1100 sq ", model.predict([[1100]]))
print("Predicting for 1300 sq ", model.predict([[1300]]))
print("Predicting for 2500 sq ", model.predict([[2500]]))

print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

"""
Slope defines in each square meter how much money is going to be added
Intercept defines the minimum value of the house price
"""

# Accuracy
print("Score of your model is ", model.score(x_test, y_test))

# Visual
import matplotlib.pyplot as plt

# Plotting the actual data points
plt.scatter(sizes, prices, color="blue", label="Actual Data")

# Plotting the regression line
plt.plot(sizes, model.predict(sizes), color="red", label="Regression Line")

# Analyzing errors
errors = y_test - model.predict(x_test)

plt.bar(range(len(errors)), errors)
plt.title("Prediction Errors")
plt.xlabel("Index")
plt.ylabel("Error")
plt.show()



# plt.xlabel("House Size (sqft)")
# plt.ylabel("Price (in thousands)")
# plt.legend()
# plt.show()
