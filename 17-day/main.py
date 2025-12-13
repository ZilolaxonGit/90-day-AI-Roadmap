import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt


# Data
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([50, 55, 65, 70, 80, 95])


# test and train
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42
)

# Create a model & give a trained data to the model

model = LinearRegression()
model.fit(x_train, y_train)

print("predicted score for 7 hours ", model.predict([[7]]))

# Testing the model accuracy
print(model.score(x_test, y_test))

# Visualization

plt.scatter(x, y)
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.show()



"""
If the model score is eqaul to 1, then the model's accuracy is perfect
If the model score is 0.5, then the model's accuracy is ok
0 = model is useless
<0 = model is trash


x => input
y => output

"""


