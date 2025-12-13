import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.array([1,2,3,4,5, 10]).reshape(-1, 1)
y = np.array([10, 27, 30, 40, 60, 100])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

model = LinearRegression()

model.fit(x_train, y_train)

print(model.predict([[6]]))

print(model.score(x_test, y_test))