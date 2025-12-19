import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# House size (sqft)
x = np.array([[500], [800], [1000], [1200], [1500], [1800], [2000]])

# House price (in $1000)
y = np.array([120, 180, 200, 240, 300, 340, 380])


# Train / Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Model

model = LinearRegression()
model.fit(x_train, y_train)

# Prediction
predict_example = [[6]]

print("Prediction for 6 is: ", model.predict(predict_example))

# Evaluate

print("Evaluation: ", model.score(x_test, y_test))


# plt.scatter(x, y, color='blue', label='Actual dots')
# plt.plot(x, model.predict(x), color='red', label='X dots|Regression')
# plt.xlabel("House Size (sqft)")
# plt.ylabel("Price ($1000)")
# plt.show()

# Errors

y_predictions = model.predict(x)

errors = y - y_predictions
print(errors)


# plt.figure(figsize=(6,4))
# plt.scatter(x, errors)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel("House Size (sqft)")
# plt.ylabel("Prediction Error (Actual - Predicted)")
# plt.title("Residuals vs House Size")
# plt.show()

mean_of_errors = np.mean(errors)

print(mean_of_errors)

mae = np.mean(np.abs(errors))
print("MAE:", mae)

rmse = np.sqrt(np.mean(errors**2))
print("RMSE:", rmse)
