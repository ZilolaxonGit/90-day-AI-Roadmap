import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])  # 0 = not buy, 1 = buy

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(x_train, y_train)

# Predicting

print(model.predict_proba([[4.5]]))
print(model.predict([[4.5]]))

# Scoring
print(model.score(x_test, y_test))

z = np.linspace(0, 12, 100)
model_sigmoid = 1 / (1 + np.exp(-(model.coef_[0][0]*z + model.intercept_)))

plt.plot(z, model_sigmoid)
plt.xlabel("Hours spent on website")
plt.ylabel("Probability")
plt.show()