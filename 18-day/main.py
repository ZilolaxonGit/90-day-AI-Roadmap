"""

1️⃣ What is Logistic Regression?

Despite the name, Logistic Regression is NOT used for predicting numbers.

Linear Regression → predicts continuous values (price, score, temperature)

Logistic Regression → predicts categories (yes/no, spam/not spam, sick/healthy)

It is used for binary classification: two possible outcomes.


Logistic vs Linear regression

linear regression defines the continuous I mean next values wheareas logistic regression categorizes the given data like spam not spam, true of false


# Predict probability
print(model.predict_proba([[2]])) => This code returns two values,
    first one defines how much does this belong to class 0
    second one defines how much does this belong to class 1



"""

import numpy as np
from sklearn.linear_model import LogisticRegression

# Features: hours studied
X = np.array([[2],[4],[6],[8]])
# Labels: 0 = fail, 1 = pass
y = np.array([0,0,1,1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict probability
print(model.predict_proba([[2]]))  # probability for fail/pass

# Predict class
print(model.predict([[5]]))        # class 0 or 1



# import matplotlib.pyplot as plt

# z = np.linspace(-10, 10, 100)
# sigmoid = 1 / (1 + np.exp(-z))

# plt.plot(z, sigmoid)
# plt.xlabel("Linear combination (z)")
# plt.ylabel("Probability")
# plt.title("Sigmoid Function")
# plt.grid(True)
# plt.show()

