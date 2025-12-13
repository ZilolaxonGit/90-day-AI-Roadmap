diagram = """
MACHINE LEARNING
│
├── 1. What is ML?
│   └── Algorithm learns patterns from data
│
├── 2. Types of ML
│   ├── Supervised
│   │     ├── Regression
│   │     └── Classification
│   ├── Unsupervised
│   │     ├── Clustering
│   │     └── Dimensionality Reduction
│   └── Reinforcement
│         └── Agent learns by rewards
│
└── 3. Core ML Concepts
      ├── Features (X)
      └── Labels (y)


"""

print(diagram)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1 — Dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "score": [20, 30, 35, 40, 50, 60, 65, 70, 85]
}

df = pd.DataFrame(data)

# 2 — Split X and y
X = df[['hours']]
y = df['score']

# 3 — Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4 — Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5 — Predict
prediction = model.predict([[5]])
print("Prediction for 7.5 hours:", prediction)

# 6 — Accuracy
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)
