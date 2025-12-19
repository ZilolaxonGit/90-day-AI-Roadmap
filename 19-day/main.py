import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Dataset: Hours studied vs Pass(1)/Fail(0)
x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

# Train & test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(np.array([[2],[5],[7]]))
print(predictions)

plt.figure(figsize=(10,6))
plot_tree(
    model, 
    feature_names=["Hours Studied"], 
    class_names=["Fail", "Pass"], 
    filled=True
)
plt.show()

