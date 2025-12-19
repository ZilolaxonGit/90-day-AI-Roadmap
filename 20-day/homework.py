import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, )


kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x, y)

print("prediction for 7 is : ", kn.predict_proba([[7]]))

kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(x, y)
print("Second prediction for 7 is: ", kn.predict_proba([[7]]))


