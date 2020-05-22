from sklearn.datasets import load_breast_cancer
import sklearn.preprocessing
from Ridge_Lasso_model.Ridge_Lasso import RidgeClassifier, LassoClassifier
import torch


x, y = load_breast_cancer(return_X_y=True)
x = sklearn.preprocessing.scale(x)

x_train, y_train = torch.from_numpy(x[:400]), torch.from_numpy(y[:400])
x_test, y_test = torch.from_numpy(x[400:]), torch.from_numpy(y[400:])

ridge = RidgeClassifier(alpha=10, lr=1e-4, random_state=123, max_iter=5000)
lasso = LassoClassifier(alpha=10, lr=1e-4, random_state=123, max_iter=5000)

print('*****Ridge*****')
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
print((y_test == y_pred).sum().item() / len(y_pred))

print('*****Lasso*****')
lasso.fit(x_train, y_train)
y_pred = lasso.predict(x_test)
print((y_test == y_pred).sum().item() / len(y_pred))

