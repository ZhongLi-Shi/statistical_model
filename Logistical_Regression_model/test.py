from sklearn.datasets import load_breast_cancer
import sklearn.preprocessing
from Logistical_Regression_model.logistical_regression import LogisticRegression
import torch

x, y = load_breast_cancer(return_X_y=True)
x = sklearn.preprocessing.scale(x)

x_train, y_train = torch.from_numpy(x[:400]), torch.from_numpy(y[:400])
x_test, y_test = torch.from_numpy(x[400:]), torch.from_numpy(y[400:])

LR = LogisticRegression(random_state=123, max_iter=5000)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
print((y_test == y_pred).sum().item() / len(y_pred))
