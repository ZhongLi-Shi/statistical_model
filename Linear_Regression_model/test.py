from sklearn.datasets import load_boston
import sklearn.preprocessing
from Linear_Regression_model.linear_regression import LinearRegression
import torch

x, y = load_boston(return_X_y=True)
x = sklearn.preprocessing.scale(x)

x_train, y_train = torch.from_numpy(x[:400]), torch.from_numpy(y[:400])
x_test, y_test = torch.from_numpy(x[400:]), torch.from_numpy(y[400:])

LR = LinearRegression(lr=1e-3, max_iter=500, random_state=123)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
rmse = torch.sqrt(torch.sum((y_test - y_pred) ** 2)).item()
print('RMSE:', rmse)
