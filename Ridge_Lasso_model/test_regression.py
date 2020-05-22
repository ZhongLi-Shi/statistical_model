from sklearn.datasets import load_boston
import sklearn.preprocessing
import torch
from Ridge_Lasso_model.Ridge_Lasso import Ridge, Lasso

x, y = load_boston(return_X_y=True)
x = sklearn.preprocessing.scale(x)

x_train, y_train = torch.from_numpy(x[:400]), torch.from_numpy(y[:400])
x_test, y_test = torch.from_numpy(x[400:]), torch.from_numpy(y[400:])

ridge = Ridge(alpha=100, lr=1e-3, max_iter=500, random_state=123)
lasso = Lasso(alpha=100, lr=1e-3, max_iter=500, random_state=123)

print('*****Ridge*****')
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
rmse = torch.sqrt(torch.sum((y_test - y_pred) ** 2)).item()
print('RMSE:', rmse)

print('*****Lasso*****')
lasso.fit(x_train, y_train)
y_pred = lasso.predict(x_test)
rmse = torch.sqrt(torch.sum((y_test - y_pred) ** 2)).item()
print('RMSE:', rmse)