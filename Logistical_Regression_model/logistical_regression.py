import torch
import numpy as np


class LogisticRegression:
    def __init__(self, lr=1e-4, tol=1e-4, max_iter=1000, bias=True, random_state=None):
        self._lr = lr
        self._tol = tol
        self._max_iter = max_iter
        self._bias = bias
        self._seed = random_state
        self._dim = None
        self._activation = torch.sigmoid
        self._ep = 1e-32

    def _init_weight(self):
        assert self._dim is not None
        if self._seed:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)
        self._w = torch.randn(self._dim).requires_grad_()
        if self._bias:
            self._b = torch.zeros(1).requires_grad_()

    def _forward(self, x):
        prediction = x * self._w
        if self._bias:
            prediction += self._b
        prediction = torch.sum(prediction, 1)
        return self._activation(prediction)

    def _get_loss(self, prediction, labels):
        loss = torch.sum((labels * prediction + self._ep).log()) + \
               torch.sum(((1 - labels) * (1 - prediction) + self._ep).log())
        return -loss

    def _remove_grad(self):
        self._w.grad.data.zero_()
        if self._bias:
            self._b.grad.data.zero_()

    def _update_parameters(self):
        with torch.no_grad():
            self._w -= self._lr * self._w.grad
            if self._bias:
                self._b -= self._lr * self._b.grad

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0]
        self._dim = x_train.shape[1]
        self._init_weight()

        for _ in range(self._max_iter):
            prediction = self._forward(x_train)
            loss = self._get_loss(prediction, y_train)
            loss.backward()
            self._update_parameters()
            self._remove_grad()
            if loss.item() < self._tol:
                break

    def predict(self, x_test):
        assert x_test.shape[1] == self._dim
        with torch.no_grad():
            prediction = self._forward(x_test)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            prediction = prediction.int()
        return prediction
