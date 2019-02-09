import torch
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston


def linear_model(x):
    return torch.mm(x, torch.reshape(w, (-1,1))) + b

def get_loss(y_, y):
    return torch.mean((y_ - y) ** 2)


if __name__ == '__main__':
    data, label = load_boston(True)

    data_avg = np.average(data, axis=0)
    data_std = np.std(data, axis=0)
    data_l2_norm = np.sqrt(np.einsum('ij,ij->i', data, data))
    label_avg = np.average(label)
    label_std = np.std(label)
    label_l2_norm = np.sqrt(np.einsum('ij,ij->i', label, label))

    data = (data - data_avg) / data_l2_norm[:, np.newaxis]
    label = (label - label_avg) / label_l2_norm[:, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    x_train = Variable(torch.from_numpy(X_train)).double()
    y_train = Variable(torch.from_numpy(y_train)).double()

    w = Variable(torch.randn(len(X_train[0])).double(), requires_grad=True)
    b = Variable(torch.zeros(1).double(), requires_grad=True)

    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)
    loss.backward()
    w.grad.zero_()
    b.grad.zero_()
    for e in range(100):
        y_ = linear_model(x_train)
        loss = get_loss(y_, y_train)
        w.grad.zero_()
        b.grad.zero_()
        loss.backward()
        w.data = w.data - 1e-2 * w.grad.data
        b.data = b.data - 1e-2 * b.grad.data

        print('epoch: {}, loss: {}'.format(e, loss.data[0]))