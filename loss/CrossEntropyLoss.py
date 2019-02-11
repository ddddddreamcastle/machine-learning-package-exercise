import autograd.numpy as np

class CrossEntropyLoss(object):
    def __init__(self):
        pass

    def errors(self, y, y_):
        return -np.mean( y * np.log(np.maximum(y_, 1e-12)) + (1 - y) * np.log(np.maximum(1 - y_, 1e-12)) )