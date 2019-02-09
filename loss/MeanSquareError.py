import autograd.numpy as np

class MeanSquareError(object):
    def __init__(self):
        pass

    def errors(self, y, y_):
        return np.mean( 0.5*(y - y_) ** 2 )