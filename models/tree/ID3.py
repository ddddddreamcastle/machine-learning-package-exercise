from sklearn.datasets import load_boston
from models.model import model
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from loss.MeanSquareError import MeanSquareError
from optimizer.SGD import SGD
from optimizer.NAG import NAG
from optimizer.Momentum import Momentum
import sys
from preprocessing.data import standardization as data_standardization
from preprocessing.label import standardization as label_standardization
class ID3(model):

    def __init__(self, ):
        super(ID3, self).__init__()
        pass

    def fit(self, weights, x):
        """ linear regression function """
        pass

    def train(self, x, y):
        """
        train function
        :param x:
        :param y:
        :return:
        """
        pass

    def predict(self, x):
        pass

if __name__ == '__main__':
    """ test code """
    pass
