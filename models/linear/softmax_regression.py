from sklearn.datasets import load_iris
from models.model import model
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from loss.CrossEntropyLoss import CrossEntropyLoss
from optimizer.SGD import SGD
import sys
import random
class SoftMaxRegression(model):

    def __init__(self, loss, optimizer, num_iterations=30, early_stopping=True, batch_size=16, learning_rate_decay = 10):
        super(SoftMaxRegression, self).__init__()
        self.loss = loss
        self.num_iterations = num_iterations
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate_decay = learning_rate_decay

    def softmax(self, x):
        e_x = np.maximum(np.exp(x - np.max(x, axis=1)[:, np.newaxis]), 1e-12)
        return e_x / np.sum(e_x, axis=1)[:, np.newaxis]

    def fit(self, weights, x):
        return self.softmax(np.dot(x, weights.T))

    def train(self, x, y):
        """
        train function
        :param x:
        :param y:
        :return:
        """
        self.x_avg = np.average(x, axis=0)
        self.x_std = np.std(x, axis=0)
        x = (x - self.x_avg) / self.x_std
        x = np.insert(x, 0, values=1, axis=1)

        num_classes = np.max(y) + 1
        y = np.array(y).reshape(-1)
        y = np.eye(num_classes)[y]

        self.w = np.random.rand(num_classes,len(x[0]))

        # register optimizer
        self.optimizer.register_model(self.fit)
        self.optimizer.register_loss(self.loss)
        n_batch = int(np.ceil(len(x) / self.batch_size))
        best_err = sys.maxsize
        for i in range(self.num_iterations):
            # early stopping
            if self.early_stopping and self.optimizer.learning_rate < 1e-8:
                break
            randomize = list(range(len(x)))
            np.random.shuffle(randomize)
            x = x[randomize]
            y = y[randomize]
            # iterating for batch
            for j_batch in range(n_batch):
                end = j_batch*self.batch_size+self.batch_size

                if end > len(x):
                    end = len(x)
                x_batch = x[j_batch*self.batch_size : end]
                y_batch = y[j_batch*self.batch_size : end]
                self.w = self.optimizer.step(self.w, x_batch, y_batch)

            y_ = self.fit(self.w, x)
            err = self.loss.errors(y, y_)
            print('Epoch {} err={}'.format(i, err))
            if err < best_err - 0.05:
                best_err = err
            else:
                self.optimizer.learning_rate /= self.learning_rate_decay
                print('learning rate from {} to {}'.format(self.optimizer.learning_rate * 10,
                                                           self.optimizer.learning_rate))

    def predict(self, x):
        x = (x - self.x_avg) / self.x_std
        x = np.insert(x, 0, values=1, axis=1)
        return np.argmax(np.around(self.fit(self.w, x), 0).astype(int), axis=1)

if __name__ == '__main__':
    """ test code """
    data, label = load_iris(True)
    acc = 0
    for _ in range(1):
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.1, random_state = random.randint(0,50))
        loss = CrossEntropyLoss()
        optimizer = SGD(learning_rate=0.01)
        reg = SoftMaxRegression(loss=loss, optimizer=optimizer, batch_size=4, num_iterations=100)
        reg.train(X_train, y_train)
        y_ = reg.predict(X_test)
        acc += np.sum(y_test==y_)/len(y_)
        print(np.sum(y_test==y_)/len(y_))
    print(acc / 1)
