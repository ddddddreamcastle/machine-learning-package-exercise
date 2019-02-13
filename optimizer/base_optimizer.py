class Optimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def register_model(self, fit):
        self.fit = fit

    def register_loss(self, loss):
        self.loss = loss