from autograd import elementwise_grad as egrad

class SGD(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def register_model(self, fit):
        self.fit = fit

    def register_loss(self, loss):
        self.loss = loss

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def optimizer(self, parameters, input, target):
        return self.loss.errors(self.fit(parameters, input), target)

    def step(self, parameters, input, target):
        optimizer_grad_func = egrad(self.optimizer, 0)
        parameters_gard = optimizer_grad_func(parameters, input, target)
        parameters = parameters - self.learning_rate * parameters_gard
        return parameters
