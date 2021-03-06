from autograd import elementwise_grad as egrad
from optimizer.base_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate):
        super(SGD, self).__init__(learning_rate)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def optimizer(self, parameters, input, target):
        return self.loss.errors(self.fit(parameters, input), target)

    def step(self, parameters, input, target):
        optimizer_grad_func = egrad(self.optimizer, 0)
        parameters_gard = optimizer_grad_func(parameters, input, target)
        parameters = parameters - self.learning_rate * parameters_gard
        return parameters
