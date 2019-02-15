from autograd import elementwise_grad as egrad
from optimizer.base_optimizer import Optimizer

class NAG(Optimizer):

    def __init__(self, learning_rate, momentum):
        super(NAG, self).__init__(learning_rate)
        self.momentum = momentum
        self.v = 0

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def optimizer(self, parameters, input, target):
        return self.loss.errors(self.fit(parameters, input), target)

    def step(self, parameters, input, target):
        optimizer_grad_func = egrad(self.optimizer, 0)
        parameters_gard = optimizer_grad_func(parameters, input + self.v, target)
        self.v = self.momentum * self.v + self.learning_rate * parameters_gard
        parameters = parameters - self.v
        return parameters