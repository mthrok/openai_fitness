import theano.tensor as T

from luchador.nn import BaseLayer
from luchador.nn import Tensor


class PReLU(BaseLayer):
    def __init__(self, alpha):
        super(PReLU, self).__init__(alpha=alpha)

    def build(self, input_tensor):
        alpha = self.args['alpha']
        x = input_tensor.get()  # Unwrap the variable
        output_tensor = T.switch(x < 0, alpha * x, x)
        return Tensor(output_tensor, shape=input_tensor.shape)
