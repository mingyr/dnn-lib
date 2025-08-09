import jax 
from flax import nnx
from utils import LossRegression
from optimizer import *

def test():
    # X and Y data
    x_train = jax.numpy.array([[1.0], [2.0], [3.0]], dtype=jax.numpy.float32)
    y_train = jax.numpy.array([[1.0], [2.0], [3.0]], dtype=jax.numpy.float32)

    class Model(nnx.Module):
        def __init__(self, rngs=nnx.Rngs(0)):
            self._W = nnx.Param(jax.random.normal(rngs.params(), (1)))
            self._b = nnx.Param(jax.numpy.zeros((1)))
        def __call__(self, x):
            # Our hypothesis XW+b
            y = x * self._W + self._b

            return y

    # cost/loss function
    model = Model()

    loss_fn = LossRegression(model)

    learning_rate = 0.01

    # Minimize
    # optimizer = Adam(model, loss_fn)
    # optimizer = RMSProp(model, loss_fn)
    optimizer = SGD(model, loss_fn, 100000)

    # optimizer = RMSProp(lr_decay = False)
    loss, _ = optimizer(x_train, y_train)

    # Launch the graph in a session.
    print(loss)

if __name__ == '__main__':
    test()


