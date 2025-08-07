import jax
from flax import nnx
import optax

class Adam(nnx.Module):
    def __init__(self, model, loss_fn, learning_rate=0.01, lr_decay=False, 
                 lr_decay_steps=None, lr_decay_factor=None, clip=False, clip_value=1.0):
        self._model = model
        self._loss_fn = loss_fn

        if lr_decay:
            assert lr_decay_steps > 0 and lr_decay_factor > 0, "Invalid learning rate factors"
            lr = optax.exponential_decay(init_value=learning_rate, 
                                         transition_steps=lr_decay_steps,
                                         decay_rate=lr_decay_factor)
        else:
            lr = learning_rate

        if clip:
            opt = optax.chain(optax.clip(clip_value), optax.adam(learning_rate=lr))
        else:
            opt = optax.adam(learning_rate=lr)

        self._optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    def __call__(self, inputs, labels):
        loss, grads = nnx.value_and_grad(self._loss_fn)(self._model, inputs, labels)
        self._optimizer.update(grads)

        return loss


class RMSProp(nnx.Module):
    def __init__(self, model, loss_fn, learning_rate=0.01, epsilon=1e-10, lr_decay=False,
                 lr_decay_steps=None, lr_decay_factor=None, clip=False, clip_value=1.0):
        self._model = model
        self._loss_fn = loss_fn

        if lr_decay:
            assert(lr_decay_steps > 0) and (lr_decay_factor > 0), "Invalid learning rate decay factors"
            lr = optax.exponential_decay(init_value=learning_rate,
                                         transition_steps=lr_decay_steps,
                                         decay_rate=lr_decay_factor)
        else:
            lr = learning_rate 

        if clip:
            opt = optax.chain(optax.clip(clip_value), optax.rmsprop(learning_rate=lr, eps=epsilon))
        else:
            opt = optax.rmsprop(learning_rate=lr, eps=epsilon)

        self._optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    def __call__(self, inputs, labels):
        loss, grads = nnx.value_and_grad(self._loss_fn)(self._model, inputs, labels)
        self._optimizer.update(grads)

        return loss


class SGD(nnx.Module):
    def __init__(self, model, loss_fn, num_iters, learning_rate=0.01, lr_decay = True):
        self._model = model
        self._loss_fn = loss_fn

        if lr_decay:
            lr = optax.schedules.linear_schedule(learning_rate, 1e-5, num_iters)
        else:
            lr = learning_rate

        self._optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=lr), wrt=nnx.Param)

    def __call__(self, inputs, labels):
        loss, grads = nnx.value_and_grad(self._loss_fn)(self._model, inputs, labels)
        self._optimizer.update(grads)

        return self._params, self._opt_state, loss_value


