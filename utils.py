import os, sys
import math
import jax
import jax.numpy as jnp
import chex
import flax
from flax import nnx
import optax
import orbax.checkpoint as ocp
from pathlib import Path
from functools import partial

class JAXVer():
    def __init__(self, jax=None):
        assert jax
        ver_str = jax.__version_info__

        assert len(ver_str) >= 2
        self._major = int(ver_str[0])
        self._minor = int(ver_str[1])

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor


def get_batch_size(inputs, time_major=False):
    if time_major:
        if "Tensor" in str(type(inputs)):
            batch_size = inputs.get_shape().with_rank_at_least(3)[1]
        else:
            assert len(inputs.shape) >= 3, "Insufficient dimensions: {}".format(inputs.shape)
            batch_size = inputs.shape[1]
    else:
        if "Tensor" in str(type(inputs)):
            batch_size = inputs.get_shape().with_rank_at_least(3)[0]
        else:
            assert len(inputs.shape) >= 3, "Insufficient dimensions: {}".format(inputs.shape)
            batch_size = inputs.shape[0]

    return batch_size


class Activation:
    def __init__(self, act=None, verbose=False):
        if act == 'sigmoid':
            if verbose: print('Activation function: sigmoid')
            self._act = jax.nn.sigmoid
        elif act == 'tanh':
            if verbose: print('Activation function: tanh')
            self._act = jax.nn.tanh
        elif act == 'relu':
            if verbose: print('Activation function: relu')
            self._act = jax.nn.relu
        elif act == 'elu':
            if verbose: print('Activation function: elu')
            self._act = jax.nn.elu
        elif act == 'swish':
            if verbose: print('Activation function: swish')
            self._act = jax.nn.swish
        else:
            if verbose: print('Activation function: identity')
            self._act = jax.nn.identity

    def __call__(self, x):
        return self._act(x)


class AvgPool():
    def __init__(self, k, padding):
        self._func = partial(flax.nnx.avg_pool, window_shape=k, strides=k, padding=padding)

    def __call__(self, inputs):
        return self._func(inputs)

class MaxPool():
    def __init__(self, k, padding):
        self._func = partial(flax.nnx.max_pool, window_shape=k, strides=k, padding=padding)

    def __call__(self, inputs):
        return self._func(inputs)

class Pooling():
    def __init__(self, pool, k, padding='VALID', verbose=False):
        if pool == 'max':
            if verbose: print('Pooling method: maximal')
            self._pool = MaxPool(k, padding)
        elif pool == 'avg':
            if verbose: print('Pooling method: averaged')
            self._pool = AvgPool(k, padding)
        else:
            if verbose: print('Pooling method: trivial')
            self._pool = lambda x: x

    def __call__(self, x):
        return self._pool(x)


class Dropout():
    def __init__(self, keep_prob, is_training, rngs=nnx.Rngs(0)):
        self._is_training = is_training
        self._func = flax.nn.Dropout(1-keep_prob, rngs=rngs)

    def __call__(self, inputs):
        if self._is_training:
            return self._func(inputs)
        else:
            return jax.nn.identity(inputs)


class LossRegression():
    def __init__(self, sanity_check=False):
        self._sanity_check = sanity_check

    def __call__(self, model, inputs, labels):
        logits = model(inputs)
        if self._sanity_check:
            chex.assert_equal_shape([logits, labels])
        
        loss = jnp.mean((logits - labels) ** 2)
            
        return loss


class LossClassification():
    def __init__(self, num_classes, sanity_check=False):
        assert(num_classes > 1), "Invalid number of classes"
        self._num_classes = num_classes 
        self._sanity_check = sanity_check

    def __call__(self, model, inputs, labels):
        logits = model(inputs)
        labels = jax.nn.one_hot(labels, self._num_classes)
        if self._sanity_check:
            chex.assert_equal_shape([logits, labels])

        loss = optax.safe_softmax_cross_entropy(logits, labels)

        return loss.mean()


class ValRegression():
    def __init__(self, sanity_check=False):
        self._model = model
        self._sanity_check = sanity_check

    def __call__(self, model, inputs, labels):
        logits = model(inputs)
        if self._sanity_check:
            chex.assert_equal_shape([logits, labels])
        
        loss = jnp.mean((logits - labels) ** 2)

        return jnp.sqrt(loss)


class ValClassification():
    def __init__(self, sanity_check=False):
        self._model = model
        self._sanity_check = sanity_check

    def __call__(self, inputs, labels):
        logits = model(inputs)
        logits = jnp.argmax(logits, -1)

        if self._sanity_check:
            chex.assert_equal_shape([logits, labels])
        
        prediction = jnp.equal(labels, logits)
        accuracy = jnp.mean(prediction)

        return accuracy


class TestRegression():
    def __init__(self, sanity_check=False):
        self._sanity_check = sanity_check

    def __call__(self, model, inputs, labels):
        logits = model(inputs)
        if self._sanity_check:
            chex.assert_equal_shape([logits, labels])
        
        loss = jnp.mean((logits - labels) ** 2)

        return jnp.sqrt(loss)


class TestClassification():
    def __init__(self, sanity_check=False):
        self._sanity_check = sanity_check

    def __call__(self, model, inputs, labels):
        logits = model(inputs)
        logits = jnp.argmax(logits, -1)
        if self._sanity_check:
            chex.assert_equal_shape([logits, labels])

        prediction = jnp.equal(labels, logits)
        accuracy = jnp.mean(prediction)

        return accuracy


class Summaries():
    def __init__(self):
        self._summaries = {'train':[], 'val':[], 'test':[], 'stat':[], 'debug':[]}

    def __call__(self, category, value):
        self._summaries[category].append(value)

    @property
    def train(self):
        return self._summaries['train']

    @property
    def val(self):
        return self._summaries['val']

    @property
    def test(self):
        return self._summaries['test']

    @property
    def stat(self):
        return self._summaries['stat']

    @property
    def debug(self):
        return self._summaries['debug']

    def dump(self):
        print("train: ", end='\t')
        for l in self.train:
            print(l, end=' ')
        print('')

        print("val: ", end='\t')
        for l in self.val:
            print(l, end=' ')
        print('')

        print("test: ", end='\t')
        for l in self.test:
            print(l, end=' ')
        print('')

        print("stat: ", end='\t')
        for l in self.stat:
            print(l, end=' ')
        print('')

        print("debug: ", end='\t')
        for l in self.debug:
            print(l, end=' ')
        print('')


class Checkpoint():
    def __init__(self, path, to_save=True):
        if to_save:
            if not os.path.exists(path):
                os.makedirs(path)
            self._ckpt_dir = ocp.test_utils.erase_and_create_empty(Path(path).resolve())
        else:
            assert os.path.exists(path)
            self._ckpt_dir = Path(path).resolve()

    def save(self, pre_state, post_state):
        pre_state_dict = nnx.traversals.flatten_mapping(pre_state)
        post_state_dict = nnx.traversals.flatten_mapping(post_state)
        keys = pre_state_dict.keys() & post_state_dict.keys()
        state_dict = {key: post_state_dict[key] for key in keys}
        state = nnx.traversals.unflatten_mapping(state_dict)
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        checkpointer.save(self._ckpt_dir / 'state', args=ocp.args.StandardSave(state))

    def load(self, model):
        graphdef, state = nnx.split(model)
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        state_restored = checkpointer.restore(self._ckpt_dir / 'state', state)
        nnx.update(model, state_restored)


class DownsampleAlongH():
    """
    Downsampling by taking every n by n entries along H (rows)
    """
    def __init__(self, factor, padding='SAME', verbose=False):
        self._factor = factor
        self._padding = padding

        if verbose:
            print('Downsample - factor: {}, padding: {}'.format(factor, padding))

    def __call__(self, inputs):
        return nnx.max_pool(inputs, [self._factor, 1], [self._factor, 1], self._padding)


class DownsampleAlongW():
    """
    Downsampling by taking every n by n entries along H (rows)
    """
    def __init__(self, factor, padding='SAME', verbose=False):
        self._factor = factor
        self._ptype = ptype
        self._padding = padding

        if verbose:
            print('Downsample - factor: {}, padding: {}'.format(factor, padding))

    def __call__(self, inputs):
        return nnx.max_pool(inputs, [1, self._factor], [1, self._factor], self._padding)


class Pad2D():
    def __init__(self, pad, mode='CONSTANT', data_format='NHWC'):
        self._pad = pad
        self._mode = mode
        self._data_format = data_format
        
    def __call__(self, inputs):
        # Padding shape.
        if self._data_format == 'NHWC':
            paddings = [[0, 0], [0, self._pad[0]], [0, self._pad[1]], [0, 0]]
        elif self._data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [0, self._pad[0]], [0, self._pad[1]]]
        else:
            raise ValueError("Unknown data format: {}".format(self._data_format))
            
        return jnp.pad(inputs, paddings, mode=self._mode)


class DepthwiseConv(nnx.Module):
    def __init__(self, in_features, kernel_size, strides=1, padding='SAME', use_bias=True, rngs=nnx.Rngs(0)):
        self._in_features = in_features
        self._convs = [nnx.Conv(in_features=1, out_features=1, kernel_size=kernel_size, padding=padding, use_bias=use_bias, rngs=rngs)
                       for _ in range(in_features)]

    def __call__(self, inputs):
        inputs = jnp.split(inputs, self._in_features, axis=-1)
        outputs = jnp.concat([c(i) for c, i in zip(self._convs, inputs)], axis=-1)

        return outputs


class SeparableConv(nnx.Module):
    def __init__(self, in_features, out_features, kernel_size, strides=1, padding='SAME', use_bias=True, rngs=nnx.Rngs(0)):
        self._in_features = in_features
        self._convs = [nnx.Conv(in_features=1, out_features=1, kernel_size=kernel_size, padding=padding, rngs=rngs)
                       for _ in range(in_features)]
        self._conv = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=[1] * len(kernel_size), padding=padding, use_bias=use_bias, rngs=rngs)

    def __call__(self, inputs):
        inputs = jnp.split(inputs, self._in_features, axis=-1)

        outputs = jnp.concat([c(i) for c, i in zip(self._convs, inputs)], axis=-1)
        outputs = self._conv(outputs)

        return outputs


class Sequential():
    def __init__(self, *modules):
        self._mods = modules

    def __call__(self, inputs):
        outputs = inputs
        for m in self._mods:
            outputs = m(outputs)

        return outputs


