import os, sys
import math
import numpy as np
import tensorflow as tf
import sonnet as snt
import pdb
from tensorflow.python.layers import normalization

class TFVer(object):
    def __init__(self, tf = None):
        assert tf
        ver_str = tf.__version__
        ver_str = ver_str.split('.')

        assert len(ver_str) >= 2

        self._major = int(ver_str[0])
        self._minor = int(ver_str[1])

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

def get_batch_size(inputs, time_major = False):
    if time_major:
        batch_size = inputs.get_shape().with_rank_at_least(3)[1]
    else:
        batch_size = inputs.get_shape().with_rank_at_least(3)[0]

    return batch_size

class Activation:
    def __init__(self, act = None, verbose = False):
        if act == 'sigmoid':
            if verbose: print('Activation function: sigmoid')
            self._act = tf.sigmoid
        elif act == 'tanh':
            if verbose: print('Activation function: tanh')
            self._act = tf.tanh
        elif act == 'relu':
            if verbose: print('Activation function: relu')
            self._act = tf.nn.relu
        elif act == 'elu':
            if verbose: print('Activation function: elu')
            self._act = tf.nn.elu
        elif act == 'swish':
            if verbose: print('Activation function: swish')
            self._act = lambda x: x * tf.sigmoid(x)
        else:
            if verbose: print('Activation function: identity')
            self._act = lambda x: tf.identity(x)

    def __call__(self, x):
        return self._act(x)

class AvgPool(snt.AbstractModule):
    def __init__(self, k = 2, padding = 'SAME', name = "avg_pool"):
        super(AvgPool, self).__init__(name = name)
        self._k = k
        self._padding = padding

    def _build(self, inputs):
        return tf.nn.avg_pool2d(inputs, self._k, self._k, self._padding)

class MaxPool(snt.AbstractModule):
    def __init__(self, k = 2, padding = 'SAME', name = "max_pool"):
        super(MaxPool, self).__init__(name = name)
        self._k = k
        self._padding = padding

    def _build(self, inputs):
        return tf.nn.max_pool2d(inputs, self._k, self._k, self._padding)

class Pooling:
    def __init__(self, pool = None, k = 2, padding = 'SAME', verbose = False):
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
 
class Dropout(snt.AbstractModule):
    def __init__(self, keep_prob, is_training, seed = None, name = "dropout"):
        super(Dropout, self).__init__(name = name)
        self._keep_prob = keep_prob
        self._is_training = is_training
        self._seed = seed

    def _build(self, inputs):
        if self._is_training:
            return tf.identity(inputs)
        else:
            return tf.nn.dropout(inputs, keep_prob = (1 - rate), seed = self._seed, rate = self._rate)

class LossRegression(snt.AbstractModule):
    def __init__(self, sanity_check = False, name = "loss_regression"):
        super(LossRegression, self).__init__(name = name)
        self._sanity_check = sanity_check

    def _build(self, logits, labels):
        if self._sanity_check:
            loss = tf.reduce_mean(tf.squared_difference(logits, labels), name = 'loss')
        else:
            with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
                loss = tf.reduce_mean(tf.squared_difference(logits, labels), name = 'loss')
            
        return loss

class LossClassification(snt.AbstractModule):
    def __init__(self, num_classes, sanity_check = False, name = 'loss_classification'):
        super(LossClassification, self).__init__(name = name)
        assert(num_classes > 1), "invalid number of classes"
        self._num_classes = num_classes 
        self._sanity_check = sanity_check

    def _build(self, logits, labels):
        labels = tf.one_hot(labels, self._num_classes)
        if self._sanity_check:
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits)
        else:
            with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits)

        loss = tf.reduce_mean(loss, name = 'loss')

        return loss

class ValRegression(snt.AbstractModule):
    def __init__(self, sanity_check = False, name = "val_regression"):
        super(ValRegression, self).__init__(name = name)
        self._sanity_check = sanity_check

    def _build(self, logits, labels):
        if self._sanity_check:
            loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits, labels)), name = 'val_deviation')
        else:
            with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
                loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits, labels)), name = 'val_deviation')

        return loss

class ValClassification(snt.AbstractModule):
    def __init__(self, sanity_check = False, name = "val_classification"):
        super(ValClassification, self).__init__(name = name)
        self._sanity_check = sanity_check

    def _build(self, logits, labels):
        logits = tf.argmax(logits, -1)

        if self._sanity_check:
            prediction = tf.equal(labels, logits)
        else:
            with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
                prediction = tf.equal(labels, logits)

        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return accuracy

class TestRegression(snt.AbstractModule):
    def __init__(self, sanity_check = False, name = "test_regression"):
        super(TestRegression, self).__init__(name = name)
        self._sanity_check = sanity_check

    def _build(self, logits, labels):
        if self._sanity_check:
            loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits, labels)), name = 'test_deviation')
        else:
            with tf.control_dependencies([tf.assert_equal(tf.rank(logits), tf.rank(labels))]):
                loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits, labels)), name = 'test_deviation')

        return loss

class TestClassification(snt.AbstractModule):
    def __init__(self, sanity_check = False, name = "test_classification"):
        super(TestClassification, self).__init__(name = name)
        self._sanity_check = sanity_check

    def _build(self, logits, labels):
        logits = tf.argmax(logits, -1)
        if self._sanity_check:
            prediction = tf.equal(labels, logits)
        else:
            with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
                prediction = tf.equal(labels, logits)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return accuracy

class Metrics(snt.AbstractModule):
    def __init__(self, indicator, name = "metrics"):
        super(Metrics, self).__init__(name = name)
        self._indicator = indicator

    def _build(self, labels, logits):
        labels = tf.cast(labels, tf.int32)
        logits = tf.cast(logits, tf.int32)

        if self._indicator == "accuracy":
            metric, metric_update = tf.metrics.accuracy(labels, logits)
            return metric, metric_update
        elif self._indicator == "precision":
            metric, metric_update = tf.metrics.precision(labels, logits)
            return metric, metric_update
        elif self._indicator == "recall":
            metric, metric_update = tf.metrics.recall(labels, logits)
            return metric, metric_update
        elif self._indicator == "f1_score":
            metric_recall, metric_update_recall = tf.metrics.recall(labels, logits)
            metric_precision, metric_update_precision = tf.metrics.precision(labels, logits)
            metric = 2.0 * metric_recall * metric_precision / (metric_recall + metric_precision)
            metric_update = tf.group([metric_update_recall, metric_update_precision])

            return metric, metric_update
        elif self._indicator == "fn":
            metric, metric_update = tf.metrics.false_negatives(labels, logits)
            return metric, metric_update
        elif self._indicator == "fp":
            metric, metric_update = tf.metrics.false_positives(labels, logits)
            return metric, metric_update
        elif self._indicator == "tp":
            metric, metric_update = tf.metrics.true_positives(labels, logits)
            return metric, metric_update
        elif self._indicator == "tn":
            metric, metric_update = tf.metrics.true_negatives(labels, logits)
            return metric, metric_update          
        else:
            raise ValueError("unsupported metrics")

def reset_metrics(sess, debug = False):
    lvars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "metrics*")
    lvars_initializer = tf.variables_initializer(var_list = lvars)
    sess.run(lvars_initializer)

    if debug:
        variables_names = [v.name for v in lvars]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}".format(k))
            print("Shape: {}".format(v.shape))
            print(v)


class Summaries():
    def __init__(self):
        self._train_summaries = []
        self._val_summaries = []
        self._test_summaries = []
        self._stats_summaries = []
        self._debug_summaries = []
      
    def register(self, category, name, tensor):
        if category == 'train':
            if 'loss' in name:
                self._train_summaries.append(tf.summary.scalar(name, tensor))
            elif 'learning_rate' in name:
                self._train_summaries.append(tf.summary.scalar(name, tensor))
            elif 'mean' in name:
                self._train_summaries.append(tf.summary.histogram(name, tensor))
            elif 'variance' in name:
                self._train_summaries.append(tf.summary.histogram(name, tensor))
            elif 'w' in name:
                self._train_summaries.append(tf.summary.histogram(name, tensor))
            elif 'b' in name:
                self._train_summaries.append(tf.summary.histogram(name, tensor))
            elif 'reward' in name:
                self._train_summaries.append(tf.summary.scalar(name, tensor))
            elif 'image' in name:
                ch = tensor.get_shape().as_list()[-1]
                data = tf.split(tensor, ch, axis = -1)
                for i in range(ch):
                    self._train_summaries.append(tf.summary.image("{}{}".format(name, i), data[i]))
            else:
                raise ErrorValue('unknown {} related to train statistics'.format(name))
        elif category == 'val':
            if 'loss' in name:
                self._val_summaries.append(tf.summary.scalar(name, tensor))
            elif name == 'val_deviation':
                self._val_summaries.append(tf.summary.scalar('val_deviation', tensor))
            elif name == 'val_accuracy':
                self._val_summaries.append(tf.summary.scalar('val_accuracy', tensor))
            elif name in ["accuracy", "precision", "recall", "f1_score"]:
                self._val_summaries.append(tf.summary.scalar('val_{}'.format(name), tensor))
            elif 'reward' in name:
                self._val_summaries.append(tf.summary.scalar(name, tensor))
            else:
                raise ErrorValue('unknown {} related to validation statistics'.format(name))
        elif category == 'test':
            if name == 'test_deviation':
                self._test_summaries.append(tf.summary.scalar('test_deviation', tensor))
            elif name == 'test_accuracy':
                self._test_summaries.append(tf.summary.scalar('test_accuracy', tensor))
            elif name in ["accuracy", "precision", "recall", "f1_score"]:
                self._test_summaries.append(tf.summary.scalar('test_{}'.format(name), tensor))
            elif 'image' in name:
                ch = tensor.get_shape().as_list()[-1]
                data = tf.split(tensor, ch, axis = -1)
                for i in range(ch):
                    self._test_summaries.append(tf.summary.image("{}{}".format(name, i), data[i]))
            else:
                self._test_summaries.append(tf.summary.scalar(name, tensor))
        elif category == 'stats':
            self._stats_summaries.append(tf.summary.scalar('test_{}'.format(name), tensor))
        elif category == 'debug':
            if 'dist' in name:
                self._debug_summaries.append(tf.summary.histogram(name, tensor))
            elif 'conv' in name:
                self._debug_summaries.append(tf.summary.histogram(name, tensor))
            elif 'input' in name:
                self._debug_summaries.append(tf.summary.histogram(name, tensor))
            elif 'label' in name:
                self._debug_summaries.append(tf.summary.histogram(name, tensor))
            elif 'state' in name:
                self._debug_summaries.append(tf.summary.histogram(name, tensor))
            elif 'action' in name:
                self._debug_summaries.append(tf.summary.histogram(name, tensor))
            else:
                raise ErrorValue('unknown {} related to debug statistics'.format(name))    
        else:
            raise ValueError('unknown category {}'.format(category)) 

    def merge(self, category, summ):
        if category == 'train':
            self._train_summaries += summ
        elif category == 'val':
            self._val_summaries += summ
        elif category == 'test':
            self._test_summaries += summ
        elif category == 'stats':
            self._stats_summaries += summ
        elif category == 'debug':
            self._debug_summaries += summ
        else:
            raise ValueError('unknown category {}'.format(category))

    def __call__(self, category):
        if category == 'train':
            assert self._train_summaries, 'list is empty'
            summ_op = tf.summary.merge(self._train_summaries)
        elif category == 'val':
            assert self._val_summaries, 'list is empty'
            summ_op = tf.summary.merge(self._val_summaries)
        elif category == 'test':
            assert self._test_summaries, 'list is empty'
            summ_op = tf.summary.merge(self._test_summaries)
        elif category == 'stats':
            assert self._stats_summaries, 'list is empty'
            summ_op = tf.summary.merge(self._stats_summaries)
        elif category == 'debug':
            assert self._debug_summaries, 'list is empty'
            summ_op = tf.summary.merge(self._debug_summaries)
        else:
            raise ValueError('unknown category {}'.format(category))

        return summ_op

class BatchNorm(snt.AbstractModule):
    def __init__(self, name = "batch_norm"):
        super(BatchNorm, self).__init__(name = name)
        with self._enter_variable_scope():
            self._bn = normalization.BatchNormalization(axis = 1,
                epsilon = np.finfo(np.float32).eps, momentum = 0.9)

    def _build(self, inputs, is_training = True, test_local_stats = False):
        outputs = self._bn(inputs, training = is_training)

        self._add_variable(self._bn.moving_mean)
        self._add_variable(self._bn.moving_variance)

        return outputs

    def _add_variable(self, var):
        if var not in tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES):
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)


class ChanNorm(snt.AbstractModule):
    def __init__(self, name = "chan_norm"):
        super(ChanNorm, self).__init__(name = name)

    def _build(self, inputs, is_training = True):
        if inputs.get_shape().ndims > 2:
            m, var = tf.nn.moments(inputs, list(range(1, inputs.get_shape().ndims - 1)), keepdims = True)
        else:
            m, var = tf.nn.moments(inputs, [1], keepdims = True)

        outputs = (inputs - m) / (var + np.finfo(np.float32).eps)

        return outputs

class PixelNorm(snt.AbstractModule):
    def __init__(self, name = "pixel_norm"):
        super(PixelNorm, self).__init__(name = name)

    def _build(self, inputs, is_training = True):
        m, var = tf.nn.moments(inputs, range(inputs.get_shape().ndims - 1, inputs.get_shape().ndims))
        m = tf.expand_dims(m, axis = -1)
        var = tf.expand_dims(var, axis = -1)
        outputs = (inputs - m) / (tf.sqrt(var) + np.finfo(np.float32).eps)

        return outputs, m, var

class Downsample1D(snt.AbstractModule):
    """
    1D downsampling by taking the average of every n entries
    """
    def __init__(self, factor, ptype = 'AVG', padding = 'SAME', 
                 verbose = False, name = "downsample_1d"):
        super(Downsample1D, self).__init__(name = name)
        self._factor = factor
        self._ptype = ptype
        self._padding = padding

        if verbose:
            print('Downsample type: {}, factor: {}, padding: {}'.format(ptype, factor, padding))

    def _build(self, inputs):
        return tf.nn.pool(inputs, [self._factor], self._ptype, self._padding, strides = [self._factor])

class Downsample2D(snt.AbstractModule):
    """
    2D downsampling by taking every n by n entries
    """
    def __init__(self, factor, ptype = 'AVG', padding = 'SAME', 
                 verbose = False, name = "downsample_2d"):
        super(Downsample2D, self).__init__(name = name)
        self._factor = factor
        self._ptype = ptype
        self._padding = padding

        if verbose:
            print('Downsample type: {}, factor: {}, padding: {}'.format(ptype, factor, padding))

    def _build(self, inputs):
        return tf.nn.pool(inputs, [self._factor, self._factor], self._ptype,
                          self._padding, strides = [self._factor, self._factor])

class DownsampleAlongH(snt.AbstractModule):
    """
    Downsampling by taking every n by n entries along H (rows)
    """
    def __init__(self, factor, ptype = 'AVG', padding = 'SAME', 
                 verbose = False, name = "downsample_along_h"):
        super(DownsampleAlongH, self).__init__(name = name)
        self._factor = factor
        self._ptype = ptype
        self._padding = padding

        if verbose:
            print('Downsample type: {}, factor: {}, padding: {}'.format(ptype, factor, padding))

    def _build(self, inputs):
        return tf.nn.pool(inputs, [self._factor, 1], self._ptype,
                          self._padding, strides = [self._factor, 1])

class DownsampleAlongW(snt.AbstractModule):
    """
    Downsampling by taking every n by n entries along H (rows)
    """
    def __init__(self, factor, ptype = 'AVG', padding = 'SAME', verbose = False, name = "downsample_along_w"):
        super(DownsampleAlongW, self).__init__(name = name)
        self._factor = factor
        self._ptype = ptype
        self._padding = padding

        if verbose:
            print('Downsample type: {}, factor: {}, padding: {}'.format(ptype, factor, padding))

    def _build(self, inputs):
        return tf.nn.pool(inputs, [1, self._factor], self._ptype,
                          self._padding, strides = [1, self._factor])

class SubpixelShuffle(snt.AbstractModule):
    def __init__(self, name = "subpixel_shuffle"):
        super(SubpixelShuffle, self).__init__(name = name)

    def _build(self, inputs):
        ch = inputs.get_shape().as_list()[-1]
        assert (ch % 4 == 0), "channels must be divided by 4"
        factor = 2
        # assert(factor == 2), "Invalid factor {}, must be 2".format(factor)

        def shuffle(in_):
            
            dim = in_.get_shape().as_list()

            ss = tf.split(in_, factor, axis = -1)

            rs0 = tf.reshape(ss[0], [-1] + [dim[1]] + [factor * dim[2]] + [1])
            rs1 = tf.reshape(ss[1], [-1] + [dim[1]] + [factor * dim[2]] + [1])

            rs = tf.stack([rs0, rs1], axis = 2)
            rs = tf.reshape(rs, [-1] + [dim[1] * factor] + [dim[2] * factor] + [1])

            return rs

        final_outputs_list = []
        for inp in tf.split(inputs, num_or_size_splits = ch >> 2, axis = -1):
            outputs = shuffle(inp)
            final_outputs_list.append(outputs)

        final_outputs = tf.concat(final_outputs_list, axis = -1)

        return final_outputs

def spectral_norm(input_):
    """Performs Spectral Normalization on a weight tensor."""
    if len(input_.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input_, (-1, input_.shape[-1]))

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        input_.name.replace(":", "") + "/u_var",
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(
            tf.matmul(tf.transpose(w), u), axis=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input_.shape)

    return w_tensor_normalized

class Synchronizer():
    def __init__(self, module_source, module_target):
        assert (module_source.is_connected and module_target.is_connected), \
            "module is not connected"

        self._module_source_variables = snt.get_variables_in_module(module_source)
        self._module_target_variables = snt.get_variables_in_module(module_target)

    def __call__(self):
        sychronize_ops = [tf.assign(t, o) for t, o in
                          zip(self._module_target_variables, self._module_source_variables)]

        return sychronize_ops


class Synchronizer2():
    def __init__(self, module_source, module_target):
        self._source = '%s_source_network_variables' % module_source.module_name
        self._target = '%s_target_network_variables' % module_target.module_name

        for var in module_source.get_variables():
            self._add_variable(var, self._source)

        for var in module_target.get_variables():
            self._add_variable(var, self._target)

    def _add_variable(self, var, collection_name):
        if var not in tf.get_collection(collection_name):
            tf.add_to_collection(collection_name, var)

    def __call__(self):

        o_params = tf.get_collection(self._source)
        t_params = tf.get_collection(self._target)

        sychronize_ops = [tf.assign(t, o) for t, o in zip(t_params, o_params)]

        return sychronize_ops

class WeightedSynchronizer():
    def __init__(self, module_source, module_target, tau = 0.1):
        assert (module_source.is_connected and module_target.is_connected), \
            "module is not connected"

        self._module_source_variables = snt.get_variables_in_module(module_source)
        self._module_target_variables = snt.get_variables_in_module(module_target)
        self._tau = tau

    def __call__(self):
        sychronize_ops = [tf.assign(t, self._tau * o + (1 - self._tau) * t) for t, o in
                          zip(self._module_target_variables, self._module_source_variables)]

        return sychronize_ops

class T2F(snt.AbstractModule):
    def __init__(self, name = "t2f"):
        super(T2F, self).__init__(name = name)

    def _build(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        num_splits = 3

        waves = tf.split(inputs, num_splits, axis = -1)
        waves = tf.stack(waves, axis = 0)

        def trans(unused, wave):

            with tf.control_dependencies([tf.assert_equal(tf.shape(wave), tf.TensorShape((30, 128)))]):
                wave_freq = tf.fft(tf.cast(wave, tf.complex64))

            wave_power = tf.log(tf.abs(wave_freq) + sys.float_info.epsilon)

            # with tf.control_dependencies([tf.print(wave_power, output_stream = sys.stdout)]):
            #     wave_power = tf.reduce_mean(wave_power, axis = -1)

            wave_power = tf.cast(wave_power, tf.float32)

            return wave_power

        with tf.control_dependencies([tf.assert_equal(tf.shape(waves), tf.TensorShape((3, 30, 128)))]):
            powers = tf.scan(trans, waves, infer_shape = False)

        power_range = np.array(range(1, 31))
        powers = tf.gather(powers, power_range, axis = -1)

        powers = tf.reduce_mean(powers, axis = -1)
        powers = tf.reduce_mean(powers, axis = 0)

        return powers

class T2B(snt.AbstractModule):
    def __init__(self, name = "t2b"):
        super(T2B, self).__init__(name = name)

    def _build(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        num_splits = 3

        waves = tf.split(inputs, num_splits, axis = -1)
        waves = tf.stack(waves, axis = 0)

        def trans(unused, wave):

            with tf.control_dependencies([tf.assert_equal(tf.shape(wave), tf.TensorShape((30, 128)))]):
                wave_freq = tf.fft(tf.cast(wave, tf.complex64))

            wave_power = tf.log(tf.abs(wave_freq) + sys.float_info.epsilon)

            # with tf.control_dependencies([tf.print(wave_power, output_stream = sys.stdout)]):
            #     wave_power = tf.reduce_mean(wave_power, axis = -1)

            wave_power = tf.cast(wave_power, tf.float32)

            return wave_power

        with tf.control_dependencies([tf.assert_equal(tf.shape(waves), tf.TensorShape((3, 30, 128)))]):
            powers = tf.scan(trans, waves, infer_shape = False)

        delta_range = np.array(range(1,  4))
        theta_range = np.array(range(4,  8))
        alpha_range = np.array(range(8,  14))
        beta_range  = np.array(range(14, 30))

        delta_powers = tf.gather(powers, delta_range, axis = -1)
        theta_powers = tf.gather(powers, theta_range, axis = -1)
        alpha_powers = tf.gather(powers, alpha_range, axis = -1)
        beta_powers  = tf.gather(powers, beta_range,  axis = -1)

        delta_powers = tf.reduce_mean(delta_powers, axis = -1)
        theta_powers = tf.reduce_mean(theta_powers, axis = -1)
        alpha_powers = tf.reduce_mean(alpha_powers, axis = -1)
        beta_powers  = tf.reduce_mean(beta_powers,  axis = -1)

        delta_powers = tf.reduce_mean(delta_powers, axis = 0)
        theta_powers = tf.reduce_mean(theta_powers, axis = 0)
        alpha_powers = tf.reduce_mean(alpha_powers, axis = 0)
        beta_powers  = tf.reduce_mean(beta_powers,  axis = 0)

        powers = tf.concat([delta_powers, theta_powers, alpha_powers, beta_powers], axis = -1)

        return powers

def draw_image(images, tensor_name = "images"):
    import tfmpl

    @tfmpl.figure_tensor
    def draw(images):
        num_figs = len(images)

        height = math.ceil(num_figs / 2)

        # draw figures in two column
        fig = tfmpl.create_figures(1, figsize= (12.8 if num_figs > 1 else 6.4, 4.8 * height))[0]

        # pdb.set_trace()
        for i in range(height):
            for j in range(2):
                seq = i * 2 + j + 1
                if seq > num_figs:
                    fig.tight_layout()
                    return fig

                if num_figs == 1:
                    ax = fig.add_subplot(1, 1, 1)
                else:
                    ax = fig.add_subplot(height, 2, seq)

                ax = fig.add_subplot(height, 2, seq)
                ax.axis('off')
                ax.imshow(images[seq-1]['data'])
                ax.set_title(images[seq-1]['title'], fontsize = 24)

        fig.tight_layout()

        return fig

    image_tensor = draw(images)
    image_summary = tf.summary.image(tensor_name, image_tensor)
    sess = tf.get_default_session()
    assert sess != None, "Invalid session"
    image_str = sess.run(image_summary)

    return image_str
    

class Pad2D(snt.AbstractModule):
    def __init__(self, pad, mode = 'CONSTANT', data_format = 'NHWC', name = "pad2d"):
        super(Pad2D, self).__init__(name = name)
        self._pad = pad
        self._mode = mode
        self._data_format = data_format
        
    def _build(self, inputs):
        # Padding shape.
        if self._data_format == 'NHWC':
            paddings = [[0, 0], [0, self._pad[0]], [0, self._pad[1]], [0, 0]]
        elif self._data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [0, self._pad[0]], [0, self._pad[1]]]
        else:
            raise ValueError("Unknown data format: {}".format(self._data_format))
            
        return tf.pad(inputs, paddings, mode = self._mode)


def draw_image_v2(images, rows, cols, tensor_name = "images"):
    import tfmpl

    @tfmpl.figure_tensor
    def draw(images):
        num_figs = len(images)
        fig = tfmpl.create_figures(1, figsize= (12.8, 12.8))[0]

        # pdb.set_trace()
        for i in range(rows):
            for j in range(cols):
                seq = i * cols + j + 1
                if seq > num_figs:
                    fig.tight_layout()
                    return fig

                if num_figs == 1:
                    ax = fig.add_subplot(1, 1, 1)
                else:
                    ax = fig.add_subplot(rows, cols, seq)

                ax.axis('off')
                ax.imshow(images[seq-1, ...])

        fig.tight_layout()
        return fig

    image_tensor = draw(images)
    image_summary = tf.summary.image(tensor_name, image_tensor)
    sess = tf.get_default_session()
    assert sess != None, "Invalid session"
    image_str = sess.run(image_summary)

    return image_str

        