import os, sys
import numpy as np
import tensorflow as tf
from utils import *

def test_tfver():
    tfver = TFVer(tf)

    print(tfver.major)
    print(tfver.minor)

def test_loss():

    num_classes = 2

    logits_reg = tf.constant([[1, 0], [0, 1]], tf.float32)
    labels_reg = tf.constant([[1, 0], [1, 0]], tf.float32)

    logits_xen = tf.constant([[1, 0], [0, 1]], tf.float32)
    labels_xen = tf.constant([[0], [0]], tf.int64)

    reg_loss = LossRegression()
    reg_loss_op = reg_loss(logits_reg, labels_reg)

    xen_loss = LossClassification(num_classes)
    xen_loss_op = xen_loss(logits_xen, labels_xen)

    reg_xval = XValRegression()
    reg_xval_op = reg_xval(logits_reg, labels_reg)

    xen_xval = XValClassification()
    xen_xval_op = xen_xval(logits_xen, labels_xen)

    reg_test = TestRegression()
    reg_test_op = reg_test(logits_reg, labels_reg)

    xen_test = TestClassification()
    xen_test_op = xen_test(logits_xen, labels_xen)

    summ_op = tf.summary.merge()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('utils_output', sess.graph)

        sess.run(tf.global_variables_initializer())

        _, _, _, _, _, _, summ_str = sess.run([reg_loss_op, xen_loss_op, reg_xval_op, xen_xval_op, reg_test_op, xen_test_op, summ_op])

        writer.add_summary(summ_str, 0)
        writer.close()

def test_metrics():
    import numpy as np


    labels = tf.constant([1], tf.int32)
    logits = tf.constant([2], tf.int32)

    metrics = Metrics("accuracy")
    accuracy, accuracy_update_op = metrics(labels, logits)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        sess.run([accuracy_update_op])
        accu = sess.run(accuracy)
        print("accu -> {}".format(accu))

def test_channorm():
    s = tf.constant([
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ]
    ], tf.float32)

    b = ChanNorm()(s)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v = sess.run(b)

        print(v)

def test_pixelnorm():
    s = tf.constant([
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ]
    ], tf.float32)

    b, m, var = PixelNorm()(s)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v, m_v, var_v = sess.run([b, m, var])

        print(v)
        print(m_v)
        print(var_v)


def test_downsampling():
    d1d = tf.constant([
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    ], tf.float32)

    d2d = tf.constant([
        [
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
            [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
        ]
    ], tf.float32)

    d1d_ds = Downsample1D(2)(d1d)
    d2d_ds = Downsample2D(3)(d2d)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        d1d_ds_val, d2d_ds_val = sess.run([d1d_ds, d2d_ds])

        print(d1d_ds_val)
        print(d2d_ds_val)

def test_synchronize():
    class Model(snt.AbstractModule):
        def __init__(self, name = "model"):
            super(Model, self).__init__(name = name)
            with self._enter_variable_scope():
                self._seq = snt.Sequential([snt.Linear(256), tf.sigmoid,
                                            snt.Linear(128), tf.sigmoid,
                                            snt.Linear(56),  tf.sigmoid,
                                            snt.Linear(1)])

        def _build(self, s, a):
            inp = tf.concat([s, a], axis = -1)
            out = self._seq(inp)

            return out

    net = Model()
    net2 = Model()

    inputs = tf.constant(1.0, tf.float32, [8, 30, 750])

    outputs = net(inputs)
    outputs2 = net2(inputs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("before synchronization")
    v, v2 = sess.run([outputs, outputs2])

    print("v:")
    print(v)
    print("v2:")
    print(v2)

    synchronizer = Synchronizer(net, net2)

    sychronize_ops = synchronizer()

    print("after synchronization")

    sess.run(sychronize_ops)
    v, v2 = sess.run([outputs, outputs2])

    print("v:")
    print(v)
    print("v2:")
    print(v2)

def test_weightedsychronizer():
    class Model(snt.AbstractModule):
        def __init__(self, name = "model"):
            super(Model, self).__init__(name = name)
            with self._enter_variable_scope():
                self._seq = snt.Sequential([snt.Linear(256), tf.sigmoid,
                                            snt.Linear(128), tf.sigmoid,
                                            snt.Linear(56),  tf.sigmoid,
                                            snt.Linear(1)])

        def _build(self, s, a):
            inp = tf.concat([s, a], axis = -1)
            out = self._seq(inp)

            return out
    
    net = Model(name = "c_network")
    net2 = Model(name = "target_c")

    states = tf.constant(1.0, tf.float32, [32, 90])
    actions = tf.constant(1.0, tf.float32, [32, 1])

    outputs = net(states, actions)
    outputs2 = net2(states, actions)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("before synchronization")
    v, v2 = sess.run([outputs, outputs2])

    print("v:")
    print(v)
    print("v2:")
    print(v2)

    if not os.path.exists("model/pre"):
        os.makedirs('model/pre')

    saver.save(sess, "model/pre/model.ckpt")

    synchronizer = WeightedSynchronizer(net, net2, tau = 0.5)

    sychronize_ops = synchronizer()

    print("after synchronization")

    sess.run(sychronize_ops)
    v, v2 = sess.run([outputs, outputs2])

    print("v:")
    print(v)
    print("v2:")
    print(v2)

    if not os.path.exists("model/post"):
        os.makedirs('model/post')

    saver.save(sess, "model/post/model.ckpt")


def test_t2f():
    t2f = T2F()

    state = tf.random_normal([30, 384], dtype = tf.float32)
    p = t2f(state)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        power_val = sess.run(p)

        np.set_printoptions(threshold = np.nan)

        print(power_val.shape)
        print(power_val)

def test_t2b():
    t2b = T2B()

    state = tf.random_normal([30, 384], dtype = tf.float32)
    p = t2b(state)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        power_val = sess.run(p)

        np.set_printoptions(threshold = np.nan)

        print(power_val.shape)
        print(power_val)

def test_subpixel_shuffle():
    s1 = tf.constant(1.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s2 = tf.constant(2.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s3 = tf.constant(3.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s4 = tf.constant(4.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s5 = tf.constant(5.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s6 = tf.constant(6.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s7 = tf.constant(7.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s8 = tf.constant(8.0, shape = (2, 3, 3, 1), dtype = tf.float32)
    s9 = tf.constant(9.0, shape = (2, 3, 3, 1), dtype = tf.float32)


    s = tf.concat([s1, s2, s3, s4, s5, s6, s7, s8, s9], axis = -1)
    k = SubpixelShuffle2(s1.get_shape().as_list(), 3)(s)

    t = tf.constant([
        [
            [ [1], [2], [3], [1], [2], [3], [1], [2], [3] ],
            [ [4], [5], [6], [4], [5], [6], [4], [5], [6] ],
            [ [7], [8], [9], [7], [8], [9], [7], [8], [9] ],
            [ [1], [2], [3], [1], [2], [3], [1], [2], [3] ],
            [ [4], [5], [6], [4], [5], [6], [4], [5], [6] ],
            [ [7], [8], [9], [7], [8], [9], [7], [8], [9] ],
            [ [1], [2], [3], [1], [2], [3], [1], [2], [3] ],
            [ [4], [5], [6], [4], [5], [6], [4], [5], [6] ],
            [ [7], [8], [9], [7], [8], [9], [7], [8], [9] ],
        ],
        [
            [ [1], [2], [3], [1], [2], [3], [1], [2], [3] ],
            [ [4], [5], [6], [4], [5], [6], [4], [5], [6] ],
            [ [7], [8], [9], [7], [8], [9], [7], [8], [9] ],
            [ [1], [2], [3], [1], [2], [3], [1], [2], [3] ],
            [ [4], [5], [6], [4], [5], [6], [4], [5], [6] ],
            [ [7], [8], [9], [7], [8], [9], [7], [8], [9] ],
            [ [1], [2], [3], [1], [2], [3], [1], [2], [3] ],
            [ [4], [5], [6], [4], [5], [6], [4], [5], [6] ],
            [ [7], [8], [9], [7], [8], [9], [7], [8], [9] ],
        ],
    ], tf.float32)

    with tf.control_dependencies([tf.assert_equal(k, t)]):
        b = tf.constant("True", tf.string)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        b_v = sess.run(b)

        print("test passed: {}".format(b_v))

        v = sess.run(k)

        print(v)


def test_subpixel_shuffle_2():
    from utils import get_batch_size

    s = tf.constant([
                        [
                            [
                                [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
                            ],
                            [
                                [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
                            ],
                            [
                                [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
                            ],
                            [
                                [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
                            ]
                        ]
                    ], dtype = tf.int32)

    s2 = tf.constant([
                         [
                             [
                                 [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]
                             ],
                             [
                                 [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]
                             ],
                             [
                                 [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]
                             ],
                             [
                                 [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]
                             ]
                         ]
                     ], dtype = tf.int32)

    t = tf.constant([
                        [
                            [[1], [2], [1], [2], [1], [2], [1], [2]],
                            [[3], [4], [3], [4], [3], [4], [3], [4]],
                            [[1], [2], [1], [2], [1], [2], [1], [2]],
                            [[3], [4], [3], [4], [3], [4], [3], [4]],
                            [[1], [2], [1], [2], [1], [2], [1], [2]],
                            [[3], [4], [3], [4], [3], [4], [3], [4]],
                            [[1], [2], [1], [2], [1], [2], [1], [2]],
                            [[3], [4], [3], [4], [3], [4], [3], [4]],

                        ]
                    ], dtype = tf.int32)

    t2 = tf.constant([
                         [
                             [[1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6]],
                             [[3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8]],
                             [[1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6]],
                             [[3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8]],
                             [[1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6]],
                             [[3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8]],
                             [[1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6], [1, 5], [2, 6]],
                             [[3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8], [3, 7], [4, 8]],
                         ]
                     ], dtype = tf.int32)

    sps = SubpixelShuffle()
    h = sps(s)

    sps2 = SubpixelShuffle()
    h2 = sps2(s2)

    with tf.control_dependencies([tf.assert_equal(h2, t2)]):
        h2 = tf.identity(h2)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        h_v = sess.run(h)

        print("h_v.shape -> {}".format(h_v.shape))
        print("test passed")

        h2_v = sess.run(h2)

        print("h2_v.shape -> {}".format(h2_v.shape))
        print("test passed")


if __name__ == "__main__":
    # test_tfver()
    # test_metrics()
    # test_channorm()
    # test_pixelnorm()

    # test_subpixel_shuffle()
    test_subpixel_shuffle_2()

    # test_weightedsychronizer()
    # test_t2b()

