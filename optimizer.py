import tensorflow as tf
import sonnet as snt

class Adam(snt.AbstractModule):
    def __init__(self, learning_rate = 0.01, lr_decay = False, lr_decay_steps = None, lr_decay_factor = None, 
                 clip = False, clip_value = 1.0, manually_update_ops = False, master = True, name = "adam"):
        super(Adam, self).__init__(name = name)

        self._lr_decay = lr_decay
        self._clip = clip
        self._clip_value = clip_value
        self._manually_update_ops = manually_update_ops
        self._master = master

        # decay_steps could also be calculated based on other parameters
        # Calculate the learning rate schedule.
        # num_batches_per_epoch = (num_train_samples / train_batch_size)
        # decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        with self._enter_variable_scope():
            # global step variable must be shared across multiple optimizer instances
            self._global_step = tf.train.get_or_create_global_step()

        if lr_decay:
            assert(lr_decay_steps > 0), "invalid learning rate decay step"
            assert(lr_decay_factor > 0), "invalid learning rate decay factor"

            # Decay the learning rate exponentially based on the number of steps.
            self._lr = tf.train.exponential_decay(learning_rate, self._global_step, 
                                                  lr_decay_steps, lr_decay_factor, staircase = True)
        else:
            self._lr = learning_rate

        self._opt = tf.train.AdamOptimizer(self._lr)

    def _build(self, loss, var_list = None):
        trainable_variables = var_list if var_list else tf.trainable_variables()

        if self._manually_update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self._clip:
            gvs = self._opt.compute_gradients(loss, var_list = trainable_variables)
            capped_gvs = [(tf.clip_by_value(grad, - self._clip_value, self._clip_value), var) for grad, var in gvs]

            if self._manually_update_ops:
                with tf.control_dependencies(update_ops):
                    train_op = self._opt.apply_gradients(capped_gvs, global_step = self._global_step if self._master else None)
            else:
                train_op = self._opt.apply_gradients(capped_gvs, global_step = self._global_step if self._master else None)
        else:
            if self._manually_update_ops:
                with tf.control_dependencies(update_ops):
                    train_op = self._opt.minimize(loss, global_step = self._global_step if self._master else None, var_list = var_list)
            else:
                train_op = self._opt.minimize(loss, global_step = self._global_step if self._master else None, var_list = var_list)

        return train_op

    def compute(self, loss, var_list = None):
        trainable_variables = var_list if var_list else tf.trainable_variables()
        grads = self._opt.compute_gradients(loss, trainable_variables)
        return grads

    def apply(self, grads):
        if self._manually_update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = self._opt.apply_gradients(grads, global_step = self._global_step if self._master else None)
        else:
            train_op = self._opt.apply_gradients(grads, global_step = self._global_step if self._master else None)

        return train_op

    @property
    def lr(self):
        return self._lr

    @property
    def global_step(self):
        return self._global_step

class RMSProp(snt.AbstractModule):
    def __init__(self, learning_rate = 0.01, epsilon = 1e-10, lr_decay = False,
                 lr_decay_steps = None, lr_decay_factor = None, clip = False, 
                 clip_value = 1.0, manually_update_ops = False, master = True, name = "rmsprop"):
        super(RMSProp, self).__init__(name = name)
        # decay_steps could also be calculated based on other parameters
        # Calculate the learning rate schedule.
        # num_batches_per_epoch = (num_train_samples / train_batch_size)
        # decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        self._lr_decay = lr_decay
        self._clip = clip
        self._clip_value = clip_value
        self._global_step = tf.train.get_or_create_global_step()
        self._manually_update_ops = manually_update_ops
        self._master = master

        if lr_decay:
            assert(lr_decay_steps > 0), "invalid learning rate decay step"
            assert(lr_decay_factor > 0), "invalid learning rate decay factor"

            # Decay the learning rate exponentially based on the number of steps.
            self._lr = tf.train.exponential_decay(learning_rate, self._global_step, 
                                                  lr_decay_steps, lr_decay_factor, staircase = True)
        else:
            self._lr = learning_rate 

        self._opt = tf.train.RMSPropOptimizer(self._lr, epsilon = epsilon)

    def _build(self, loss, var_list = None):
        trainable_variables = var_list if var_list else tf.trainable_variables()

        if self._manually_update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self._clip:
            gvs = self._opt.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, - self._clip_value, self._clip_value), var) for grad, var in gvs]

            if self._manually_update_ops:
                with tf.control_dependencies(update_ops):
                    train_op = self._opt.apply_gradients(capped_gvs, global_step = self._global_step if self._master else None)
            else:
                train_op = self._opt.apply_gradients(capped_gvs, global_step = self._global_step)
        else:
            if self._manually_update_ops:
                with tf.control_dependencies(update_ops):
                    train_op = self._opt.minimize(loss, global_step = self._global_step if self._master else None)
            else:
                train_op = self._opt.minimize(loss, global_step = self._global_step if self._master else None)
 
        return train_op

    def compute(self, loss):
        trainable_variables = var_list if var_list else tf.trainable_variables()
        grads = self._opt.compute_gradients(loss, trainable_variables)
        return grads

    def apply(self, grads):
        if self._manually_update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = self._opt.apply_gradients(grads, global_step = self._global_step if self._master else None)
        else:
            train_op = self._opt.apply_gradients(grads, global_step = self._global_step if self._master else None)

        return train_op

    @property
    def lr(self):
        return self._lr

class SGD(snt.AbstractModule):
    def __init__(self, num_iters, learning_rate, 
                 lr_decay = True, manually_update_ops = False, name = "SGD"):
        super(SGD, self).__init__(name = name)
        self._global_step = tf.train.get_or_create_global_step()
        self._lr_decay = lr_decay
        self._manually_update_ops = manually_update_ops

        with self._enter_variable_scope():
            if lr_decay:
                # Decay the learning rate exponentially based on the number of steps.
                self._lr = learning_rate / (1.0 + 10.0 * tf.cast(self._global_step, tf.float32) / tf.cast(num_iters, tf.float32)) ** 0.75
            else:
                self._lr = learning_rate

            self._opt = tf.train.GradientDescentOptimizer(self._lr)
 
    def _build(self, loss):
        if self._manually_update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                if self._lr_decay:
                    train_op = self._opt.minimize(loss, global_step = self._global_step)
                else:
                    train_op = self._opt.minimize(loss)
        else:
            if self._lr_decay:
                train_op = self._opt.minimize(loss, global_step = self._global_step)
            else:
                train_op = self._opt.minimize(loss)

        return train_op

    @property
    def lr(self):
        return self._lr

def test():
    # X and Y data
    x_train = [1, 2, 3]
    y_train = [1, 2, 3]

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Our hypothesis XW+b
    hypothesis = x_train * W + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    learning_rate = 0.01
    
    # Minimize
    optimizer = Adam()
    # optimizer = RMSProp(lr_decay = False)
    train = optimizer(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))

if __name__ == '__main__':
    test()

