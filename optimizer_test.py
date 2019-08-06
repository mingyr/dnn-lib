from optimizer import *

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

