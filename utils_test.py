import os, sys
from utils import *
import jax
import chex


def test_jax_ver():
    jax_ver = JAXVer(jax)

    print(jax_ver.major)
    print(jax_ver.minor)

def test_loss():
    num_classes = 2

    logits_reg = jnp.array([[1, 0], [0, 1]], jnp.float32)
    labels_reg = jnp.array([[1, 0], [1, 0]], jnp.float32)

    logits_xen = jnp.array([[1, 0], [0, 1]], jnp.float32)
    labels_xen = jnp.array([[0], [0]], jnp.int32)

    reg_loss, _ = LossRegression()(jax.nn.identity, logits_reg, labels_reg)
    xen_loss, _ = LossClassification(num_classes)(jax.nn.identity, logits_xen, labels_xen)

    reg_val = ValRegression(jax.nn.identity)(logits_reg, labels_reg)
    xen_val = ValClassification(jax.nn.identity)(logits_xen, labels_xen)

    reg_test = TestRegression(jax.nn.identity)(logits_reg, labels_reg)
    xen_test = TestClassification(jax.nn.identity)(logits_xen, labels_xen)

    print(type(reg_loss))
    print(reg_loss)


    
    # summary = Summary('/tmp')
    # with summary.as_default():
    with Summary('/tmp') as summary:
        summary('loss', reg_loss, 0)
        summary('loss', xen_loss, 1)
        summary('accu', reg_val, 0)
        summary('accu', xen_val, 1)
        summary('stat', reg_test, 0)
        summary('stat', xen_test, 1)


def test_conv():
    rngs = nnx.Rngs(0)
    x = jnp.ones((1, 8, 3))

    # SAME padding
    layer = DepthwiseConv(in_features=3, kernel_size=(3,),
                          padding='SAME', rngs=rngs)
    y = layer(x)

    chex.assert_equal_shape([x, y])

    x = jnp.ones((1, 8, 16, 3))
    layer = DepthwiseConv(in_features=3, kernel_size=(3,3),
                          padding='SAME', rngs=rngs)
    y = layer(x)

    chex.assert_equal_shape([x, y])

    x = jnp.ones((1, 8, 3))

    # SAME padding
    layer = SeparableConv(in_features=3, out_features=6, kernel_size=(3,),
                          padding='SAME', rngs=rngs)
    y = layer(x)

    chex.assert_shape(y, (1, 8, 6))

    x = jnp.ones((1, 8, 16, 3))
    layer = SeparableConv(in_features=3, out_features=6, kernel_size=(3,3),
                          padding='SAME', rngs=rngs)
    y = layer(x)

    chex.assert_shape(y, (1, 8, 16, 6))


if __name__ == "__main__":
    # test_jax_ver()
    test_loss()
    # test_conv()


