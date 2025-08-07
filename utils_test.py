import os, sys
from utils import *
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

    reg_loss = LossRegression()(logits_reg, labels_reg)
    xen_loss = LossClassification(num_classes)(logits_xen, labels_xen)

    reg_val = ValRegression()(logits_reg, labels_reg)
    xen_val = ValClassification()(logits_xen, labels_xen)

    reg_test = TestRegression()(logits_reg, labels_reg)
    xen_test = TestClassification()(logits_xen, labels_xen)

    summaries = Summaries()
    summaries('train', reg_loss)
    summaries('train', xen_loss)
    summaries('val', reg_val)
    summaries('val', xen_val)
    summaries('test', reg_test)
    summaries('test', xen_test)

    summaries.dump()

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
    # test_loss()
    test_conv()


