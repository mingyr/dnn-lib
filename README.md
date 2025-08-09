# dnn-lib

This repo contains various files with routines based on JAX and related libraries mainly from Google DeepMind

Notably, ordinary class definition allows underscore preceeding member variables, but classes inherited from nnx.Module should avoid doing so. Otherwise, especially your class contains member variables of the type nnx.Param, probably it will not be output by nnx.display method.

Meantime, to cater to the functional programming paradigm of JAX, you CANNOT directly jit the object's invoking method. Traditionally, it means you implement the __call__ method, instantiate an object and directly call the object by passing some parameters. Instead, you should do in the following way:
```python
    return_value, _ = nnx.jit(Class.__call__)(Obj, x, y)
```python


To use these routines, it is suggested to setup the env variable like "export PYTHONPATH=/path-to-where-dnn-lib-resides/dnn-lib:$PYTHONPATH" in Linux

