"""Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


__TF_VERSION__ = int(tf.__version__.split(".")[0])

    
def dense_layers(tensor, 
                 units, 
                 activation="relu", 
                 batch_norm=True,
                 use_bias=True,  
                 linear_top_layer=False, 
                 **kwargs):

  """Builds a stack of fully connected layers."""
  
  assert isinstance(units, list)
  n_layers = len(units)
  activation = _to_array(activation, n_layers)

  if len(get_shape(tensor)) != 2:
    tensor = tf.keras.layers.Flatten()(tensor)

  for i, (size, act) in enumerate(zip(units, activation)):
    act = tf.keras.activations.get(act) if act else None
    if i == len(units) - 1 and linear_top_layer:
      act = None

    with tf.compat.v1.variable_scope("dense_block_%d" % i):
      layer = tf.keras.layers.Dense(
        units=size, 
        activation=None,
        use_bias=use_bias,
        **kwargs
      )

      tensor = layer(tensor)

      if act:
        if batch_norm:
          tensor = batch_normalization(tensor)

        tensor = act(tensor)

  return tensor


def convolutional_layers(tensor, 
                         fmaps, 
                         kernels, 
                         stride, 
                         pool_size=0, 
                         pool_stride=0,
                         padding="same", 
                         activation="relu", 
                         pool_activation=None, 
                         pool_method="conv",
                         batch_norm=True,
                         use_bias=True,
                         flatten=False, 
                         factor=1, 
                         linear_top_layer=False, 
                         **kwargs):
                         
  """Builds a stack of convolutional layers."""

  assert isinstance(fmaps, list)  
  n_layers = len(fmaps)

  dims = get_shape(tensor)
  if len(dims) == 5:
    tensor = tf.reshape(tensor, (-1, dims[-3], dims[-2], dims[-1]))
  else:
    assert len(dims) == 4
      
  kernels = _to_array(kernels, n_layers)
  stride = _to_array(stride, n_layers)
  activation = _to_array(activation, n_layers)    
  pool_method = _to_array(pool_method, n_layers)
  pool_activation = _to_array(pool_activation, n_layers)
  pool_size = _to_array(pool_size, n_layers)
  pool_stride = _to_array(pool_stride, n_layers)
  
  for i, (fs, ks, ss, pz, pr, act, pool_act, pm) in enumerate(
    zip(fmaps, kernels, stride, pool_size, pool_stride,
        activation, pool_activation, pool_method)):

    act = tf.keras.activations.get(act) if act else None
    pool_act = tf.keras.activations.get(pool_act) if pool_act else None

    if i == len(fmaps) - 1 and linear_top_layer:
      act = None
      pool_act = None

    with tf.compat.v1.variable_scope("conv_block_%d" % i):
      if ks > 0 and ss > 0:
        layer = tf.keras.layers.Conv2D(
          filters=fs, 
          kernel_size=ks, 
          activation=None,
          strides=ss, 
          padding=padding, 
          use_bias=use_bias, 
          name="conv2d", 
          **kwargs
        )

        tensor = layer(tensor)

        if act:
          if batch_norm:     
            tensor = batch_normalization(tensor)

          tensor = act(tensor)

      if pz > 1 and pr > 0:
        if pm == "max":
          tensor = tf.keras.layers.MaxPool2D(
            pz, pr, padding, name="max_pool").apply(tensor)
        elif pm == "std":
          tensor = tf.space_to_depth(tensor, pz, name="space_to_depth")
        elif pm == "dts":
          tensor = tf.depth_to_space(tensor, pz, name="depth_to_space")
        elif pm == "conv":
          layer = tf.keras.layers.Conv2D(
            filters=fs,
            kernel_size=pz,
            activation=None,
            strides=pr,
            padding=padding,
            use_bias=use_bias,
            name="strided_conv2d",
            **kwargs
          )
          
          tensor = layer(tensor)

        else:
          raise("Unknown pool method.")

        if pool_act:
          if batch_norm:
            tensor = batch_normalization(tensor)

          tensor = pool_act(tensor)

  if flatten:
    num_features = np.prod(get_shape(tensor)[1:])
    units = factor * num_features
    tensor = tf.reshape(tensor, (-1, units))

  return tensor


def batch_normalization(tensor, epsilon=0.001, momentum=0.9):
  """Batch normalization."""

  with tf.compat.v1.variable_scope("batch_normalization"):
    n_channels = get_shape(tensor)[-1]

    beta = tf.compat.v1.get_variable(
      "BatchNorm/beta", n_channels, initializer=tf.zeros_initializer())
    gamma = tf.compat.v1.get_variable(
      "BatchNorm/gamma", n_channels, initializer=tf.ones_initializer())

    moving_mean = tf.compat.v1.get_variable(
      "BatchNorm/moving_mean", n_channels, initializer=tf.zeros_initializer())
    moving_variance = tf.compat.v1.get_variable(
      "BatchNorm/moving_variance", n_channels, initializer=tf.ones_initializer())

    tensor = tf.nn.batch_normalization(
      tensor, moving_mean, moving_variance, beta, gamma, epsilon) 

    session = tf.compat.v1.get_default_session()
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    variables = tf.compat.v1.global_variables(scope=scope)
    session.run(tf.compat.v1.variables_initializer(variables))

  return tensor    


def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""

  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [
    s[1] if s[0] is None else s[0]
    for s in zip(static_shape, dynamic_shape)
  ]

  return dims


def entry_stop_gradients(tensor, mask):
  mask_h = tf.logical_not(mask)      
  mask = tf.cast(mask, dtype=tensor.dtype)
  mask_h = tf.cast(mask_h, dtype=tensor.dtype)

  return mask * tensor + tf.stop_gradient(mask_h * tensor)


def get_mu(decision):
  assert len(get_shape(decision)) == 2
  batch_size, n_leaves = get_shape(decision)
  depth = int(np.log2(n_leaves))

  decision = tf.expand_dims(decision, 2)
  decision_comp = 1 - decision
  decision = tf.concat([decision, decision_comp], 2)

  _mu = tf.ones((batch_size, 1, 1))

  begin = 1
  end = 2

  for d in range(0, depth):
    _mu = tf.reshape(_mu, (batch_size, -1, 1))
    _mu = tf.tile(_mu, (1, 1, 2))
    _mu *= decision[:, begin:end, :]

    begin = end
    end = begin + 2 ** (d + 1)

  mu = tf.reshape(_mu, (batch_size, n_leaves))

  return mu


def tree_op(decision, leaves):
  """Neural tree op."""

  batch_size, n_leaves, n_units = get_shape(leaves)

  mask = np.ones(get_shape(decision)[1])
  mask[0] = 0
  mask = mask.astype(np.bool)
  decision = entry_stop_gradients(decision, mask)

  mu = get_mu(decision)
  mu = tf.expand_dims(mu, 2)
  mu = tf.tile(mu, (1, 1, n_units))

  w_output = tf.multiply(mu, leaves)
  output = tf.reduce_sum(w_output, 1)

  return output


def extract_patches(image, pts_2d, patch_shape):
  """Extracts patches from landmarks.

  Returns:
    Tensor [batch_size, n_patches, patch_shape[0], patch_shape[1], num_channels]
  """

  patches = []
  n_patches = get_shape(pts_2d)[1]

  for i in range(n_patches):
    offsets = tf.gather(pts_2d, [i], axis=1)
    offsets = offsets[:, 0, :]

    patch = tf.compat.v1.image.extract_glimpse(
      image,
      size=patch_shape,
      offsets=offsets,
      centered=False,
      normalized=False
    )
    
    patches.append(patch)

  patches = tf.stack(patches, 1)
  
  return patches


def _to_array(value, size, default_val=None):
  if value is None:
    value = [default_val] * size
  elif not isinstance(value, list):
    value = [value] * size
  if len(value) == 1:
    value = value * size
  return value
