"""Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import common.model
import common.ops as ops


class Model(common.model.Model):
  """Deep ensemble."""

  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs)


  def __call__(self, input):
    """Forward."""

    params = self.params
    assert params.model.num_iterations
    n_landmarks = params.dataset.pts_2d.shape[0]
    delta = tf.zeros_like(input["pts_initial"])

    output = {}

    for step in range(params.model.num_iterations):
      with tf.compat.v1.variable_scope("step%d" % step, reuse=False):
        current = input["pts_initial"] + delta

        with tf.compat.v1.variable_scope("features"):
          patches = ops.extract_patches(
            image=input["image"],
            pts_2d=current,
            patch_shape=params.model.features.patch_shape
          )

          features = ops.convolutional_layers(
            patches,
            fmaps=params.model.features.fmaps,
            kernels=params.model.features.kernels,
            stride=params.model.features.stride,
            padding=params.model.features.padding,
            batch_norm=params.model.batch_norm,
            pool_method=params.model.features.pool_method,
            pool_size=params.model.features.pool_size,
            pool_stride=params.model.features.pool_stride,
            flatten=True,
            factor=n_landmarks
          )

        with tf.compat.v1.variable_scope("predictor"):
          committee = []

          for i in range(params.model.predictor.n_block):
            with tf.compat.v1.variable_scope(str(i)):
              hidden_units = params.model.predictor.hidden_units
              if not isinstance(hidden_units, list):
                hidden_units = [hidden_units]

              ds_prediction_i = ops.dense_layers(
                features,
                units=hidden_units + [2 * n_landmarks],
                activation=["relu"] * len(hidden_units) + [""],
                batch_norm=params.model.batch_norm
              )

              committee.append(ds_prediction_i)

          committee = tf.stack(committee, axis=1)

          ds_prediction = tf.reduce_mean(committee, 1)
          ds_prediction = tf.reshape(ds_prediction, (-1, n_landmarks, 2))
        
        delta = delta + ds_prediction
        prediction_shape = input["pts_initial"] + delta
        output["prediction"] = prediction_shape

    return output
