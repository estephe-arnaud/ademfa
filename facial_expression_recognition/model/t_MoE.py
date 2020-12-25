"""Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import common.model
import common.ops as ops
from common.pretuned_model import vggface


class Model(common.model.Model):
  """Tree gated Mixture-of-Experts."""

  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs)


  def __call__(self, input):
    """Forward."""

    params = self.params
    output = {}
    
    with tf.compat.v1.variable_scope("vggface"):
      vggface_model = vggface.VGGFace(
        include_top=False, 
        input_tensor=input["image"]
      )

      fmaps = vggface_model.output

    with tf.compat.v1.variable_scope("gates"):
      x = tf.keras.layers.GlobalAveragePooling2D()(fmaps)
      x = tf.keras.layers.Flatten()(x)
      
      gates = ops.dense_layers(
        x,
        units=[params.model.predictor.n_block],
        activation=["sigmoid"],
        batch_norm=params.model.batch_norm
      )      

    with tf.compat.v1.variable_scope("FE_recognizer"):      
      with tf.compat.v1.variable_scope("features"):        
        fmaps = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(fmaps)
        features = tf.keras.layers.Flatten(name="features")(fmaps)

      with tf.compat.v1.variable_scope("predictor"):
        committee = []

        for i in range(params.model.predictor.n_block):
          with tf.compat.v1.variable_scope(str(i)):            
            p = ops.dense_layers(
              features,
              units=params.model.predictor.units,
              activation=params.model.predictor.activation,
              batch_norm=params.model.batch_norm
            )

            committee.append(p)

        committee = tf.stack(committee, axis=1)
        logits = ops.tree_op(decision=gates, leaves=committee) 

      output["logits"] = tf.identity(logits, name="logits")
      output["probabilities"] = tf.nn.softmax(logits, axis=1)
      output["confidence"] = tf.reduce_max(output["probabilities"], axis=1)
      output["prediction"] = tf.argmax(logits, axis=1, name="prediction") 

    return output