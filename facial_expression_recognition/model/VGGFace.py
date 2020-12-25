"""Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import common.model
import common.ops as ops
from common.pretuned_model import vggface


class Model(common.model.Model):
  """VGGFace."""

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

    with tf.compat.v1.variable_scope("FE_recognizer"):
      with tf.compat.v1.variable_scope("features"):        
        features = tf.keras.layers.Flatten(name="features")(fmaps)

      with tf.compat.v1.variable_scope("predictor"):
        logits = ops.dense_layers(
          features,
          units=params.model.predictor.units,
          activation=params.model.predictor.activation,
          batch_norm=params.model.batch_norm          
        )
      
      output["logits"] = tf.identity(logits, name="logits")
      output["probabilities"] = tf.nn.softmax(logits, axis=1)
      output["confidence"] = tf.reduce_max(output["probabilities"], axis=1)
      output["prediction"] = tf.argmax(logits, axis=1, name="prediction")         
    
    return output
