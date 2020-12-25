"""Model."""

import tensorflow as tf


class Model(object):
  def __init__(self, params):   
    self.params = params
    self.graph = tf.Graph()

    with self.graph.as_default() as graph:
      config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
      )
            
      config.allow_soft_placement = True
      config.gpu_options.allow_growth = True
      session = tf.compat.v1.Session(
        config=config, 
        graph=graph
      )

      self.session = session

      with session.as_default():
        self.input = self._input_fn()
        self.output = self(self.input)


  def restore(self, ckpt_path):
    """Restore."""

    with self.graph.as_default():
      saver = tf.compat.v1.train.Saver()
      saver.restore(self.session, ckpt_path)


  def _input_fn(self):
    """Input."""

    dataset = {}

    for key in self.params.dataset.keys():
      shape = self.params.dataset[key].shape
      dtype = self.params.dataset[key].dtype
      
      tensor = tf.compat.v1.placeholder(
        dtype=dtype,
        shape=shape,
        name=key
      )

      # Add batch axis
      dataset[key] = tf.expand_dims(tensor, axis=0)

    return dataset          
