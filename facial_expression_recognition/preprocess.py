"""Facial expression recognition preprocess."""

import functools
import common.preprocess


class Preprocess(common.preprocess.Preprocess):
  def __init__(self, **kwargs):
    super(Preprocess, self).__init__(**kwargs)

  def __call__(self):    
    return [
      self.get_image,
      self.get_face,
      functools.partial(self.crop, p=0.),
      self.resize
    ]