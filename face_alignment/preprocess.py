"""Face alignment preprocess."""

import functools
import common.preprocess


class Preprocess(common.preprocess.Preprocess):
  def __init__(self, **kwargs):
    super(Preprocess, self).__init__(**kwargs)

  def __call__(self):
    return [
      self.get_image,
      self.get_face,
      self.get_initial_pts,
      functools.partial(self.crop, p=0.3),
      self.resize,
      self.tfslim_preprocess
    ]