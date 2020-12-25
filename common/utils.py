"""Utils."""

import numpy as np
import menpo
import cv2
import matplotlib.pyplot as plt


class Hparams(dict):
  MARKER = object()

  def __init__(self, value=None):
    if value is None:
      pass
    elif isinstance(value, dict):
      for key in value:
        self.__setitem__(key, value[key])
    else:
      raise TypeError('Expected dict.')


  def __setitem__(self, key, value):
    if isinstance(value, dict) and not isinstance(value, Hparams):
      value = Hparams(value)
    super(Hparams, self).__setitem__(key, value)


  def __getitem__(self, key):
    found = self.get(key, Hparams.MARKER)
    if found is Hparams.MARKER:
      found = Hparams()
      super(Hparams, self).__setitem__(key, found)
    return found


  __setattr__, __getattr__ = __setitem__, __getitem__


def to_rgb(im):
  assert im.n_channels in [1, 3]

  if im.n_channels == 3:
    return im

  im.pixels = np.vstack([im.pixels] * 3)
  return im


def draw_pts_2d(image, pts_2d):
  fig = plt.figure()
  im = menpo.image.Image(image.transpose((2, 0, 1)))

  if not isinstance(pts_2d, list):
    pts_2d = [pts_2d]

  for i, pts_2d_i in enumerate(pts_2d):    
    im.landmarks["pts_2d_{}".format(i)] = pts_2d_i

    im.view_landmarks(
      group="pts_2d_{}".format(i),
      marker_size=3, 
      marker_face_colour="red", 
      marker_edge_colour="red",
      line_colour="red"
    )

  fig.canvas.draw()
  image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :-1]
  plt.close()

  return image
