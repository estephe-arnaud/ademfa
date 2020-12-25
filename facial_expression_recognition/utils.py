"""Facial expression recognition utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def draw(image, prediction):
  fig, axes = plt.subplots(1, 2, figsize=(20, 10))
  ax = axes[0]
  ax.imshow(image)
  ax.axis("off")
  ax = axes[1]
  n_labels = len(prediction)
  color = ["grey", "gold", "red", "orange", "blue", "green", "black"]
  ax.set_xticks(range(n_labels))
  ax.set_ylim((0., 1.))
  ax.set_yticks(np.arange(0., 1.1, 0.1))
  ax.bar(range(n_labels), list(prediction.values()), color=color)
  ax.set_xticklabels(list(prediction.keys()))
  ax.set_ylabel("Probability", fontsize=15)
  ax.tick_params(axis='both', which='major', labelsize=15)
  ax.tick_params(axis='both', which='minor', labelsize=15)
  ax.yaxis.grid(True)
  fig.canvas.draw()
  image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :-1]
  plt.close()

  return image  