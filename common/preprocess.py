"""Preprocess."""

import numpy as np
import menpo
from menpofit.fitter import align_shape_with_bounding_box
from menpodetect.opencv import detect
import common.utils
import face_alignment.utils


__DETECTOR__ = detect.load_opencv_frontal_face_detector()   


class Preprocess(object):
  """Preprocess."""

  def __init__(self, params, detection=False, **inputs):
    self.params = params    
    self.inputs = inputs
    self.detection = detection
    self.__output_keys = sorted(params.dataset.keys())    
    self.transformation = []


  def get_transformation(self):
    t = self.transformation[0]
    for t_ in self.transformation[1:]:
      t = t.compose_after(t_)
    
    return t


  def get_image(self):
    image = self.inputs["image"]
    
    if type(image) == np.ndarray:
      pixels = np.transpose(image, (2, 0, 1))
    else:
      pixels = image.pixels

    if np.max(pixels) > 1:
      pixels = pixels / 255.

    pixels = np.float32(pixels)
    self.im = menpo.image.Image(pixels)


  def get_face(self):
    if "pts_2d" in self.inputs.keys():
      pts_2d = self.inputs["pts_2d"].reshape((-1, 2))
      pts_2d = menpo.shape.pointcloud.PointCloud(pts_2d)
      self.im.landmarks["pts_2d"] = pts_2d

    else:
      h, w = self.im.shape
      pts_2d = np.array([
        [0, 0],
        [h, 0],
        [h, w],
        [0, w]
      ])
      
      pts_2d = menpo.shape.pointcloud.PointCloud(pts_2d)

      if self.detection:
        try:
          pts_2d = __DETECTOR__(self.im)[0]
        except IndexError:
          pass

    self.im.landmarks["bbox"] = pts_2d.bounding_box()    


  def get_initial_pts(self):
    reference_shape = face_alignment.utils.REFERENCE_SHAPE
    reference_shape = menpo.shape.pointcloud.PointCloud(reference_shape)
    self.im.landmarks["pts_initial"] = align_shape_with_bounding_box(
      reference_shape,
      self.im.landmarks["bbox"]
    )


  def crop(self, p=0.):
    pointcloud = self.im.landmarks["bbox"]
    boundary = p * np.min(pointcloud.range())
    (y_min, x_min), (y_max, x_max) = pointcloud.bounds(boundary=boundary)    
    self.im.landmarks["bbox"] = menpo.shape.pointcloud.PointCloud(np.array([
      [y_min, x_min],
      [y_max, x_min],
      [y_max, x_max],
      [y_min, x_max]
    ])).bounding_box()

    self.im, t = self.im.crop_to_landmarks(
      group="bbox",
      return_transform=True
    )

    self.transformation.append(t)


  def resize(self):
    width, height, n_channels = self.params.dataset["image"].shape

    self.im, t = self.im.resize([width, height], return_transform=True)
    self.transformation.append(t)

    if n_channels == 1:
      try:
        self.im = self.im.as_greyscale()
      except:
        assert self.im.n_channels == 1

    elif n_channels == 3:
      self.im = common.utils.to_rgb(self.im)


  def tfslim_preprocess(self):
    assert np.max(self.im.pixels) <= 1.
    assert np.min(self.im.pixels) >= 0.

    self.im.pixels -= 0.5
    self.im.pixels *= 2.
  
    
  def make_pipeline(self, detection=False):
    output = dict()
    
    pipelines = self()
    if not isinstance(pipelines[0], list):
      pipelines = [pipelines]

    for pipeline in pipelines:
      for operation in pipeline:
        operation()

    for key in self.__output_keys:
      shape = self.params.dataset[key].shape
      dtype = self.params.dataset[key].dtype

      try:
        output[key] = self.inputs[key]
      except KeyError:
        output[key] = np.zeros(shape, dtype=dtype)

      try:
        if "image" in key:
          output[key] = self.im.pixels.transpose((1, 2, 0))

        elif "pts" in key:
          output[key] = self.im.landmarks[key].points

      except KeyError:
        output[key] = np.zeros(shape, dtype=dtype)

      output[key] = output[key].astype(dtype)      

    return output