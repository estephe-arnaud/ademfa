"""Predictor."""

import abc
import os
import json
import numpy as np
import tensorflow as tf
import mapping
import common.utils
import cv2


class AbstractPredictor(object):
  """Abstract predictor class."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def predict(self, **inputs):
    return inputs

  @abc.abstractmethod
  def draw(self, image, prediction):
    return image

  @abc.abstractmethod
  def video(self):
    pass


class Predictor(AbstractPredictor):
  def __init__(self, ckpt_path):    
    # Hyperparameters
    model_dir = os.path.dirname(os.path.dirname(ckpt_path))
    d = json.load(open(os.path.join(model_dir, "hparams.json"), "rb"))
    params = common.utils.Hparams(d)
    self.params = params
    
    # Preprocess
    module_preprocess = mapping.PREPROCESS[params.project_name]
    self.preprocess = lambda **inputs: module_preprocess.Preprocess(
      params=params, 
      detection=True,
      **inputs
    )    
        
    # Model
    module_model = mapping.MODEL[params.project_name][params.model_name]    
    self.model = module_model.Model(params=params)
    self.model.restore(ckpt_path=ckpt_path)
    
    # IO
    self.input = self.model.input
    self.output = self.model.output


  def predict(self, **inputs):
    output = self.get_output(**inputs)
    prediction = output["prediction"]
    
    return prediction  


  def video(self):
    capture = cv2.VideoCapture(0)

    while True:
      _, image = capture.read()    
      
      try:
        prediction = self.predict(image=image)
        image = self.draw(image, prediction)
      except:
        continue

      cv2.imshow("", image)

      if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break


  def get_output(self, **inputs):
    with self.model.graph.as_default() as graph:
      self.__preprocess = self.preprocess(**inputs)
      inputs = self.__preprocess.make_pipeline()      

      feed_dict = {}
      for key, value in inputs.items():
        tensor = graph.get_tensor_by_name("{}:0".format(key))
        feed_dict[tensor] = value

      output_k = list(self.output.keys())
      output_t = list(self.output.values())        
      output_v = self.model.session.run(output_t, feed_dict)

      # Remove batch axis
      output = {k: v[0] for k, v in zip(output_k, output_v)}

    return output
