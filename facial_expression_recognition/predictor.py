"""Facial expression recognition predictor."""

import common.predictor
import common.utils
import facial_expression_recognition.utils


class Predictor(common.predictor.Predictor):
  def __init__(self, ckpt_path="./weights/facial_expression_recognition/model/model.ckpt"):
    super(Predictor, self).__init__(ckpt_path=ckpt_path)


  def predict(self, **inputs):
    output = self.get_output(**inputs)
    probabilities = output["probabilities"]

    labels = self.params.dataset.expression.labels
    prediction = dict(zip(labels, probabilities))

    return prediction


  def draw(self, image, prediction):
    bbox = self.__preprocess.im.landmarks["bbox"]
    t = self.__preprocess.get_transformation()
    bbox = t.apply(bbox)
    
    image = common.utils.draw_pts_2d(image, bbox)        
    image = facial_expression_recognition.utils.draw(image, prediction)

    return image
