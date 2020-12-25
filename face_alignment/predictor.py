"""Face alignment predictor."""

import common.predictor
import common.utils
import menpo


class Predictor(common.predictor.Predictor):
  def __init__(self, ckpt_path="./weights/face_alignment/model/model.ckpt"):
    super(Predictor, self).__init__(ckpt_path=ckpt_path)


  def predict(self, **inputs):
    output = self.get_output(**inputs)
    prediction = output["prediction"]

    t = self.__preprocess.get_transformation()
    prediction = t.apply(prediction)

    return prediction


  def draw(self, image, prediction):
    bbox = self.__preprocess.im.landmarks["bbox"]
    t = self.__preprocess.get_transformation()
    bbox = t.apply(bbox)

    prediction = menpo.shape.pointcloud.PointCloud(prediction)
    image = common.utils.draw_pts_2d(image, [bbox, prediction])

    return image
