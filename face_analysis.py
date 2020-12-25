import os
import cv2
import argparse
import mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--camera", action="store_true")
    args = parser.parse_args()

    predictor = mapping.PREDICTOR[args.task].Predictor()

    if args.camera:
        predictor.video()

    if args.path:
        image = cv2.imread(args.path)

        prediction = predictor.predict(image=image)
        image = predictor.draw(image, prediction)

        filename, extension = os.path.splitext(args.path)
        filename += "_{}".format(args.task) 
        path = filename + extension
        
        cv2.imwrite(path, image)
  