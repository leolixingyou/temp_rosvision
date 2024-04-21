import os
from .utils.utils import BaseEngine
import numpy as np
import cv2
import time
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path,day_night):
        super(Predictor, self).__init__(engine_path)
        if day_night == 'day':
          self.n_classes = 18  # your model classes
          self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'green3_h', 'bus',
                              'red3_h', 'truck', 'yellow3_h', 'green4_h', 'red4_h', 'yellow4_h',
                              'redgreen4_h', 'redyellow4_h', 'greenarrow4_h', 'red_v', 'yellow_v', 'green_v']

        if day_night == 'night':
          self.n_classes = 13  # your model classes
          self.class_names = ['bicycle', 'bus', 'car', 'green3_h', 'green4_h', 'greenarrow4_h', 'motorcycle', 'red3_h', 'red4_h', 'redgreen4_h', 'traffic light', 'traffic sign', 'truck']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")

    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
      origin_img,box_result = pred.inference(img_path, conf=0.1, end2end=args.end2end)

      cv2.imwrite("%s" %args.output , origin_img)
    if video:
      pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam