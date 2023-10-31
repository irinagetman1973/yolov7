import sys
sys.path.append('./yolov7') 

import singleinference_yolov7 as singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
from PIL import Image
from io import BytesIO
import os
import logging
import requests
from utils.general import check_img_size
from io import BytesIO
import numpy as np
import cv2

# from config import DETECTION_MODEL_LIST_V7, DETECTION_MODEL_DIR_V7, YOLOv7, YOLOv7_Champion, YOLOv7_e6, YOLOv7_w6, YOLOv7x
from pathlib import Path



class YOLOv7Wrapper(SingleInference_YOLOV7):

      DETECTION_MODEL_DIR_V7 = Path('C:/Users/irina/capstone/weights') / 'detection'
      MODELS = {
            "yolov7.pt": DETECTION_MODEL_DIR_V7 / "yolov7.pt",
            "v7_champion.pt": DETECTION_MODEL_DIR_V7 / "v7_champion.pt",
            "yolov7-e6.pt": DETECTION_MODEL_DIR_V7 / "yolov7-e6.pt",
            "yolov7-w6.pt": DETECTION_MODEL_DIR_V7 / "yolov7-w6.pt",
            "yolov7x.pt": DETECTION_MODEL_DIR_V7 / "yolov7x.pt"
      }

      def __init__(self, model_name, img_size=(640, 640)):
            self.model_name = model_name
            self.model_path = self.get_model_path(model_name)
            self.stride = 32
            self.bboxes = []
            self.confidences = []
            # Adjust the width and height of the image size
            width, height = img_size
            adjusted_width = check_img_size(width, s=self.stride)
            adjusted_height = check_img_size(height, s=self.stride)
            
            self.img_size = (adjusted_width, adjusted_height)
            
            # Initializing the super class with the required parameters
            super().__init__(self.img_size, self.model_path, path_img_i='None', device_i='cpu', conf_thres=0.25, iou_thres=0.5)
            
            # Load the YOLOv7 model
            self.load_model()


      def get_model_path(self, model_name):
            return self.MODELS.get(model_name) or self.raise_error(model_name)

      @staticmethod
      def raise_error(model_name):
            raise ValueError(f"Model {model_name} not recognized.")
      
      def read_image_from_path(self, image_path: str):
            """
            Reads an image from the given file path and prepares it for detection.
            
            Parameters:
            - image_path (str): The path to the image file to be read.
            """
            self.read_img(image_path)
      
      
      @staticmethod
      def non_max_suppression(boxes, scores, threshold=0.5):
            if not boxes:
                  return []

            boxes = np.array(boxes)

            # pick the coordinates of the bounding boxes
            x1 = boxes[:,0]
            y1 = boxes[:,1]
            x2 = boxes[:,2]
            y2 = boxes[:,3]

            # compute the area of the bounding boxes
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            

            scores = np.array(scores)

            # sort the bounding boxes by their scores
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                  i = order[0]
                  keep.append(i)
                  
                  # find the coordinates for the intersection of the current box and the rest
                  xx1 = np.maximum(x1[i], x1[order[1:]])
                  yy1 = np.maximum(y1[i], y1[order[1:]])
                  xx2 = np.minimum(x2[i], x2[order[1:]])
                  yy2 = np.minimum(y2[i], y2[order[1:]])

                  w = np.maximum(0.0, xx2 - xx1 + 1)
                  h = np.maximum(0.0, yy2 - yy1 + 1)

                  # compute the ratio of overlap
                  overlap = (w * h) / (areas[order[1:]] + areas[i] - w * h)

                  # delete all indexes from the index list that have overlap > threshold
                  inds = np.where(overlap <= threshold)[0]
                  order = order[inds + 1]

            return keep


      def detect_and_draw_boxes_from_np(self, img_np: np.ndarray,confidence_threshold: float = 0.5):
            """
            Detect objects in the provided numpy array image and draw bounding boxes.
            
            Parameters:
            - img_np (np.ndarray): The image data as a numpy array.
            
            Returns:
            - PIL.Image: Image with bounding boxes drawn.
            - list[str]: List of captions for the detections.
            """
            # Assuming that the image data is a numpy array, you can set it directly:
            self.im0 = img_np

            # Load the image to prepare it for inference
            self.load_cv2mat(self.im0)  # Passing the numpy image to load_cv2mat
            # Perform inference
            self.inference()

            if self.image is None:
                  raise ValueError("No image has been loaded or processed.")
                  
             # Iterate over bounding boxes and confidences, then draw
            for box, confidence in zip(self.bboxes, self.confidences):
                  if confidence < confidence_threshold:
                        continue  # Skip this box as it is below the confidence threshold
                  
                  x0, y0, x1, y1 = box
                  cv2.rectangle(self.image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)  # Green color box
                  cv2.putText(self.image, str(confidence), (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert the image with bounding boxes to a format suitable for returning
            self.img_screen = Image.fromarray(self.image).convert('RGB')
            
            # Create a caption for the detections
            captions = []
            if len(self.predicted_bboxes_PascalVOC) > 0:
                  for item in self.predicted_bboxes_PascalVOC:
                        name = str(item[0])
                        x1, y1, x2, y2 = map(int, item[1:5])  # Extracting and converting the coordinates to integers
                        conf = str(round(100 * item[-1], 2))
                        captions.append(f'name={name} coordinates=({x1}, {y1}, {x2}, {y2}) confidence={conf}%')

            # Reset the internal image representation (if necessary)
            self.image = None

            return self.img_screen, captions










            