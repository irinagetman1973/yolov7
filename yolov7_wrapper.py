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
import random
# from config import DETECTION_MODEL_LIST_V7, DETECTION_MODEL_DIR_V7, YOLOv7, YOLOv7_Champion, YOLOv7_e6, YOLOv7_w6, YOLOv7x
from pathlib import Path



class YOLOv7Wrapper(SingleInference_YOLOV7):

      DETECTION_MODEL_DIR_V7 = Path(__file__).parent.parent/ 'weights' / 'detection'
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
            print("Available models:", self.MODELS)
            print("Requested model name:", model_name)
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
                  color = self.colors[self.names.index(name)]
                 
                  # Draw a thicker rectangle for better visibility
                  cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=4)

                  # Prepare the text with class name and confidence
                  text = f"{name} {confidence:.2f}"
                  text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, thickness=2)[0]
                  text_x = x0
                  text_y = y0 - 10  # Adjust position to be above the rectangle

                  # Draw a filled rectangle for the text background
                  cv2.rectangle(image, (x0, y0 - text_size[1] - 10), (x0 + text_size[0], y0), color, thickness=cv2.FILLED)

                  # Put the text on top of the filled rectangle
                  cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), thickness=2)

                  # cv2.rectangle(self.image, (int(x0), int(y0)), (int(x1), int(y1)),(0, 255, 0), thickness=3)  # Green color box
                  
                  # cv2.putText(self.image, label, (int(x0), int(y0) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

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










            