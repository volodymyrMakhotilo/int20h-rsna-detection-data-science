# int20h_test_task
## Overview
This project is an implementation of solution to Kaggle problem [RSNA pneumonia detection challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge). The task consists of two subtasks: classification (presence or absence of illness) and object detection (if the presence of the disease is suspected, it is necessary to localize the suspicious darkening on the chest x-ray).

YOLOv8-model (https://docs.ultralytics.com/ru/models/yolov8/) was choosen as the main model for current solution. YOLO model uses You Only Look Ones principal to extract all possible square bounds around object and then non-max suppression method to filter the bounding boxes. Thus, there is only one predicted bounding box for one object. YOLO model is known for its perfomanse due to only one run of the same image through a neural network (You Only Look Ones principal itself).  In our solution YOLOv8s pre-trained version was used as the balanced version between performance and learning speed, limited by Kaggle notebook and local hardware. 

## Data preprocessing 
Original data of competition has been changed for the increasing easy-to-use rate. dcm-extention was changed to jpg-extention. Images with no bound boxes (this type of images belongs to "Normal" or "No Lung Opacity / Not Normal" target classes) were removed from training. As the result, less data is required for processing and YOLO model can handle images with no objects without directly training on such pieces of data.

## Model characteristics
- Model name: YOLOv8s
- Number of epochs for training: 25
- Number of images for train: 5389 files
- Number of images for validation: 623 files
- Number of images for test: 3000 files
- Training image size: 128x128

## Instructions for local run
1. Clone the github repo to your local hardware.
2. Open the repo in IDE or other editor.
3. Check for library version requiremens in requiremens.txt, if something is missing then execute terminal command "pip install -r requirements.txt" in project folder.
4. Execute terminal command "python inference.py \<*your path to the .dcm test images*>\" in project folder.
5. Recieve/update filled submission.csv file in current folder.  
