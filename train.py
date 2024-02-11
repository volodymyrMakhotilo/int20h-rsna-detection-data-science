import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pydicom
import cv2
import os
from ultralytics import YOLO
import torch
import wandb

# Directories for raw and preprocessed data
TRAIN_DIR_RAW = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
TEST_DIR_RAW = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images'
TRAIN_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge-yolo/rsna/train"
VAL_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge-yolo/rsna/val"
TEST_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge-yolo/rsna/test"

# Class labels and bounding box coordinates
labels = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
boxes = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')


# Create directories for train, validation, and test sets if they don't exist
try:
    boxes = boxes.set_index('patientId')

    os.mkdir(TRAIN_DIR)
    os.mkdir(os.path.join(TRAIN_DIR, 'images'))
    os.mkdir(os.path.join(TRAIN_DIR, 'labels'))

    os.mkdir(VAL_DIR)
    os.mkdir(os.path.join(VAL_DIR, 'images'))
    os.mkdir(os.path.join(VAL_DIR, 'labels'))

    os.mkdir(TEST_DIR)
    os.mkdir(os.path.join(TEST_DIR, 'images'))
except:
    pass

# Transform and preprocess training images
def train_image_transform():

    # Create txt file with labels corresponding to image
    def create_label_file(DIR, patient_id):
        label_txt = ""
        for (x, y, width, height) in zip(xs, ys, widths, heights):
            label_txt += "\n" + "0" + " " + str(x) + " " + str(y) + " " + str(width) + " " + str(height)
        f = open(os.path.join(os.path.join(DIR, 'labels'), patient_id + '.txt'), "x")
        f.write(label_txt)
        f.close()

    for root, dirs, files in os.walk(TRAIN_DIR_RAW):
        for file in files:
            img = pydicom.dcmread(os.path.join(TRAIN_DIR_RAW, file)).pixel_array
            patient_id = file.replace('.dcm', '')
            box = boxes.loc[[patient_id]]

            # There are some samples with multiple boxes per image
            xs = box.x.unique()
            ys = box.y.unique()
            widths = box.width.unique()
            heights = box.height.unique()

            xs /= img.shape[1]
            widths /= img.shape[1]

            ys /= img.shape[0]
            heights /= img.shape[0]

            img = cv2.resize(img, (128, 128)) # RESIZE


            label = box.Target.unique()[0]

            if label == 0:
                if 0 == np.random.randint(low=0, high=20):
                                if 0 == np.random.randint(low=0, high=10):
                                    cv2.imwrite(os.path.join(os.path.join(VAL_DIR, 'images'), patient_id + '.jpg'), img)
                                else:
                                    cv2.imwrite(os.path.join(os.path.join(TRAIN_DIR, 'images'), patient_id + '.jpg'), img)
                else:
                    continue;

            if 0 == np.random.randint(low=0, high=10):
                cv2.imwrite(os.path.join(os.path.join(VAL_DIR, 'images'), patient_id + '.jpg'), img)
                create_label_file(VAL_DIR, patient_id)
            else:
                cv2.imwrite(os.path.join(os.path.join(TRAIN_DIR, 'images'), patient_id + '.jpg'), img)
                create_label_file(TRAIN_DIR, patient_id)

# Resize test images
def test_image_transform():
    for root, dirs, files in os.walk(TEST_DIR_RAW):
        for file in files:
            img = pydicom.dcmread(os.path.join(TEST_DIR_RAW, file)).pixel_array
            patient_id = file.replace('.dcm', '')
            img = cv2.resize(img, (128, 128)) # RESIZE
            cv2.imwrite(os.path.join(os.path.join(TEST_DIR, 'images'), patient_id + '.jpg'), img)

# YAML configuration for YOLO model training
yaml = f"""
        train: {TRAIN_DIR}
        val: {VAL_DIR}
        test: {TEST_DIR}
        nc: 1
        names: ["Poly"]
        """
# VAL


try:
    f = open("poly.yaml", "x")
except:
    f = open("poly.yaml", "w")
f.write(yaml)
f.close()

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Choose device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# Disable Weights & Biases logging
wandb.init(mode="disabled")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

# Train YOLO model
_ = model.train(data='poly.yaml', epochs=25, pretrained=True, imgsz = 128, device = '1', exist_ok=True, close_mosaic=5,
               erasing=0.0, scale=0.25, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, shear=0.1)