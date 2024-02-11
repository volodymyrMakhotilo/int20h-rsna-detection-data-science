from ultralytics import YOLO
import pydicom

import torch
import numpy as np
import pandas as pd
import cv2
import os
import sys



TEST_DIR_RAW = sys.argv[-1]
print(TEST_DIR_RAW)
TEST_DIR = os.path.join(os.getcwd(), 'test')

def define_folders():
    try:
        os.mkdir(TEST_DIR)
        os.mkdir(os.path.join(TEST_DIR, 'images'))
    except:
        pass

def test_image_transform():
        for file in os.listdir(TEST_DIR_RAW):
            img = pydicom.dcmread(os.path.join(TEST_DIR_RAW, file)).pixel_array
            patient_id = file.replace('.dcm', '')
            img = cv2.resize(img, (128, 128))
            cv2.imwrite(os.path.join(os.path.join(TEST_DIR, 'images'), patient_id + '.jpg'), img)

def get_confs(predict_vals):
    confs = []
    for val in predict_vals:
        if (len(val.boxes.data) != 0):
            for i in range(len(val.boxes.xywhn)):
                confs.append(np.round(val.boxes.conf[i].cpu().numpy()*100)/100)
    return confs

def form_the_result(predict_vals, conf_n):
    result_ids = []
    result_labels = []
    for val in predict_vals:
        patient_id = val.path.split('/')[-1]
        result_ids.append(patient_id.split('.')[0].split('\\')[-1])
        result_label = " "
        if (len(val.boxes.data) != 0):
            for i in range(len(val.boxes.xywhn)):
                original_bounds = (val.boxes.xywhn[i].cpu().numpy() * 1024).astype(np.int64)
                result_label += str(np.round(val.boxes.conf[i].cpu().numpy()*10)/10/conf_n) + " " + " ".join(map(str, original_bounds)) + " "
        result_labels.append(result_label)
    result_dataset = pd.DataFrame(columns = ["patientId", "PredictionString"])
    result_dataset["patientId"] = result_ids
    result_dataset["PredictionString"] = result_labels
    return result_dataset

define_folders()
test_image_transform()

model = YOLO('weights.pt')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

rez = model.predict(source =os.path.join(TEST_DIR, 'images'), verbose = False, conf=0.15)


confs = get_confs(rez)
conf_n = np.max(confs)

test = form_the_result(rez, conf_n)

test.to_csv("submission.csv", index=False)