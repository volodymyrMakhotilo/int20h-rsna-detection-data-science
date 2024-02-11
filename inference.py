model =

rez = model.predict(source = "/kaggle/input/rsna-pneumonia-detection-challenge-yolo/rsna/test/images", verbose = False, conf=0.15)

def get_confs(predict_vals):
    confs = []
    for val in predict_vals:
        patient_id = val.path.split('/')[-1]
        if (len(val.boxes.data) != 0):
            for i in range(len(val.boxes.xywhn)):
                confs.append(np.round(val.boxes.conf[i].cpu().numpy()*100)/100)
    return confs

def form_the_result(predict_vals, conf_n):
    result_ids = []
    result_labels = []
    for val in predict_vals:
        patient_id = val.path.split('/')[-1]
        result_ids.append(patient_id.split('.')[0])
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

confs = get_confs(rez)
conf_n = np.max(confs)
print(conf_n)
test = form_the_result(rez, conf_n)
display(test)
test.to_csv("submission.csv", index=False)
pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_sample_submission.csv')



