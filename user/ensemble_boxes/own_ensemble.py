
import os
import numpy as np
import pandas as pd
import csv
from ensemble_boxes import *
from pipeline_functions import *

def listBoxes(list_csv_parent_paths, width=30, height=30):

    data = {}

    for parent_path in list_csv_parent_paths:
        files = os.listdir(parent_path)
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(parent_path, file)
                filename = os.path.splitext(file)[0]

                with open(file_path, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    rows = np.array(list(reader))

                if rows.size > 0:
                    x_center = rows[:, 0].astype(float)
                    y_center = rows[:, 1].astype(float)
                    class_name = rows[:, 2].astype(float).astype(int)
                    probability = rows[:, 3].astype(float)

                    xmin = ((x_center - width / 2)/1024).astype(float)
                    ymin = ((y_center - height / 2)/1024).astype(float)
                    xmax = ((x_center + width / 2)/1024).astype(float)
                    ymax = ((y_center + height / 2)/1024).astype(float)

                    # Crop the values to be between 0 and 1
                    xmin = np.clip(xmin, 0, 1)
                    ymin = np.clip(ymin, 0, 1)
                    xmax = np.clip(xmax, 0, 1)
                    ymax = np.clip(ymax, 0, 1)

                    boxes_list = np.column_stack((xmin, ymin, xmax, ymax))

                    if filename not in data:
                        data[filename] = [[boxes_list.tolist()], [probability.tolist()], [class_name.tolist()]]
                    else:
                        data[filename][0].append(boxes_list.tolist())
                        data[filename][1].append(probability.tolist())
                        data[filename][2].append(class_name.tolist())
                else:
                    data[filename] = [[[]],[[]],[[]]]

    return [(filename, rows) for filename, rows in data.items()]

def ensemble(CFG, subfolder, models, weights=None, iou=0.5):

    # Path to the folders containing the labels
    labels_path = os.path.join(CFG.directory, 'datasets\\valid\\labels')

    # Path to GT Json Prediction Files
    ground_truth_json_path  =  os.path.join(labels_path, 'cell_ground_truth_valid.json')# Path to the folders containing the labels
    labels_path = os.path.join(CFG.directory, 'datasets\\valid\\labels')

    # Path to GT Json Prediction Files
    ground_truth_json_path  =  os.path.join(labels_path, 'cell_ground_truth_valid.json')

    model_folder_paths = [os.path.join(CFG.directory, f'models\\{model}\\{subfolder}') for model in models]

    data = listBoxes(model_folder_paths)

    # create subfolder, to save the new predictions
    ensemble_path = os.path.join(CFG.directory, "ensembles")
    if os.path.exists(ensemble_path):
        # Join the prefixes with a hyphen as the separator
        ensemble_name = '-'.join([name[:3] for name in models])
        ensemble_folder_path = os.path.join(ensemble_path, ensemble_name)
        if not os.path.exists(ensemble_folder_path):
            os.makedirs(ensemble_folder_path)
    else:
        raise FileNotFoundError(f"Directory '{ensemble_path}' does not exist.")

    # Output directory where predictions are saved as csv file
    preds_output_path = os.path.join(ensemble_folder_path, subfolder)
    if not os.path.exists(preds_output_path):
            os.makedirs(preds_output_path)
    # Output directory where predictions are saved as a json file
    # preds_output_path_json = os.path.join(ensemble_folder_path, f"{subfolder}.json")

    for filename, lists in data:
        boxes_list, scores_list, labels_list = lists[0], lists[1], lists[2]
        if len(boxes_list) != 3:
            print("WARANING: Length of boxes at file {0}. Probably this file has no annotations".format(filename))
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou, conf_type="absent_model_aware_avg") # absent_model_aware_avg | box_and_model_avg | avg | max

        # Check if the lists are empty
        if not boxes.tolist() or not scores.tolist() or not labels.tolist():
            # Create an empty CSV file
            with open(os.path.join(preds_output_path, f"{filename}.csv"), 'w', newline=''):
                pass
        else:
            # Write the data to the CSV file
            with open(os.path.join(preds_output_path, f"{filename}.csv"), 'w', newline='') as file:
                writer = csv.writer(file)
                for box, score, label in zip(boxes, scores, labels):
                    x_center = int((box[0] + box[2]) / 2 * 1024)
                    y_center = int((box[1] + box[3]) / 2 * 1024)
                    row = [x_center, y_center, int(label), score]
                    writer.writerow(row)


    convert_csv_to_json(preds_output_path, ensemble_folder_path, subfolder)

    preds_json_path = os.path.join(ensemble_folder_path, f"{subfolder}_predictions_valid.json")
    scores = evaluation(preds_json_path ,ground_truth_json_path)
    # with open(f"{ensemble_folder_path}\\{subfolder}prediction_scores.txt", "w") as file:
    #     # Redirect the print output to the file
    #     print(scores, file=file)
    return scores






