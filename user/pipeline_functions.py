""" This is Libary for all created function necesarry to run a full 
evaluation test of a specific model

Author: Luke Voss
"""
import os
import csv
import json
import glob
import re

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from omegaconf import OmegaConf

### These are fixed, don't change!!
DISTANCE_CUTOFF = 15
CLS_IDX_TO_NAME = {1: "BC", 2: "TC"}


def generate_predictions(model, data_directory, output_directory):
    """
    This function generates .csv predictions given an specific model

    Keyword arguments:
    data directory -- path to folder containing images that should be predicted
    output_directory -- path to folder where prediction .csv shoul be created
    """
    
    # Create the folder if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over the images in the folder
    for filename in os.listdir(data_directory):
        if filename.endswith('.jpg'):
            
            # Load and preprocess the image
            image_path = os.path.join(data_directory, filename)
            results = model(image_path)
            boxes = results[0].boxes
            pred_confidence = boxes.conf.tolist()
            xywh = boxes.xywh.tolist()
            x_coordinates = [int(inner_list[0]) for inner_list in xywh]
            y_coordinates = [int(inner_list[1]) for inner_list in xywh]
            pred_class = boxes.cls.tolist()

            # Combine the data into a list of tuples
            data = list(zip(x_coordinates, y_coordinates, [cls + 1 for cls in pred_class], pred_confidence))

            # Save the data to a CSV file
            file_path = os.path.join(output_directory, filename[:-4] + ".csv")
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)

def postprocessing(predictions_folder, segmentations_folder, metadata_json_filepath, output_folder):
    """
    This function corrects raw predictions by changing class labels according to segmentations masks
    In this preliminary version, the predicted classes are simply overwritten

    Parameters:
    ------------
    predictions_folder: str
        path to folder containing raw prediction files in form of .csv files
    segmentations_folder: str
        path to folder contatining segmentation masks
    metadata_json_filepath: str
        path to metadata.json containing information about the location of the image in the segmentation mask/surrounding tissue
    output_folder: str
        path where corrected predictions should be saved
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the JSON file
    with open(metadata_json_filepath) as f:
        data = json.load(f)

    seg_dir = segmentations_folder

    # Loop over files in predictions
    for filename in os.listdir(predictions_folder):
        
        # Check if the file is an image file
        if filename.endswith('.csv'):
            # Construct the paths to the image, annotations, and predictions files
            predictions_path = os.path.join(predictions_folder, filename)
            segmentation_path = os.path.join(seg_dir, f'{filename[:-4]}.png')

            # Iterate over each sample in the JSON file
            sample_data = data['sample_pairs'][str(filename[:-4])]

            # Open the segmentation mask using PIL
            seg = Image.open(segmentation_path)

            # Convert the segmentation mask to a numpy array
            seg_array = np.array(seg)

            # Get the coordinates of the cell in the slice
            x_start_cell, y_start_cell = sample_data["cell"]["x_start"], sample_data["cell"]["y_start"]
            x_end_cell, y_end_cell = sample_data["cell"]["x_end"], sample_data["cell"]["y_end"]

            # Get the coordinates of the tissue in the slice
            x_start_tissue, y_start_tissue = sample_data["tissue"]["x_start"], sample_data["tissue"]["y_start"]
            x_end_tissue, y_end_tissue = sample_data["tissue"]["x_end"], sample_data["tissue"]["y_end"]
            
            # calculate the new x_start and y_start and x_end and y_end
            x_start = (x_start_cell - x_start_tissue) / (x_end_tissue - x_start_tissue) * 1024
            x_end = (x_end_cell - x_start_tissue) / (x_end_tissue - x_start_tissue) * 1024
            y_start = (y_start_cell - y_start_tissue) / (y_end_tissue - y_start_tissue) * 1024
            y_end = (y_end_cell - y_start_tissue) / (y_end_tissue - y_start_tissue) * 1024

            # crop the segmentation mask of the tissue to the image of the cell
            seg_array = seg_array[int(round(y_start,0)):int(round(y_end,0)), int(round(x_start,0)):int(round(x_end,0))]

            # converting the segmentation mask back to a pil image
            seg = Image.fromarray(seg_array)
            # resize the segmentation mask to the size of the image
            seg = seg.resize((1024,1025), Image.ANTIALIAS)
            # convert the segmentation mask back to a numpy array
            seg_array = np.array(seg)

            # correct the prediction class depending on the position in the segmentation mask
            predictions = pd.read_csv(predictions_path, names=['x', 'y', 'class', 'confidence'])

            # Extract the columns from the DataFrame
            x_coordinates = predictions['x'].values
            y_coordinates = predictions['y'].values
            classes = predictions['class'].values
            confidences = predictions['confidence'].values


            # Update the classes in the prediction numpy array based on the segmentation mask
            indices = (y_coordinates.astype(int), x_coordinates.astype(int))  # (y, x) indexing
            updated_classes = seg_array[indices]
            
            # Keep the old class, where the area is unknown in the segmentation mask
            # updated_classes = np.where(updated_classes == 255, classes, updated_classes)

            # Create a condition to identify elements in new_classes that are not 1 or 2
            condition = (updated_classes != 1) & (updated_classes != 2)

            # Update the classes array based on the condition
            updated_classes = np.where(condition, classes, updated_classes)

            prediction_array = np.column_stack((x_coordinates, y_coordinates, updated_classes, confidences))
            # Save the updated prediction array as a new CSV file
            formats = ['%d', '%d', '%d', '%.4f']  # Formats for the first two columns are integers, and the last column is a float with 4 decimals
            
            np.savetxt(output_folder+"/"+filename, prediction_array, delimiter=',', fmt=formats)

def convert_csv_to_json(csv_file_path, output_path):
    """ Convert csv annotations into a single JSON and save it,
        to match the format with the algorithm submission output.

    Parameters:
    -----------
    csv_file_path: str
        path to the csv files. (e.g. /home/user/ocelot2023_v0.1.1)
    output_path: str
        path to where the json is going to be saved. (e.g. /home/user/ocelot2023_v0.1.1)
    """

    pred_paths = sorted(glob.glob(f"{csv_file_path}/*.csv"))
    num_images = len(pred_paths)

    pred_json = {
        "type": "Multiple points",
        "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }
    
    for idx, pred_path in enumerate(pred_paths):
        with open(pred_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c, conf = line.split(",")
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": float(conf),
            }
            pred_json["points"].append(point)

    with open(f"{output_path}\\predictions_valid.json", "w") as g:
        json.dump(pred_json, g)
        print(f"JSON file saved in {output_path}\predictions.json")

###########################OCELOT UTILS##########################################

def _check_validity(inp):
    """ Check validity of algorithm output.

    Parameters
    ----------
    inp: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    """
    for cell in inp:
        assert sorted(list(cell.keys())) == ["name", "point", "probability"]
        assert re.fullmatch(r'image_[0-9]+', cell["name"]) is not None
        assert type(cell["point"]) is list and len(cell["point"]) == 3
        assert type(cell["point"][0]) is int and 0 <= cell["point"][0] <= 1023
        assert type(cell["point"][1]) is int and 0 <= cell["point"][1] <= 1023
        assert type(cell["point"][2]) is int and cell["point"][2] in (1, 2)
        #print(type(float(cell["probability"])))
        assert 0.0 <= float(cell["probability"]) <= 1.0
        assert type(cell["probability"]) is float and 0.0 <= float(cell["probability"]) <= 1.0


def _convert_format(pred_json, gt_json, num_images):
    """ Helper function that converts the format for easy score computation.

    Parameters
    ----------
    pred_json: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    
    gt_json: List[Dict]
        List of cell ground-truths, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is always 1.0.
    
    num_images: int
        Number of images.
    
    Returns
    -------
    pred_after_convert: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    
    gt_after_convert: List[List[Tuple(int, int, int, float)]]
        List of GT, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls, prob (always 1.0).
    """
    
    pred_after_convert = [[] for _ in range(num_images)]
    for pred_cell in pred_json:
        x, y, c = pred_cell["point"]
        prob = pred_cell["probability"]
        img_idx = int(pred_cell["name"].split("_")[-1])
        pred_after_convert[img_idx].append((x, y, c, prob))

    gt_after_convert = [[] for _ in range(num_images)]
    for gt_cell in gt_json:
        x, y, c = gt_cell["point"]
        prob = gt_cell["probability"]
        img_idx = int(gt_cell["name"].split("_")[-1])
        gt_after_convert[img_idx].append((x, y, c, prob))
    
    return pred_after_convert, gt_after_convert


def _preprocess_distance_and_confidence(pred_all, gt_all):
    """ Preprocess distance and confidence used for F1 calculation.

    Parameters
    ----------
    pred_all: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.

    gt_all: List[List[Tuple(int, int, int)]]
        List of GTs, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls.

    Returns
    -------
    all_sample_result: List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.
    """

    all_sample_result = []

    for pred, gt in zip(pred_all, gt_all):
        one_sample_result = {}

        for cls_idx in sorted(list(CLS_IDX_TO_NAME.keys())):
            pred_cls = np.array([p for p in pred if p[2] == cls_idx], np.float32)
            gt_cls = np.array([g for g in gt if g[2] == cls_idx], np.float32)
            
            if len(pred_cls) == 0:
                distance = np.zeros([0, len(gt_cls)])
                confidence = np.zeros([0, len(gt_cls)])
            else:
                pred_loc = pred_cls[:, :2].reshape([-1, 1, 2])
                if len(gt_cls) == 0:
                    distance = np.zeros([len(pred_cls), 0])
                else: 
                    gt_loc = gt_cls[:, :2].reshape([1, -1, 2])
                    distance = np.linalg.norm(pred_loc - gt_loc, axis=2)
                confidence = pred_cls[:, 2]

            one_sample_result[cls_idx] = (distance, confidence)

        all_sample_result.append(one_sample_result)

    return all_sample_result


def _calc_scores(all_sample_result, cls_idx, cutoff):
    """ Calculate Precision, Recall, and F1 scores for given class 
    
    Parameters
    ----------
    all_sample_result: List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.

    cls_idx: int
        1 or 2, where 1 and 2 corresponds Tumor (TC) and Background (BC) cells, respectively.

    cutoff: int
        Distance cutoff that used as a threshold for collecting candidates of 
        matching ground-truths per each predicted cell.

    Returns
    -------
    precision: float
        Precision of given class

    recall: float
        Recall of given class

    f1: float
        F1 of given class
    """
    
    global_num_gt = 0
    global_num_tp = 0
    global_num_fp = 0

    for one_sample_result in all_sample_result:
        distance, confidence = one_sample_result[cls_idx]
        num_pred, num_gt = distance.shape
        assert len(confidence) == num_pred

        sorted_pred_indices = np.argsort(-confidence)
        bool_mask = (distance <= cutoff)

        num_tp = 0
        num_fp = 0
        for pred_idx in sorted_pred_indices:
            gt_neighbors = bool_mask[pred_idx].nonzero()[0]
            if len(gt_neighbors) == 0:  # No matching GT --> False Positive
                num_fp += 1
            else:  # Assign neares GT --> True Positive
                gt_idx = min(gt_neighbors, key=lambda gt_idx: distance[pred_idx, gt_idx])
                num_tp += 1
                bool_mask[:, gt_idx] = False

        assert num_tp + num_fp == num_pred
        global_num_gt += num_gt
        global_num_tp += num_tp
        global_num_fp += num_fp
        
    precision = global_num_tp / (global_num_tp + global_num_fp + 1e-7)
    recall = global_num_tp / (global_num_gt + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluation(predictions_json_path, ground_truth_json_path, model_folder_path):
    """ Calculate mF1 score and save scores.

    Returns
    -------
    float
        A mF1 value which is average of F1 scores of BC and TC classes.
    """


    # Path where algorithm output is stored
    algorithm_output_path = predictions_json_path
    with open(algorithm_output_path, "r") as f:
        pred_json = json.load(f)["points"]
    
    # Path where GT is stored
    gt_path = ground_truth_json_path
    with open(gt_path, "r") as f:
        gt_json = json.load(f)["points"]
    with open(gt_path, "r") as f:
        num_images = json.load(f)["num_images"]

    # Check the validity (e.g. type) of algorithm output
    _check_validity(pred_json)
    _check_validity(gt_json)

    # Convert the format of GT and pred for easy score computation
    pred_all, gt_all = _convert_format(pred_json, gt_json, num_images)

    # For each sample, get distance and confidence by comparing prediction and GT
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)

    # Calculate scores of each class, then get final mF1 score
    scores = {}
    for cls_idx, cls_name in CLS_IDX_TO_NAME.items():
        precision, recall, f1 = _calc_scores(all_sample_result, cls_idx, DISTANCE_CUTOFF)
        scores[f"Pre/{cls_name}"] = precision
        scores[f"Rec/{cls_name}"] = recall
        scores[f"F1/{cls_name}"] = f1
    
    scores["mF1"] = sum([
        scores[f"F1/{cls_name}"] for cls_name in CLS_IDX_TO_NAME.values()
    ]) / len(CLS_IDX_TO_NAME)
    
    print(scores)

    # OUR CODE:
    # Open the file in write mode
    with open(f"{model_folder_path}\\prediction_scores.txt", "w") as file:
        # Redirect the print output to the file
        print(scores, file=file)
    
    #return scores


