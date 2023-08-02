# News

follow our current progress on the official leadership board (currently 7th place and 5th team as of 21.07.2023): https://ocelot2023.grand-challenge.org/evaluation/challenge/leaderboard/

# Cell Detection

Welcome to the repository for Cell Detection! This project aims to detect cell centers and differentiate between tumor and background cells using YOLOv8 for detection and classification, and Segformer for segmenting the tumor and background areas to postprocess the cells.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Introduction

This repository contains the implementation for the Cell Detection project. The primary goal of this project is to detect cell centers and classify them into tumor and background cells. To achieve this, we leverage the YOLOv8 architecture for the detection and classification tasks, and the Segformer model for segmenting the tumor and background areas to further process the detected cells.

## Getting Started

To get started with the project, follow these instructions:

1. Clone the repository to your local machine: `git clone https://github.com/your_username/your_repository.git`.
2. Install the required dependencies by running: `pip install -r requirements.txt`.
3. Request access and download the dataset from https://zenodo.org/record/7844149.
4. Structure the dataset as follows:
    - dataset
        - train
            - csv
            - images
            - labels
            - segmentations
        - valid
            - csv
            - images
            - labels
            - segmentations
            - tissue_images
            - predictions
        data.yaml
        metadata.json


## Usage

To reproduce the results of this project, follow these steps:

1. **Request Model Weights**: Due to their large size, the model weights for YOLOv8 and Segformer are not included in this repository. Please contact the creators of this GitHub repository to request the necessary model weights.
2. Once you have obtained the model weights, place them in the respective directories: `models/yolo/modelname/weights` for YOLOv8 and `models/segformer/modelname/weights` and `models/segformer_extractor/modelname/weights` for Segformer.
3. Create a config_yourname.yaml file to set your configurations.
4. Run the `inference.py` script in the `pipeline` folder to perform inference on your data. Make sure to adapt the "config_yourname.yaml" filename in inference.py.
5. To visualize your predictions use the prediction_inspection.ipynb script in the `utils` folder.


