import os
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from user.inference import Model
from user.eval_general import evaluation
from util import gcio




def main(CFG):

    # Path to the folders containing the images and metadata
    CELL_FPATH = Path(os.path.join(CFG.directory, 'datasets\\valid\\images'))
    TISSUE_FPATH = Path(os.path.join(CFG.directory, 'datasets\\valid\\tissue_images'))
    METADATA_FPATH = Path(os.path.join(CFG.directory, 'datasets\\metadata.json'))
    GROUND_TRUTH_LABELS = Path(os.path.join(CFG.directory, 'datasets\\valid\\labels\\cell_ground_truth_valid.json'))
    DETECTION_OUTPUT_PATH = Path(os.path.join(CFG.directory, 'test\\prediction.json'))
    

    cell_fpath = [os.path.join(CELL_FPATH, f) for f in os.listdir(CELL_FPATH) if ".jpg" in f]
    tissue_fpath = [os.path.join(TISSUE_FPATH, f) for f in os.listdir(TISSUE_FPATH) if ".jpg" in f]

    cell_patches = [np.array(Image.open(cell_fpath[i])) for i in range(len(cell_fpath))]
    tissue_patches = [np.array(Image.open(tissue_fpath[i])) for i in range(len(cell_fpath))]

    # Cell detection writer
    writer = gcio.DetectionWriter(DETECTION_OUTPUT_PATH)

    # Loading metadata
    meta_dataset = gcio.read_json(METADATA_FPATH)
    meta_dataset = meta_dataset['sample_pairs']

    # Instantiate the inferring model
    model = Model(meta_dataset)

    # NOTE: Batch size is 1
    for image_idx in range(40):
        pair_id = str(image_idx+1).zfill(3)
        pair_id_writer = str(image_idx).zfill(3)
        print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_classification = model(cell_patches[image_idx], tissue_patches[image_idx], pair_id)
        # Updating predictions
        writer.add_points(cell_classification, pair_id_writer)

    # Export the prediction into a json file
    writer.save()

    evaluation(DETECTION_OUTPUT_PATH, GROUND_TRUTH_LABELS)

    # # Loop over the files in the image directory
    # for filename in os.listdir(CELL_FPATH):
    #     # Check if the file is an image file
    #     if filename.endswith('.jpg'):
    #         # Construct the paths to the image, annotations, and predictions files
    #         image_path = os.path.join(CELL_FPATH, filename)

    #         # Call the plot_predictions function with the image and annotations paths, and the predictions path
    #         plot_predictions(image_path, GROUND_TRUTH_LABELS, DETECTION_OUTPUT_PATH)

    #     else:
    #         print(f'Error: annotations file not found for {filename}')


if __name__ == "__main__":

    # Load individual config file
    CFG = OmegaConf.load("config_luke.yaml")

    main(CFG)