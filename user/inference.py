import numpy as np
from PIL import Image

from ultralytics import YOLO
from lightning import ModelCheckpoint, MyLightningModule

def process_patch_pair(cell_patch, tissue_patch, pair_id, meta_dataset):
    """This function detects the cells in the cell patch. Additionally
    the broader tissue context is provided. 

    NOTE: this implementation offers a dummy inference example. This must be
    updated by the participant.

    Parameters
    ----------
    cell_patch: np.ndarray[uint8]
        Cell patch with shape [1024, 1024, 3] with values from 0 - 255
    tissue_patch: np.ndarray[uint8] 
        Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
    pair_id: str
        Identification number of the patch pair
    meta_dataset: Dict
        Dataset metadata in case you wish to compute statistics

    Returns
    -------
        List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    """
    model_folder_name = 'glamourous_sweep_3'
    model_weights_path = f'./models/{model_folder_name}/weights/best.pt'
    
    # Getting the metadata corresponding to the patch pair ID
    meta_pair = meta_dataset[pair_id]

    #############################################
    #### YOUR INFERENCE ALGORITHM GOES HERE #####
    #############################################
    
    # Load the trained Yolo model
    model = YOLO(model_weights_path)
    results = model(cell_patch)
    boxes = results[0].boxes
    probs = boxes.conf.tolist()
    xywh = boxes.xywh.tolist()
    xs = [int(inner_list[0]) for inner_list in xywh]
    ys = [int(inner_list[1]) for inner_list in xywh]
    pred_class = boxes.cls.tolist()
    class_id = [cls + 1 for cls in pred_class]

    # The following is a dummy cell detection algorithm
    prediction = np.copy(cell_patch[:, :, 2])
    prediction[(cell_patch[:, :, 2] <= 40)] = 1
    xs, ys = np.where(prediction.transpose() == 1)
    class_id = [1] * len(xs) # Type of cell
    probs = [1.0] * len(xs) # Confidence score

    #############################################
    ####### RETURN RESULS PER SAMPLE ############
    #############################################

    # We need to return a list of tuples with 4 elements, i.e.:
    # - int: cell's x-coordinate in the cell patch
    # - int: cell's y-coordinate in the cell patch
    # - int: class id of the cell, either 1 (BC) or 2 (TC)
    # - float: confidence score of the predicted cell
    return list(zip(xs, ys, class_id, probs))

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        yolo_folder_name = 'glamourous_sweep_3'
        segformer_folder_name = 'TODO'
        
        yolo_weights_path = f'./models/{yolo_folder_name}/weights/best.pt'
        segformer_weights_path = f'./models/{segformer_folder_name}/weights/best.pt'
        segformer_checkpoint_callback = ModelCheckpoint(dirpath=segformer_weights_path)

        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
        
        # Load the trained Yolo model
        yolo_model = YOLO(yolo_weights_path)
        results = yolo_model(cell_patch)
        boxes = results[0].boxes
        probs = boxes.conf.tolist()
        xywh = boxes.xywh.tolist()
        xs = [int(inner_list[0]) for inner_list in xywh]
        ys = [int(inner_list[1]) for inner_list in xywh]
        pred_class = boxes.cls.tolist()
        # update classes
        yolo_pred_classes = [cls + 1 for cls in pred_class]


        # Load the SegFormer Model
        segformer_model = MyLightningModule.load_from_checkpoint(segformer_checkpoint_callback.best_model_path)
        segformer_model.eval()
        pred_mask = segformer_model(tissue_patch)

        # Zoom in on mask to corresponding location of cell patch
        # Convert the segmentation mask to a numpy array, TODO: necesarry?
        pred_mask_array = np.array(pred_mask)

        # Get the coordinates of the cell in the slice
        x_start_cell, y_start_cell = meta_pair["cell"]["x_start"], meta_pair["cell"]["y_start"]
        x_end_cell, y_end_cell = meta_pair["cell"]["x_end"], meta_pair["cell"]["y_end"]

        # Get the coordinates of the tissue in the slice
        x_start_tissue, y_start_tissue = meta_pair["tissue"]["x_start"], meta_pair["tissue"]["y_start"]
        x_end_tissue, y_end_tissue = meta_pair["tissue"]["x_end"], meta_pair["tissue"]["y_end"]
        
        # calculate the new x_start and y_start and x_end and y_end
        x_start = (x_start_cell - x_start_tissue) / (x_end_tissue - x_start_tissue) * 1024
        x_end = (x_end_cell - x_start_tissue) / (x_end_tissue - x_start_tissue) * 1024
        y_start = (y_start_cell - y_start_tissue) / (y_end_tissue - y_start_tissue) * 1024
        y_end = (y_end_cell - y_start_tissue) / (y_end_tissue - y_start_tissue) * 1024

        # crop the segmentation mask of the tissue to the image of the cell
        pred_mask_array = pred_mask_array[int(round(y_start,0)):int(round(y_end,0)), int(round(x_start,0)):int(round(x_end,0))]

        # converting the segmentation mask back to a pil image
        pred_mask = Image.fromarray(pred_mask_array)
        # resize the segmentation mask to the size of the image
        pred_mask = pred_mask.resize((1024,1025), Image.ANTIALIAS)
        # convert the segmentation mask back to a numpy array
        pred_mask_array = np.array(pred_mask)



        # Update the classes in the prediction numpy array based on the segmentation mask
        indices = (ys.astype(int), xs.astype(int))  # (y, x) indexing
        segformer_pred_classes = pred_mask_array[indices]
        
        # Keep the yolo class, where the area is unknown in the predicted segformer mask
        class_id = np.where(segformer_pred_classes == 255, yolo_pred_classes, segformer_pred_classes)




        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, class_id, probs))
