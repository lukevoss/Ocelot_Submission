from PIL import Image

import numpy as np
from torch import nn
from transformers import SegformerFeatureExtractor
from ultralytics import YOLO

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata
        yolo_folder_name = 'glamourous_sweep_3'
        segformer_folder_name = 'earnest-sun-3'
        
        yolo_weights_path = f'./models/yolo/{yolo_folder_name}/weights/best.pt'
        segformer_checkpoint_path = f'./models/segformer/{segformer_folder_name}/model.ckpt'
        # segformer_checkpoint_callback = ModelCheckpoint(dirpath=segformer_checkpoint_path)

        pretrained_model = "nvidia/segformer-b3-finetuned-ade-512-512"
        IMAGE_SIZE = 512
        id2label = {'1': 'background', '2': 'tumor'}


        self.yolo_model = YOLO(yolo_weights_path)
        self.segformer_model = SegformerFinetuner.load_from_checkpoint(segformer_checkpoint_path, map_location={"cuda:0" : "cpu"}, id2label=id2label)
        self.segformer_model.eval()
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained('./models/segformer_extractor/preprocessor_config.json')# TODO: pretrained_model?
        self.feature_extractor.size = IMAGE_SIZE 



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
        
        

        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
        
        # Load the trained Yolo model
        cell_patch_yolo = cell_patch[...,::-1] #  RGB to BGR
        results = self.yolo_model.predict(cell_patch_yolo, imgsz=640, conf=0.23, iou=0.4)
        boxes = results[0].boxes
        probs = boxes.conf.tolist()
        xywh = boxes.xywh.tolist()
        xs = [int(inner_list[0]) for inner_list in xywh]
        ys = [int(inner_list[1]) for inner_list in xywh]
        yolo_pred_classes = [int(element+1) for element in boxes.cls.tolist()]


        # Load the SegFormer Model
        
        encoded_tissue = self.feature_extractor(images=tissue_patch, return_tensors="pt")
        logits = self.segformer_model.model(**encoded_tissue)
        logits = logits[0]

        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=(1024,1024), 
            mode="bilinear", 
            align_corners=False
        )

        pred_mask_array = upsampled_logits.argmax(dim=1).cpu().numpy()
        # Increase each element by one to match size
        pred_mask_array += 1
        pred_mask_array = np.squeeze(pred_mask_array)
        # pred_mask = feature_extractor.post_process_semantic_segmentation(logits)

        # Zoom in on mask to corresponding location of cell patch
        # Convert the segmentation mask to a numpy array, TODO: necesarry?
        # pred_mask_array = pred_mask.detach().numpy()

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
        pred_mask_array = pred_mask_array[int(round(y_start,0)):int(round(y_end,0)), int(round(x_start,0)):int(round(x_end,0))].astype(np.uint8)

        # converting the segmentation mask back to a pil image
        pred_mask = Image.fromarray(pred_mask_array)
        # resize the segmentation mask to the size of the image
        pred_mask = pred_mask.resize((1024,1024), Image.ANTIALIAS)
        # convert the segmentation mask back to a numpy array
        pred_mask_array = np.array(pred_mask)



        # Update the classes in the prediction numpy array based on the segmentation mask
        indices = (ys, xs)  # (y, x) indexing
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
