from PIL import Image
import os

import numpy as np
import torch
from torch import nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import lightning as L

from metrics.mean_iou.mean_iou import MeanIoU
from user.ensemble_boxes import *

# TODO uncomment for submission
# os.environ['TRANSFORMERS_OFFLINE']='1'
# os.environ['HF_DATASETS_OFFLINE']='1'


class SegformerFinetuner(L.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, metrics_interval=100, pretrained_model="nvidia/segformer-b2-finetuned-ade-512-512"):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.pretrained_model = pretrained_model
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.pretrained_model, 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True, #TODO True
        )
        #self.model.save_pretrained('./models/segformer/b3-finetuned-ade-512-512')
        
        self.train_mean_iou = MeanIoU()
        self.val_mean_iou = MeanIoU()
        # self.val_step_outputs = []
        
    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
            for k,v in metrics.items():
                self.log(k,v)
            
            self.log("test_loss", loss)

            return(metrics)
        else:
            return({'loss': loss})
    
    def validation_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]

        #self.val_step_outputs.extend(loss)

        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        self.log("val_loss", loss)
        
        return({'val_loss': loss})
    
    # def on_validation_epoch_end(self):
    #     metrics = self.val_mean_iou.compute(
    #           num_labels=self.num_classes, 
    #           ignore_index=255, 
    #           reduce_labels=False,
    #       )
        
    #     avg_val_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
    #     val_mean_iou = metrics["mean_iou"]
    #     val_mean_accuracy = metrics["mean_accuracy"]
        
    #     metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
    #     for k,v in metrics.items():
    #         self.log(k,v)

    #     return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata
        yolo_folder_name_1 = 'absurd-sweep-2'
        yolo_folder_name_2 = 'curious-sweep-4'
        yolo_folder_name_3 = 'fragrant-sweep-10'

        segformer_folder_name = 'sage-paper-37'
        
        yolo_weights_path_1 = f'./models/yolo/{yolo_folder_name_1}/weights/best.pt'
        yolo_weights_path_2 = f'./models/yolo/{yolo_folder_name_2}/weights/best.pt'
        yolo_weights_path_3 = f'./models/yolo/{yolo_folder_name_3}/weights/best.pt'

        segformer_checkpoint_path = f'./models/segformer/{segformer_folder_name}/model.ckpt'

        IMAGE_SIZE = 512
        id2label = {'1': 'background', '2': 'tumor'}


        self.yolo_model_1 = YOLO(yolo_weights_path_1)
        self.yolo_model_2 = YOLO(yolo_weights_path_2)
        self.yolo_model_3 = YOLO(yolo_weights_path_3)
        self.segformer_model = SegformerFinetuner.load_from_checkpoint(segformer_checkpoint_path, map_location={"cuda:0" : "cpu"}, id2label=id2label)
        self.segformer_model.eval()
        #self.feature_extractor = SegformerFeatureExtractor.from_pretrained('./models/segformer_extractor/preprocessor_config.json')
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')
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
        #TODO: Image size for each model might need to be adapted
        results_1 = self.yolo_model_1.predict(cell_patch_yolo, imgsz=640, conf=0.22, iou=0.4)
        results_2 = self.yolo_model_2.predict(cell_patch_yolo, imgsz=640, conf=0.22, iou=0.4)
        results_3 = self.yolo_model_3.predict(cell_patch_yolo, imgsz=640, conf=0.22, iou=0.4)
        results = [results_1, results_2, results_3]
        boxes = [result[0].boxes for result in results]
        probs = [box.conf.tolist() for box in boxes]
        xywh = [box.xywh.tolist() for box in boxes]
        xs = [[int(inner_list[0]) for inner_list in xywh_] for xywh_ in xywh]
        ys = [[int(inner_list[1]) for inner_list in xywh_] for xywh_ in xywh]
        yolo_pred_classes = [[int(element+1) for element in box.cls.tolist()] for box in boxes]


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
        indices = [(ys_, xs_) for ys_, xs_ in zip(ys,xs)]  # (y, x) indexing
        segformer_pred_classes = [pred_mask_array[indices_] for indices_ in indices]
        
        # Keep the yolo class, where the area is unknown in the predicted segformer mask
        class_id = [np.where(segformer_pred_classes_ == 255, yolo_pred_classes_, segformer_pred_classes_) for yolo_pred_classes_, segformer_pred_classes_ in zip(yolo_pred_classes, segformer_pred_classes)]

        # Initialize an empty list to store the updated class IDs
        # updated_class_ids = []

        # for yolo_pred_classes_, segformer_pred_classes_, probs_ in zip(yolo_pred_classes, segformer_pred_classes, probs):
        #     # Create a copy of segformer_pred_classes_ to avoid modifying the original list
        #     class_id_ = yolo_pred_classes_.copy()

        #     # Check if the probability is below 0.5
        #     # If yes, update the class_id_ using yolo_pred_classes_
        #     # Otherwise, keep the original segformer_pred_classes_
        #     for i in range(len(probs_)):
        #         if probs_[i] < 0.5:
        #             class_id_[i] = segformer_pred_classes_[i]

        #     # Append the updated class_id_ to the list of updated_class_ids
        #     updated_class_ids.append(class_id_)

        # class_id = updated_class_ids
        # weighted boxes fusion

        #TODO: what happens with empty image? return [[[]],[[]],[[]]] if no predictions
        boxes = [result[0].boxes for result in results]
        xywh = [box.xywh.tolist() for box in boxes]
        xs = [[int(inner_list[0]) for inner_list in xywh_] for xywh_ in xywh]
        ys = [[int(inner_list[1]) for inner_list in xywh_] for xywh_ in xywh]
    	# convert back to bounding boxes
        xmin = [((np.array(xs_) - 15)/1024).astype(float) for xs_ in xs]
        ymin = [((np.array(ys_) - 15)/1024).astype(float) for ys_ in ys]
        xmax = [((np.array(xs_) + 15)/1024).astype(float) for xs_ in xs]
        ymax = [((np.array(ys_) + 15)/1024).astype(float) for ys_ in ys]
        # Crop the values to be between 0 and 1
        xmin = [np.clip(xmin_, 0, 1) for xmin_ in xmin]
        ymin = [np.clip(ymin_, 0, 1) for ymin_ in ymin]
        xmax = [np.clip(xmax_, 0, 1) for xmax_ in xmax]
        ymax = [np.clip(ymax_, 0, 1) for ymax_ in ymax]

        boxes_list = [np.column_stack((xmin_, ymin_, xmax_, ymax_)) for xmin_, ymin_, xmax_, ymax_ in zip(xmin, ymin, xmax, ymax)]
        weights = [0.22565555705457563, 0.7895587213980079, 0.9291715002666618]
        iou = 0.35
        

        boxes, probs, labels = weighted_boxes_fusion(boxes_list, probs, class_id, weights=weights, iou_thr=iou, conf_type="absent_model_aware_avg")

        if not boxes.tolist() or not probs.tolist() or not labels.tolist():
            return list(zip([1],[1],[2],[0.1])) #TODO: remove this
        else:
            boxes_array = np.array(boxes)
            xs = np.int64((boxes_array[:, 0] + boxes_array[:, 2]) / 2 * 1024).tolist()
            ys = np.int64((boxes_array[:, 1] + boxes_array[:, 3]) / 2 * 1024).tolist()
            # xs = int((boxes[0] + boxes[2]) / 2 * 1024)
            # ys = int((boxes[1] + boxes[3]) / 2 * 1024)
            class_id = np.array(labels, dtype=int).tolist()

        #############################################
        ####### RETURN RESULTS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, class_id, probs.tolist()))
