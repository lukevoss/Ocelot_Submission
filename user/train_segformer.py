import os
from PIL import Image



import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip,  ColorJitter, RandomResizedCrop, RandomRotation
from matplotlib import pyplot as plt
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import lightning as L
from lightning.pytorch.loggers import WandbLogger
# from lightning.callbacks.early_stopping import EarlyStopping
# from lightning.callbacks.model_checkpoint import ModelCheckpoint

class SemanticSegmentationDataset(Dataset):
    """Tissue Dataset for (semantic) segmentation with SegFormer."""

    def __init__(self, image_dir, mask_dir, feature_extractor, both_augmentations=None, image_augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.both_augmentations = both_augmentations
        self.image_augmentations = image_augmentations
        self.classes_csv_file = os.path.join(self.mask_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i,l in enumerate(fid) if i !=0]
        self.id2label = {x[0]:x[1] for x in data}
        
        image_file_names = [f for f in os.listdir(self.image_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.mask_dir) if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.image_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.mask_dir, self.masks[idx]))

        # Data Augmentation
        if self.both_augmentations is not None:
            image, segmentation_map = self.both_augmentations(image, segmentation_map)

        if self.image_augmentations is not None:
            image = self.image_augmentations(image)

        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        return encoded_inputs

class SegformerFinetuner(L.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, metrics_interval=100, pretrained_model="nvidia/segformer-b3-finetuned-ade-512-512"):
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
        
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
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


if __name__ == "__main__":

    # argparser = argparse.ArgumentParser(description='Specify Hyper-parameters')
    # argparser.add_argument('--img_size', type=int, help='Image size to train with')
    #argparser.add_argument('--pretrained_model', type=float, help='Huggingface API Key for pretrained Models')

    # args = argparser.parse_args()
    

    wandb_logger = WandbLogger(project="SegFormer", log_model=True)
    
    IMAGE_SIZE = 512 #args.img_size
    EPOCHS = 300
    pretrained_model = "nvidia/segformer-b3-finetuned-ade-512-512"
    #smallest model: "nvidia/segformer-b0-finetuned-ade-512-512"
    #middle model: "nvidia/segformer-b3-finetuned-ade-512-512"
    #biggest model: nvidia/segformer-b5-finetuned-ade-640-640

    feature_extractor = SegformerFeatureExtractor.from_pretrained(pretrained_model)
    feature_extractor.size = IMAGE_SIZE 

    # Define augmentation transforms, these are applied to image AND segmentation mask
    both_augmentations = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # size: output size of crop, scale: min and max scale factors, ratio: fixed ration so no streching
        RandomResizedCrop(size=(512,512), scale=(0.5, 1.0),ratio=(1.0, 1.0)),
        #RandomRotation(degrees=(0, 180))
    ])

    # Define image augmentation transforms, these are applied only to the image
    image_augmentations = transforms.Compose([
        ColorJitter(0.05, 0.05, 0.05, 0.05)
    ])

    train_dataset = SemanticSegmentationDataset("./datasets/train/tissue_images/","./datasets/train/labels/segmentations/", feature_extractor)
    val_dataset = SemanticSegmentationDataset("./datasets/valid/tissue_images/","./datasets/valid/labels/segmentations/", feature_extractor)

    BATCH_SIZE = 1
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=3, prefetch_factor=8)

    segformer_finetuner = SegformerFinetuner(
    train_dataset.id2label, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    metrics_interval=10,
    pretrained_model=pretrained_model
    )

    # early_stop_callback = EarlyStopping(
    # monitor="val_loss", 
    # min_delta=0.00, 
    # patience=3, 
    # verbose=False, 
    # mode="min",
    # )

    # checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    trainer = L.Trainer(
        #default_root_dir="./models/",
        accelerator='gpu', 
        # callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=EPOCHS,
        check_val_every_n_epoch = 1,
        logger=wandb_logger
    )


    trainer.fit(segformer_finetuner)

    # res = trainer.test(ckpt_path="best", dataloaders=val_dataloader)

    ############# plotting ##################

    color_map = {
    0:(0,0,0),
    1:(255,0,0),
    }

    def prediction_to_vis(prediction):
        vis_shape = prediction.shape + (3,)
        vis = np.zeros(vis_shape)
        for i,c in color_map.items():
            vis[prediction == i] = color_map[i]
        return Image.fromarray(vis.astype(np.uint8))

    for batch in val_dataloader:
        images, masks = batch['pixel_values'], batch['labels']
        outputs = segformer_finetuner.model(images, masks)
            
        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1).cpu().numpy()
        masks = masks.cpu().numpy()

    
    f, axarr = plt.subplots(predicted.shape[0],2)
    for i in range(predicted.shape[0]):
        axarr[i,0].imshow(prediction_to_vis(predicted[i,:,:]))
        axarr[i,1].imshow(prediction_to_vis(masks[i,:,:]))

    # Get the plot as an image array
    fig_image = plt.gcf()
    fig_image.canvas.draw()
    plot_image = np.array(fig_image.canvas.renderer._renderer)

    # using tensors, numpy arrays or PIL images
    wandb_logger.log_image(key="validation set", images=[plot_image])


    # Finish logging
    wandb_logger.experiment.finish()