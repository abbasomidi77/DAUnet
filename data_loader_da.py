import monai
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.data import Dataset, ArrayDataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld,\
                             RandAxisFlipd, RandGaussianNoised, RandGibbsNoised, RandSpatialCropd, Compose, \
                             CropForegroundd,AdjustContrastd)
import pandas as pd
import numpy as np
from monai.data.utils import pad_list_data_collate

source_transforms = Compose(
    [
        LoadImaged(keys=["img", "brain_mask"]),
        EnsureChannelFirstd(keys=["img",  "brain_mask"]),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandSpatialCropd(keys=["img","brain_mask"], roi_size=(112, 112, 112), random_size=False),
        #RandCropByPosNegLabeld(
        #    keys=["img", "brain_mask"],
        #    spatial_size=(64, 64, 64),
        #    label_key="brain_mask",
        #    pos = 0.9,
        #    neg=0.1,
        #    num_samples=1,
        #    image_key="img",
        #    image_threshold=-0.1
        #),
        #AdjustContrastd(keys=["img"], gamma=2.0),
        RandAxisFlipd(keys=["img", "brain_mask"], prob = 0.2),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6))
    ]
)

def threshold(x):
    # threshold at 1
    return x > 0.015


target_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["img"], source_key = "img", select_fn=threshold, margin=3),
        RandSpatialCropd(keys=["img"], roi_size=(112, 112, 112), random_size=False),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6)),
        RandAxisFlipd(keys=["img"], prob = 0.2)
    ]
)


def load_data(source_dev_images_csv, source_dev_masks_csv,
              target_dev_images_csv = None, batch_size = 1, val_split = 0.2, verbose = False):


    source_dev_images = pd.read_csv(source_dev_images_csv)
    source_dev_masks = pd.read_csv(source_dev_masks_csv)

    assert source_dev_images.size == source_dev_masks.size

    if target_dev_images_csv:
        target_dev_images = pd.read_csv(target_dev_images_csv)

    if verbose:
        print("Shape source images:", source_dev_images.shape)
        print("Shape source masks:",  source_dev_masks.shape)
        if target_dev_images_csv:
            print("Shape target images:", target_dev_images.shape)
        else:
            print("Target images CSV file path not provided")    
    
    
    indexes_source = np.arange(source_dev_images.shape[0])
    
    np.random.seed(100)  
    np.random.shuffle(indexes_source)
    
  
    source_dev_images = np.array(source_dev_images["filename"])[indexes_source]
    source_dev_masks = np.array(source_dev_masks["filename"])[indexes_source]
    
    ntrain_samples = int((1 - val_split)*indexes_source.size)
    source_train_images = source_dev_images[:ntrain_samples]
    source_train_masks = source_dev_masks[:ntrain_samples]

    source_val_images = source_dev_images[ntrain_samples:]
    source_val_masks = source_dev_masks[ntrain_samples:]

    if verbose:
        print("Source train set size:", source_train_images.size)
        print("Source val set size:", source_val_images.size)


    # Putting the filenames in the MONAI expected format - source train set
    filenames_train_source = [{"img": x, "brain_mask": y, "domain_label": 0.0}\
                              for (x,y) in zip(source_train_images, source_train_masks)]
       
    source_ds_train = monai.data.Dataset(filenames_train_source,
                                         source_transforms)

    source_train_loader = DataLoader(source_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here


    # Putting the filenames in the MONAI expected format - source val set
    filenames_val_source = [{"img": x, "brain_mask": y, "domain_label": 0.0}\
                              for (x,y) in zip(source_val_images, source_val_masks)]
       
    source_ds_val = monai.data.Dataset(filenames_val_source,
                                         source_transforms)
                                         
    source_val_loader = DataLoader(source_ds_val, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here



    # If there is not target domain data - return the source domain train and val datasets and loaders
    if not target_dev_images_csv:
        return source_ds_train, source_train_loader, source_ds_val, source_val_loader

    
    
    indexes_target = np.arange(target_dev_images.shape[0])
    np.random.seed(100)  
    np.random.shuffle(indexes_target)

    target_dev_images = np.array(target_dev_images["filename"])[indexes_target]
    
    ntrain_samples_target = int((1 - val_split)*indexes_target.size)
    target_train_images = target_dev_images[:ntrain_samples_target]
    
    target_val_images = target_dev_images[ntrain_samples_target:]

    if verbose:
        print("Traget train set size:", target_train_images.size)
        print("Target val set size:", target_val_images.size)


    # Putting the filenames in the MONAI expected format - target train set
    filenames_train_target = [{"img": x, "domain_label": 1.0}\
                              for x in target_train_images]
       
    target_ds_train = monai.data.Dataset(filenames_train_target,
                                         target_transforms)

    target_train_loader = DataLoader(target_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here

    # Putting the filenames in the MONAI expected format - target val set
    filenames_val_target = [{"img": x, "domain_label": 1.0}\
                              for x in target_val_images]


    target_ds_val = monai.data.Dataset(filenames_val_target,
                                         target_transforms)
                                         
    target_val_loader = DataLoader(target_ds_val, 
                                   batch_size=batch_size, 
                                   shuffle = True, 
                                   num_workers=0, 
                                   pin_memory=True, 
                                   collate_fn=pad_list_data_collate,
                                   drop_last=True) # add drop_last argument here

    return source_ds_train, source_train_loader, source_ds_val, source_val_loader,\
           target_ds_train, target_train_loader, target_ds_val, target_val_loader

