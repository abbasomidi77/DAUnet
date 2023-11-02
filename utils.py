import torch
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning
from monai.utils import set_determinism
import monai
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    DivisiblePadd,
    ScaleIntensityd,
    RandRotated
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data import CacheDataset, Dataset, list_data_collate, decollate_batch, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.config import print_config
from monai.apps import download_and_extract
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from tqdm.notebook import tqdm_notebook
import torch.nn.functional as Fun
import nibabel as nib
from main import *

#not using this anywhere anymore
mask_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img"]),
        DivisiblePadd(["img"], 16),
    ]
)

#not using this anywhere anymore
def get_binary_mask(binary_mask_path):
    mask_dir = sorted(glob.glob(r"C:\Users\nehag\OneDrive\Desktop\sample_data\binary_mask\*.nii.gz"))
    mask_name = [{"img": x} for x in mask_dir]

    ds_mask = monai.data.Dataset(mask_name, mask_transforms)
    mask_loader = DataLoader(ds_mask, batch_size=2, shuffle=True, num_workers=1, pin_memory=True, collate_fn=pad_list_data_collate, drop_last=True)

    for batch in mask_loader:
        mask = batch["img"].as_tensor()
        break

    return mask


post_segpred = Compose([EnsureType("tensor", device="cpu")])
post_seglabel = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=4)])
#post_agepred = Compose([EnsureType("tensor", device="cpu")])
post_agelabel = Compose([EnsureType("tensor", device="cpu")])


step_loss_values=[]
dicemetric_values=[]
maemetric_values=[]
globmaemetric_values =[]
dice_val_best = 0.0
mae_val_best = 0.0
global_step = 0
global_step_best = 0

eval_num = 1