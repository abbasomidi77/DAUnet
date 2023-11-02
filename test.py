import torch
import argparse
from model import *
from verbose_utils import plot_samples_source_loaders, plot_samples_target_loaders
import monai
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.data import Dataset, ArrayDataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld,
                              RandAxisFlipd, RandGaussianNoised, RandGibbsNoised, RandSpatialCropd, Compose,
                              CropForegroundd, DivisiblePadd)
from monai.data.utils import pad_list_data_collate
from monai.inferers import sliding_window_inference
import pandas as pd
import numpy as np
import nibabel as nib
from verbose_utils import plot_centre_slices

# python test.py --results_dir /home/abbas.omidi/multitask_model/mu96/New_Results/ --test_images /home/abbas.omidi/multitask_model/mu96/Data-split/target_test_set.csv --model_path /home/abbas.omidi/multitask_model/mu96/New_Results/unet_neg_neck.pth --verbose True

# python test.py --results_dir ./results-unet/ --test_images ./Data-split/source_test_set.csv --model_path ./results-unet/unet_orig.pth --verbose True
# python test.py --results_dir ./results-unet-neg/ --test_images ./Data-split/target_test_set.csv --model_path ./results-unet-neg/unet_neg.pth --verbose True

# python test.py --results_dir ./result-daunet/ --test_images ./Data-split/target_test_set.csv --model_path ./result-daunet/unet_neg_neck.pth --verbose True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default="./results-unet/", help='results directory')
    parser.add_argument('--test_images', type=str,
                        default=None, help='path to test set CSV file')
    parser.add_argument('--model_path', type=str,
                        default=None, help='path to pre-trained model')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='verbose debugging flag')

    args = parser.parse_args()

    root_dir = args.results_dir  # Path to store results
    verbose = args.verbose  # Debugging flag

    test_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            ScaleIntensityd(
                keys=["img"],
                minv=0.0,
                maxv=1.0
            ),
            DivisiblePadd(["img"], 16)

        ]
    )
 
    model = DAUnet()
    #model = Unet()
    model.cuda()
    model.eval()
    #model.load_state_dict(torch.load(args.model_path))
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    global_step = 1
    alpha = 2. / (1. + np.exp(-10 * global_step)) - 1
    test_images = np.array(pd.read_csv(args.test_images)["filename"])
    filenames_test = [{"img": x} for x in test_images]

    def model_slide(a):
        o1,o2 = model(a, alpha = 1.0)
        #o1 = model(a)
        return o1
    ds_test = monai.data.Dataset(filenames_test, test_transforms)

    test_loader = DataLoader(ds_test,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            collate_fn=pad_list_data_collate,
                            drop_last=True)


    roi_size = (112, 112, 112)
    sw_batch_size = 2

    with torch.no_grad():
        for (i, test_image) in enumerate(test_loader):
            # calculate outputs by running images through the network
            pred_seg = sliding_window_inference(test_image["img"].cuda(),
                                                roi_size, sw_batch_size, model_slide)
            pred_seg = torch.argmax(pred_seg, dim=1).cpu().numpy()[0]
            print(pred_seg.shape)
            print(pred_seg.dtype)
            test_path = test_images[i]
            print(root_dir)
            print(test_path)
            out_file = root_dir + \
                test_path.split("/")[-1].split(".")[0] + "_seg.nii.gz"
            out_plot = root_dir + "test_" + \
                test_path.split("/")[-1].split(".")[0]
            aux = nib.load(test_path)
            print(aux.shape)
            affine = aux.affine
            H, W, Z = aux.shape
            nii = nib.Nifti1Image(pred_seg[:H,:W,:Z].astype(np.uint8), affine)
            nib.save(nii, out_file)
            if verbose:
                plot_centre_slices(test_image["img"][0, 0, 64:-32, 64:-32, 32:-32].numpy(),
                                   pred_seg, out_plot)
