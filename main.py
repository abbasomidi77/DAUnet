import torch
import argparse
import matplotlib.pylab as plt
import numpy as np
from model import *
from data_loader_da import *
from training import train_unet, train_unet_da
from verbose_utils import plot_samples_source_loaders, plot_samples_target_loaders
from torch.optim.lr_scheduler import StepLR

# python main.py --batch_size 1 --source_dev_images ./Data-split/source_train_set_neg.csv --source_dev_masks ./Data-split/source_train_set_masks_neg.csv  --target_dev_images ./Data-split/target_train_set.csv --verbose True
# python main.py --batch_size 1 --source_dev_images ./Data-split/source_train_set_neg.csv --source_dev_masks ./Data-split/source_train_set_masks_neg.csv   --verbose True
# python main.py --batch_size 2 --source_dev_images ./Data-split/source_train_set_neg_hybrid.csv --source_dev_masks ./Data-split/source_train_set_masks_neg_hybrid.csv   --verbose True --results_dir ./results-unet-neg-neck/


#python main.py --batch_size 1 --source_dev_images ./Data-split/source_train_set_neg.csv --source_dev_masks ./Data-split/source_train_set_masks_neg.csv  --target_dev_images ./Data-split/target_train_set.csv --verbose True --results_dir ./results-daunet/


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--val_split', type=float, default=0.3, help='Vaal split')
    parser.add_argument('--results_dir', type=str, default ="./results/", help='results directory')
    parser.add_argument('--source_dev_images', type=str, help='path to source dev images')
    parser.add_argument('--source_dev_masks', type=str, help='path to source dev masks')
    parser.add_argument('--target_dev_images', type=str, default=None, help='path to target dev images')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')
    parser.add_argument('--resume_training', type=bool, default=False, help='Continue training from the last checkpoint')

    args = parser.parse_args()

    root_dir = args.results_dir # Path to store results
    verbose = args.verbose # Debugging flag
    start_epoch = 1
    # Set our data loaders - supervised training with no domain adaptation
    if not args.target_dev_images:
        source_ds_train, source_train_loader, \
        source_ds_val, source_val_loader = load_data(args.source_dev_images,\
                                                     args.source_dev_masks, \
                                                     batch_size = args.batch_size, val_split = args.val_split, verbose = verbose)
    
    # Set our data loaders - supervised soruce domain training with unsupervised target domain adaptation
    else:
        source_ds_train, source_train_loader, \
        source_ds_val, source_val_loader, \
        target_ds_train, target_train_loader,\
        target_ds_val, target_val_loader = load_data(args.source_dev_images,\
                                                     args.source_dev_masks, \
                                                     args.target_dev_images,\
                                                     batch_size = args.batch_size, val_split = args.val_split, verbose = verbose)

    # Inspecting output of source domain data loaders
    if verbose:
        plot_samples_source_loaders(source_train_loader, source_val_loader, root_dir)

# Inspecting output of target domain data loaders
    if verbose and args.target_dev_images:
        plot_samples_target_loaders(target_train_loader, target_val_loader, root_dir)
    
    # Build unet model with domain adaptation
    if args.target_dev_images:
        model = DAUnet()
    else: # Build Unet model without domain adaptation
        model = Unet()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    model = model.cuda()
    # to do - learning rate scheduler, early stoppin, weights and biases

    # Learning rate decay scheduler
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    print("Start of training...")
    if not args.target_dev_images:
        train_unet(source_train_loader, source_val_loader,\
                model, optimizer, scheduler, args.epochs, root_dir)
    else:
        train_unet_da(source_train_loader, source_val_loader,\
              target_train_loader, target_val_loader,\
              model, optimizer, scheduler,  
              args.epochs, root_dir, loss_weight=2, start_epoch=start_epoch, resume_training=args.resume_training)


    print("End of training.")
    model.load_state_dict(torch.load(root_dir + "unet_neg_neck.pth"))

    source_ds_train, source_train_loader, \
    source_ds_val, source_val_loader, \
    target_ds_train, target_train_loader,\
    target_ds_val, target_val_loader = load_data(args.source_dev_images,\
                                                     args.source_dev_masks, \
                                                     args.target_dev_images,\
                                                     batch_size = 1, val_split = args.val_split, verbose = verbose)
    global_step = 1
    alpha = 2. / (1. + np.exp(-10 * global_step)) - 1
    with torch.no_grad():
        for step, batch in enumerate(target_val_loader):
            img = (batch["img"].cuda())
            pred_tissue_mask = model(img,alpha)
            img = img.cpu()
            pred_tissue_mask = pred_tissue_mask[0].cpu()
            plt.figure()
            plt.subplot(121)
            plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
            plt.subplot(122)
            plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
            plt.imshow(np.argmax(pred_tissue_mask.numpy(),axis = 1)[0,32,:,:], alpha = 0.4)
            plt.savefig(root_dir + "Test_sample_" + str(step) + ".png")
            plt.close()
