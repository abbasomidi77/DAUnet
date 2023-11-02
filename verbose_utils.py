import matplotlib.pylab as plt
import numpy as np
from datetime import datetime
import os
# Lots of stuff hardcoded, which is bad programming practice,
# My goal was just to inspect the outputs from the data loaders
# to make sure they were correct
# I also looked into the intensities of the images.
def plot_samples_source_loaders(train_loader, val_loader, save_path):
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    for ii in range(2):
    
        val_batch = next(val_iter)    
        train_batch = next(train_iter)
        print("Source train batch shape:", train_batch["img"].shape, 
                                    train_batch["brain_mask"].shape)

        print("Source train batch min-max:", train_batch["img"].numpy().min(),
                                      train_batch["img"].numpy().max())
        
        print(np.unique(train_batch["brain_mask"].numpy()))
        
        print("Source val batch shape:", val_batch["img"].shape, val_batch["brain_mask"].shape)
        
        print("Source val batch min-max:", val_batch["img"].numpy().min(),
                                      val_batch["img"].numpy().max())
        
        print(np.unique(val_batch["brain_mask"].numpy()))
        
        # Train
        plt.figure()
        plt.subplot(121)
        plt.imshow(train_batch["img"][0,0,32,:,:], cmap = "gray")

        plt.subplot(122)
        plt.imshow(train_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.imshow(train_batch["brain_mask"][0,0,32,:,:], alpha = 0.4)
        date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
        plt.savefig(save_path + "source_train_sample" \
                    + str(ii) + "_" + date + ".png")

        # Val
        plt.figure()
        plt.subplot(121)
        plt.imshow(val_batch["img"][0,0,32,:,:], cmap = "gray")

        plt.subplot(122)
        plt.imshow(val_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.imshow(val_batch["brain_mask"][0,0,32,:,:], alpha = 0.4)
        date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
        plt.savefig(save_path + "source_val_sample" \
                    + str(ii) + "_" + date + ".png")

    return


def plot_samples_target_loaders(train_loader, val_loader, save_path):
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    for ii in range(2):
    
        val_batch = next(val_iter)    
        train_batch = next(train_iter)
        print("Target train batch shape:", train_batch["img"].shape)

        print("Target train batch min-max:", train_batch["img"].numpy().min(),
                                      train_batch["img"].numpy().max())
        
        
        print("Target val batch shape:", val_batch["img"].shape)
        
        print("Target val batch min-max:", val_batch["img"].numpy().min(),
                                      val_batch["img"].numpy().max())
        
        
        # Train
        plt.figure()
        plt.imshow(train_batch["img"][0,0,32,:,:], cmap = "gray")
        date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
        plt.savefig(save_path + "target_train_sample" \
                    + str(ii) + "_" + date + ".png")

        # Val
        plt.figure()
        plt.imshow(val_batch["img"][0,0,32,:,:], cmap = "gray")
        date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
        plt.savefig(save_path + "target_val_sample" \
                    + str(ii) + "_" + date + ".png")

    return



def plot_centre_slices(img,mask,save_path):
        
    # Train
    H,W,Z = img.shape
    plt.figure()
    plt.subplot(131)
    plt.imshow(img[H//2,:,:], cmap = "gray")
    plt.imshow(mask[H//2,:,:],  alpha = 0.4)
    
    plt.subplot(132)
    plt.imshow(img[:,W//2,:], cmap = "gray")
    plt.imshow(mask[:,W//2,:],  alpha = 0.4)
    
    plt.subplot(133)
    plt.imshow(img[:,:,Z//2], cmap = "gray")
    plt.imshow(mask[:,:,Z//2],  alpha = 0.4)
    
    date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
    plt.savefig(save_path + "_" + date + ".png")

    return