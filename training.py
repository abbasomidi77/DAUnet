from monai.losses import DiceLoss
import torch
import matplotlib.pylab as plt
import numpy as np
import torch.nn as nn
import numpy as np
def train_unet(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    best_val_loss = 1e+10

    loss_object = DiceLoss(to_onehot_y = True)
    for epoch in range(1,max_epochs +1):
        train_loss = 0.0
        val_loss = 0.0
    
        print("Epoch ", epoch, flush=True)
        print("Train:", end ="", flush=True)
        for step, batch in enumerate(train_loader):
            img, brain_mask= (batch["img"].cuda(), batch["brain_mask"].cuda()
                                            )
            optimizer.zero_grad()

            pred_tissue_mask = model(img)

            loss = loss_object(pred_tissue_mask,brain_mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print("=", end = "", flush=True)

        train_loss = train_loss/(step+1)

        print()
        print("Val:", end ="", flush=True)
        with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    img, brain_mask = (batch["img"].cuda(), batch["brain_mask"].cuda())
                    brain_img = img#*brain_mask
       
                    pred_tissue_mask = model(brain_img)

                    loss = loss_object(pred_tissue_mask,brain_mask)
                    val_loss += loss.item()
                    print("=", end = "", flush=True)
                print()
                val_loss = val_loss/(step+1)
        img = img.cpu()
        pred_tissue_mask = pred_tissue_mask.cpu()
        plt.figure()
        plt.subplot(121)
        plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
        plt.subplot(122)
        plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
        plt.imshow(np.argmax(pred_tissue_mask.numpy(),axis = 1)[0,32,:,:], alpha = 0.4)
        plt.savefig(root_dir +"val_sample_epoch_" + str(epoch) + ".png")

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, flush=True)

        if val_loss < best_val_loss:
            print("Saving model", flush=True)
            torch.save(model.state_dict(), root_dir + "unet_neg_neck.pth")    
            best_val_loss = val_loss
    return


def train_unet_da(source_train_loader, source_val_loader,\
                  target_train_loader, target_val_loader,\
                  model, optimizer, scheduler, max_epochs, root_dir,\
                  loss_weight=2, start_epoch=1, resume_training=False):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    best_val_loss = 1e+10
    segmentation_loss = DiceLoss(to_onehot_y=True)
    domain_classifier_loss = nn.BCELoss()#nn.CrossEntropyLoss()
    m = nn.Sigmoid()
    #global_step = 0
    print('resume_training: ',resume_training)
    if resume_training:
        checkpoint = torch.load(root_dir + "checkpoint.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch} with best_val_loss = {best_val_loss:.4f}")

    for epoch in range(start_epoch, max_epochs + 1):

        global_step = 0
        len_dataloader = min(len(source_train_loader), len(target_train_loader))
        train_loss = 0.0
        train_seg_loss = 0.0
        train_dc_loss = 0.0
        train_dc_loss_source = 0.0
        train_dc_loss_target = 0.0

        val_loss = 0.0
        val_seg_loss = 0.0
        val_dc_loss = 0.0
        val_dc_loss_source = 0.0
        val_dc_loss_target = 0.0

        print("Epoch ", epoch, flush=True)
        print("Train:", end="", flush=True)
        
        for step, batch in enumerate(zip(target_train_loader, source_train_loader)):
            batch_target, batch_source = batch
            target_img, target_domain_label = batch_target["img"].to(device),\
                                              batch_target["domain_label"].type(torch.float32).to(device)

                                            
            source_img, source_mask, source_domain_label = batch_source["img"].to(device), \
                                      batch_source["brain_mask"].to(device),batch_source["domain_label"].type(torch.float32).to(device)
            
            optimizer.zero_grad()

            
            p = float(global_step + epoch * len_dataloader) / max_epochs / len_dataloader 
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            pred_seg_source, pred_domain_source = model(source_img, alpha)

            pred_seg_target, pred_domain_target = model(target_img, alpha)
            loss_seg = segmentation_loss(pred_seg_source, source_mask)
            #loss_dc = domain_classifier_loss(m(pred_domain_source).reshape(torch.unsqueeze(source_domain_label,0).shape), torch.unsqueeze(source_domain_label,0)) \
                       #+ domain_classifier_loss(m(pred_domain_target).reshape(torch.unsqueeze(target_domain_label,0).shape), torch.unsqueeze(target_domain_label,0))

            dc_loss_source = domain_classifier_loss(m(pred_domain_source).reshape(torch.unsqueeze(source_domain_label,0).shape), torch.unsqueeze(source_domain_label,0))
            dc_loss_target = domain_classifier_loss(m(pred_domain_target).reshape(torch.unsqueeze(target_domain_label,0).shape), torch.unsqueeze(target_domain_label,0))
            loss_dc = dc_loss_source + dc_loss_target
            total_loss = loss_seg + loss_dc

            total_loss.backward()
            optimizer.step()
            train_dc_loss_source += dc_loss_source.item()
            train_dc_loss_target += dc_loss_target.item()
            train_seg_loss += loss_seg.item()
            train_dc_loss += loss_dc.item()
            global_step += 1
            print("=", end="", flush=True)

        train_seg_loss = train_seg_loss / (step + 1)
        train_dc_loss = train_dc_loss / (step + 1)
        train_loss = train_seg_loss + train_dc_loss
        train_dc_loss_source = train_dc_loss_source / (step + 1)
        train_dc_loss_target = train_dc_loss_target / (step + 1)
        

        print("Val:", end ="", flush=True)
        with torch.no_grad():
            for step, batch in enumerate(zip(target_val_loader, source_val_loader)):
                batch_target, batch_source = batch
                target_img, target_domain_label = batch_target["img"].cuda(),\
                                                batch_target["domain_label"].type(torch.float32).cuda()

                                                
                source_img, source_mask, source_domain_label = batch_source["img"].cuda(), \
                                        batch_source["brain_mask"].cuda(),batch_source["domain_label"].type(torch.float32).cuda()

                
                #p = float(global_step + epoch * len_dataloader) / max_epochs / len_dataloader 
                #alpha = 2. / (1. + np.exp(-10 * p)) - 1

                pred_seg_source,pred_domain_source = model(source_img,alpha)

                pred_seg_target,pred_domain_target = model(target_img,alpha)
                loss_seg = segmentation_loss(pred_seg_source,source_mask)

                dc_loss_source = domain_classifier_loss(m(pred_domain_source).reshape(torch.unsqueeze(source_domain_label,0).shape), torch.unsqueeze(source_domain_label,0))
                dc_loss_target = domain_classifier_loss(m(pred_domain_target).reshape(torch.unsqueeze(target_domain_label,0).shape), torch.unsqueeze(target_domain_label,0))
                loss_dc = dc_loss_source + dc_loss_target
                
                val_loss = loss_seg + loss_dc

                val_dc_loss_source += dc_loss_source.item()
                val_dc_loss_target += dc_loss_target.item()
                val_seg_loss += loss_seg.item()
                val_dc_loss += loss_dc.item()
                print("=", end = "", flush=True)
                #global_step+=1

            val_seg_loss = val_seg_loss/(step+1)
            val_dc_loss = val_dc_loss/(step+1)
            val_loss = val_seg_loss + val_dc_loss
            val_dc_loss_source = val_dc_loss_source / (step + 1)
            val_dc_loss_target = val_dc_loss_target / (step + 1)
        
        print(f"\nTraining epoch {epoch}, train loss: {train_loss:.4f}, train seg loss: {train_seg_loss:.4f}, train domain loss: {train_dc_loss:.4f}, train_dc_loss_source: {train_dc_loss_source:.4f}, train_dc_loss_target: {train_dc_loss_target:.4f}",flush=True)
        print(f"\nvalidation loss: {val_loss:.4f}, validation seg loss: {val_seg_loss:.4f}, validation domain loss: {val_dc_loss:.4f}, val_dc_loss_source: {val_dc_loss_source:.4f}, val_dc_loss_target: {val_dc_loss_target:.4f}",flush=True)
        

        #print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        #print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        #print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
        
        if val_seg_loss < best_val_loss:
            print("Saving model", flush=True)
            torch.save(model.state_dict(), root_dir + "unet_neg_neck.pth")    
            best_val_loss = val_seg_loss
        # Save model and optimizer state after every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, root_dir + "checkpoint.pth")
    return


'''
def train_unet_da(source_train_loader, source_val_loader,\
                  target_train_loader, target_val_loader,\
                  model, optimizer, scheduler, max_epochs, root_dir,\
                 loss_weight = 2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    
    best_val_loss = 1e+10

    segmentation_loss = DiceLoss(to_onehot_y = True)
    domain_classifier_loss = nn.BCELoss()#nn.CrossEntropyLoss()
    m = nn.Sigmoid()
    global_step = 1
    for epoch in range(1,max_epochs +1):
        train_loss = 0.0
        train_seg_loss = 0.0
        train_dc_loss = 0.0

        val_loss = 0.0
        val_seg_loss = 0.0
        val_dc_loss = 0.0
    
        print("Epoch ", epoch, flush=True)
        print("Train:", end ="", flush=True)
        for step, batch in enumerate(zip(target_train_loader, source_train_loader)):
            batch_target, batch_source = batch
            target_img, target_domain_label = batch_target["img"].cuda(),\
                                              batch_target["domain_label"].type(torch.float32).cuda()

                                            
            source_img, source_mask, source_domain_label = batch_source["img"].cuda(), \
                                      batch_source["brain_mask"].cuda(),batch_source["domain_label"].type(torch.float32).cuda()
            
            optimizer.zero_grad()

            alpha = 2. / (1. + np.exp(-10 * global_step)) - 1

            pred_seg_source,pred_domain_source = model(source_img,alpha)

            pred_seg_target,pred_domain_target = model(target_img,alpha)
            loss_seg = segmentation_loss(pred_seg_source,source_mask)
            loss_dc = domain_classifier_loss(m(pred_domain_source).reshape(torch.unsqueeze(source_domain_label,0).shape),torch.unsqueeze(source_domain_label,0))\
                       + domain_classifier_loss(m(pred_domain_target).reshape(torch.unsqueeze(target_domain_label,0).shape),torch.unsqueeze(target_domain_label,0))

            total_loss = loss_seg + loss_dc
            total_loss.backward()
            optimizer.step()
            train_seg_loss += loss_seg.item()
            train_dc_loss += loss_dc.item()
            global_step+=1
            print("=", end = "", flush=True)
        train_seg_loss = train_seg_loss/(step+1)
        train_dc_loss = train_dc_loss/(step+1)
        train_loss = train_seg_loss + train_dc_loss

        # Temporary
        print("Training epoch ", epoch, ", train loss:", train_loss,
            ", train seg loss:", train_seg_loss,", train domain loss:", train_dc_loss, flush=True)
        if train_loss < best_val_loss:
            print("Saving model", flush=True)
            torch.save(model.state_dict(), root_dir + "unet_neg_neck.pth")    
            best_val_loss = train_loss

    '''






'''
        print("Val:", end ="", flush=True)
        with torch.no_grad():
            for step, batch in enumerate(zip(target_val_loader, source_val_loader)):
                batch_target, batch_source = batch
                target_img, target_domain_label = batch_target["img"].cuda(),\
                                                batch_target["domain_label"].type(torch.float32).cuda()

                                                
                source_img, source_mask, source_domain_label = batch_source["img"].cuda(), \
                                        batch_source["brain_mask"].cuda(),batch_source["domain_label"].type(torch.float32).cuda()

                alpha = 2. / (1. + np.exp(-10 * global_step)) - 1

                pred_seg_source,pred_domain_source = model(source_img,alpha)

                pred_seg_target,pred_domain_target = model(target_img,alpha)
                loss_seg = segmentation_loss(pred_seg_source,source_mask)

                loss_dc = domain_classifier_loss(m(pred_domain_source).reshape(torch.unsqueeze(source_domain_label,0).shape),torch.unsqueeze(source_domain_label,0))\
                        + domain_classifier_loss(m(pred_domain_target).reshape(torch.unsqueeze(target_domain_label,0).shape),torch.unsqueeze(target_domain_label,0))
                print('Domain_classifier_loss:',domain_classifier_loss(m(pred_domain_source).reshape(torch.unsqueeze(source_domain_label,0).shape),torch.unsqueeze(source_domain_label,0)))
                print('Source_classifier_loss',domain_classifier_loss(m(pred_domain_target).reshape(torch.unsqueeze(target_domain_label,0).shape),torch.unsqueeze(target_domain_label,0)))
                total_loss = loss_seg + loss_dc
                val_seg_loss += loss_seg.item()
                val_dc_loss += loss_dc.item()
                print("=", end = "", flush=True)
                global_step+=1

            val_seg_loss = val_seg_loss/(step+1)
            val_dc_loss = val_dc_loss/(step+1)
            val_loss = val_seg_loss + val_dc_loss
        print("Training epoch ", epoch, ", train loss:", train_loss,
            ", train seg loss:", train_seg_loss,", train domain loss:", train_dc_loss,", validation loss:", val_loss,
            ", validation seg loss:", val_seg_loss,", validation domain loss:", val_dc_loss, flush=True)
        if val_loss < best_val_loss:
            print("Saving model", flush=True)
            torch.save(model.state_dict(), root_dir + "unet_neg_neck.pth")    
            best_val_loss = val_loss
'''
    #return