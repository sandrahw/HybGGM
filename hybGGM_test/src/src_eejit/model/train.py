
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
import os
import torch.optim as optim
import gc
import sys


# def rmse(outputs, targets, mas):
#     return torch.sqrt(F.mse_loss(outputs[mas], targets[mas]))
def rmse(outputs, targets, mas):
    diff = (outputs[mas] - targets[mas])
    # if torch.isnan(diff).any():
    #     print("NaN in diff")
    squared_diff = diff ** 2
    # if torch.isnan(squared_diff).any():
    #     print("NaN in squared_diff")
    mse = torch.mean(squared_diff)
    # if torch.isnan(mse).any():
    #     print("NaN in mse")
    rmse_value = torch.sqrt(mse)
    return rmse_value
# Training function
def train_model(model, train_loader, validation_loader, lr_rate, num_epochs, save_path, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0

        for inputs, targets, masks in train_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()

            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = criterion(outputs[masks], targets[masks])
            loss = rmse(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
    
        if writer:
            writer.add_scalar('Loss/train_epoch', running_train_loss / len(train_loader), epoch)
            
        # Validation Phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets, masks in validation_loader:
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()

                outputs = model(inputs)
                # loss = criterion(outputs[masks], targets[masks])
                loss = rmse(outputs, targets, masks)
                running_val_loss += loss.item()
         
        epoch_val_loss = running_val_loss / len(validation_loader.dataset)

        if writer:
            writer.add_scalar('Loss/validation_epoch', running_val_loss / len(validation_loader), epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print("Best model saved!")

    return

def test_model(model, test_loader, writer=None):
    model.eval()
    running_test_loss = 0.0
    all_outputs = []

    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()

            outputs = model(inputs)

            # Compute the loss
            # loss = criterion(outputs[masks], targets[masks])
            loss = rmse(outputs, targets, masks)
            running_test_loss += loss.item()

            # Store outputs for further analysis or visualization if needed
            all_outputs.append(outputs.cpu().numpy())

        # Compute the average loss, RMSE, and MAE for the entire test dataset
        test_loss = running_test_loss / len(test_loader.dataset)

        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/test_epoch', test_loss)
        # Combine all output batches into a single array
        # all_outputs = np.concatenate(all_outputs, axis=0)

        # Print test results
        print(f"Test Loss: {test_loss:.4f}")

    return test_loss




'''random samples'''
# Function to check memory usage
def print_memory_usage(stage=""):
    allocated = torch.cuda.memory_allocated() / 1024**2  # Convert bytes to MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # Convert bytes to MB
    print(f"{stage} - Allocated memory: {allocated:.2f} MB, Reserved memory: {reserved:.2f} MB")

def RS_train_test_model(model, train_loader, test_loader, lr_rate, num_epochs, save_path, patience, device, bs, sub_batch_size, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    best_test_loss = float('inf')
    scaler = torch.amp.GradScaler()
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        sys.stdout.flush()
        model.train()
        running_train_loss = 0.0
        len_train_loader = 0
        len_test_loader = 0
        #print train_loader
        print(next(iter(train_loader))[0].shape)
        with torch.set_grad_enabled(True):
            for i, (inputs, targets, masks) in enumerate(train_loader):
                print(i)
                # inputs = inputs.to(device, non_blocking=True).float()# inputs.to(device).float()
                # targets = targets.to(device, non_blocking=True).float()#targets.to(device).float()
                # masks = masks.to(device, non_blocking=True).bool()#masks.to(device).bool()
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()
                # print('input size', sys.getsizeof(inputs.storage()))
                # print('target size', sys.getsizeof(targets.storage()))
                # print('mask size', sys.getsizeof(masks.storage()))    

                # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                if inputs.shape != targets.shape:
                    inputs = inputs[:, :, 0, :, :]
                # print('input shape', inputs.shape)

                for j in range(0, bs, sub_batch_size):
                    print('sub batch:', j)
                    print(range(0, bs, sub_batch_size))
                    print(j,j + sub_batch_size)
                    # Select sub-batch from inputs and targets (still on CPU)
                    inputs_sub = inputs[j:j + sub_batch_size, :, :, :]
                    targets_sub = targets[j:j + sub_batch_size, :, :, :]
                    masks_sub = masks[j:j + sub_batch_size, :, :, :] 
                    # print('input sub shape', inputs_sub.shape)
                    # print('input sub type', inputs_sub.type())
                    inputs_sub = inputs_sub.to(device, non_blocking=True)
                    targets_sub = targets_sub.to(device, non_blocking=True)
                    masks_sub = masks_sub.to(device, non_blocking=True)
                
                    # print('input sub size', sys.getsizeof(inputs_sub.storage()))
                    # print('target sub size', sys.getsizeof(targets_sub.storage()))
                    # print('mask sub size', sys.getsizeof(masks_sub.storage()))                     

                    # optimizer.zero_grad()
                    # print('run model on inputs')
                    outputs = model(inputs_sub)
                    # print('outputs nan:', torch.isnan(outputs).any())
                    if targets_sub.shape != outputs.shape:
                        targets_sub = targets_sub[:,0,:,:]
                        print('targets_sub dim', targets_sub.dim(), targets_sub.shape)
                    if masks_sub.shape != outputs.shape:
                        masks_sub = masks_sub[0,0,:,:,:]
                        print('mask dim', masks_sub.dim(), masks_sub.shape)
                    # print('calc rmse')
                    loss = rmse(outputs, targets_sub, masks_sub) 
                    # print('train loss item:', loss.item())    
                    scaler.scale(loss).backward()
                    # optimizer.step()
                    
                    if not torch.isnan(loss):
                        running_train_loss += loss.item()# * accumulation_steps  # Multiply back for tracking
                        len_train_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
            # running_train_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_loss = running_train_loss / len_train_loader #if I skip nan values I have to adapt the calc
        print('epoch train loss:', epoch_train_loss)

        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)

        # Testing Phase
        print('testing phase')
        model.eval()
        running_test_loss = 0.0
        with torch.set_grad_enabled(True):
        # with torch.no_grad():
            for i, (inputs, targets, masks) in enumerate(test_loader):
                # inputs = inputs.to(device, non_blocking=True).float()# inputs.to(device).float()
                # targets = targets.to(device, non_blocking=True).float()#targets.to(device).float()
                # masks = masks.to(device, non_blocking=True).bool()#masks.to(device).bool()
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()
                if inputs.shape != targets.shape:
                    inputs = inputs[:, :, 0, :, :]

                for j in range(0, bs, sub_batch_size):

                    # Select sub-batch from inputs and targets (still on CPU)
                    inputs_sub = inputs[j:j + sub_batch_size, :, :, :]
                    targets_sub = targets[j:j + sub_batch_size, :, :, :]
                    masks_sub = masks[j:j + sub_batch_size, :, :, :] 
                    # print('input sub shape', inputs_sub.shape)
                    # print('input sub type', inputs_sub.type())
                    inputs_sub = inputs_sub.to(device, non_blocking=True)
                    targets_sub = targets_sub.to(device, non_blocking=True)
                    masks_sub = masks_sub.to(device, non_blocking=True)
                
                    # print('input sub size', sys.getsizeof(inputs_sub.storage()))
                    # print('target sub size', sys.getsizeof(targets_sub.storage()))
                    # print('mask sub size', sys.getsizeof(masks_sub.storage()))                     

                    outputs = model(inputs_sub)
                
                    # print('outputs nan:', torch.isnan(outputs).any())
                    if targets_sub.shape != outputs.shape:
                        targets_sub = targets_sub[:,0,:,:]
                    if masks_sub.shape != outputs.shape:
                        masks_sub = masks_sub[0,0,:,:,:]
        
                    # loss = rmse(outputs, targets, masks)
                    loss = rmse(outputs, targets_sub, masks_sub)# / accumulation_steps
                    # print('test loss item:', loss.item())
                
                    if not torch.isnan(loss):
                        running_test_loss += loss.item()
                        len_test_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
                
            # epoch_test_loss = running_test_loss / len(test_loader)
            epoch_test_loss = running_test_loss / len_test_loader #if I skip nan values I have to adapt the calc

            if writer:
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Testing Loss: {epoch_test_loss:.4f}")
            sys.stdout.flush()
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print("Best model saved!")
            else:
                epochs_without_improvement += 1
                print('no improvement in test loss for', epochs_without_improvement, 'epochs')
            if epochs_without_improvement >= patience:
                    print("Early stopping!")
                    break
        # Check memory usage before cleanup
        print_memory_usage("Before cleanup")
        # Clear cache and run garbage collection after each epoch
        torch.cuda.empty_cache()  # Free up unallocated cached memory
        gc.collect()              # Run garbage collection to clear Python references
        # Check memory usage after cleanup
        print_memory_usage("After cleanup")
    print('training and testing done')
    return
    
def RS_val_model(model, validation_loader, device, bs, sub_batch_size, writer=None):
    model.eval()
    running_val_loss = 0.0
    len_validation_loader = 0
    with torch.no_grad():
        for inputs, targets, masks in validation_loader:
            # inputs = inputs.float()# inputs.to(device).float()
            # targets = targets.float()#targets.to(device).float()
            # masks = masks.bool()#masks.to(device).bool()
            inputs = inputs.to(device).float()# inputs.to(device).float()
            targets = targets.to(device).float()#targets.to(device).float()
            masks = masks.to(device).bool()#masks.to(device).bool()

            # print('inputs shape:', inputs.shape)
            # print('targets shape:', targets.shape)
            # print('masks shape:', masks.shape)

            if inputs.shape != targets.shape:
                inputs = inputs[:, :, 0, :, :]

            for j in range(0, bs, sub_batch_size):
                print('sub batch:', j)
                print(range(0, bs, sub_batch_size))
                print(j,j + sub_batch_size)
                # Select sub-batch from inputs and targets (still on CPU)
                inputs_sub = inputs[j:j + sub_batch_size, :, :, :]
                targets_sub = targets[j:j + sub_batch_size, :, :, :]
                masks_sub = masks[j:j + sub_batch_size, :, :, :] 
                # print('input sub shape', inputs_sub.shape)
                # print('input sub type', inputs_sub.type())
                inputs_sub = inputs_sub.to(device, non_blocking=True)
                targets_sub = targets_sub.to(device, non_blocking=True)
                masks_sub = masks_sub.to(device, non_blocking=True)
            
                # print('input sub size', sys.getsizeof(inputs_sub.storage()))
                # print('target sub size', sys.getsizeof(targets_sub.storage()))
                # print('mask sub size', sys.getsizeof(masks_sub.storage()))                     

                outputs = model(inputs_sub)
            
                # print('outputs nan:', torch.isnan(outputs).any())
                if targets_sub.shape != outputs.shape:
                    targets_sub = targets_sub[:,0,:,:]
                if masks_sub.shape != outputs.shape:
                    masks_sub = masks_sub[0,0,:,:,:]
    
                # loss = rmse(outputs, targets, masks)
                loss = rmse(outputs, targets_sub, masks_sub)# / accumulation_steps
                # print('val loss item:', loss.item())
            
                if not torch.isnan(loss):
                    running_val_loss += loss.item()
                    len_validation_loader += 1
                # else:
                #     print("NaN encountered in loss calculation. Skipping this instance.")
        val_loss = running_val_loss / len_validation_loader
        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/val_epoch', val_loss)
        # Print test results
        print(f"Val Loss: {val_loss:.4f}")
        sys.stdout.flush()
    print('validation done')
    return 

def rmse_new(outputs, targets, mas):
    diff = (targets[mas] - outputs[mas])
    # if torch.isnan(diff).any():
    #     print("NaN in diff")
    squared_diff = diff ** 2
    # if torch.isnan(squared_diff).any():
    #     print("NaN in squared_diff")
    mse = torch.mean(squared_diff)
    # if torch.isnan(mse).any():
    #     print("NaN in mse")
    rmse_value = torch.sqrt(mse)
    return rmse_value
def RS_train_test_model_newrmse(model, train_loader, test_loader, lr_rate, num_epochs, save_path, patience, device, bs, sub_batch_size, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    best_test_loss = float('inf')
    scaler = torch.amp.GradScaler()
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        sys.stdout.flush()
        model.train()
        running_train_loss = 0.0
        len_train_loader = 0
        len_test_loader = 0
        #print train_loader
        print(next(iter(train_loader))[0].shape)
        with torch.set_grad_enabled(True):
            for i, (inputs, targets, masks) in enumerate(train_loader):
                print(i, 'out of', len(train_loader))
                sys.stdout.flush()
                # inputs = inputs.to(device, non_blocking=True).float()# inputs.to(device).float()
                # targets = targets.to(device, non_blocking=True).float()#targets.to(device).float()
                # masks = masks.to(device, non_blocking=True).bool()#masks.to(device).bool()
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()
                # print('input shape', inputs.shape)

                # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                if inputs.shape != targets.shape:
                    inputs = inputs[:, :, 0, :, :]
                # print('input shape', inputs.shape)

                for j in range(0, bs, sub_batch_size):
                    # print('sub batch:', j)
                    sys.stdout.flush()

                    # Select sub-batch from inputs and targets (still on CPU)
                    inputs_sub = inputs[j:j + sub_batch_size, :, :, :]
                    targets_sub = targets[j:j + sub_batch_size, :, :, :]
                    masks_sub = masks[j:j + sub_batch_size, :, :, :] 
                    # print('input sub shape', inputs_sub.shape)
                    # print('input sub type', inputs_sub.type())
                    inputs_sub = inputs_sub.to(device, non_blocking=True)
                    targets_sub = targets_sub.to(device, non_blocking=True)
                    masks_sub = masks_sub.to(device, non_blocking=True)
                
                    # print('input sub size', sys.getsizeof(inputs_sub.storage()))
                    # print('target sub size', sys.getsizeof(targets_sub.storage()))
                    # print('mask sub size', sys.getsizeof(masks_sub.storage()))                     

                    # optimizer.zero_grad()
                    # print('run model on inputs')
                    outputs = model(inputs_sub)
                    # print('outputs nan:', torch.isnan(outputs).any())
                    if targets_sub.shape != outputs.shape:
                        targets_sub = targets_sub[:,0,:,:]
                        print('targets_sub dim', targets_sub.dim(), targets_sub.shape)
                    if masks_sub.shape != outputs.shape:
                        masks_sub = masks_sub[0,0,:,:,:]
                        print('mask dim', masks_sub.dim(), masks_sub.shape)
                    # print('calc rmse')
                    loss = rmse_new(outputs, targets_sub, masks_sub) 
                    # print('train loss item subbatch %s:' %j, loss.item()) 
                    # sys.stdout.flush()   
                    scaler.scale(loss).backward()
                    # optimizer.step()
                    
                    if not torch.isnan(loss):
                        running_train_loss += loss.item()# * accumulation_steps  # Multiply back for tracking
                        len_train_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
            # running_train_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_loss = running_train_loss / len_train_loader #if I skip nan values I have to adapt the calc
        print('epoch train loss:', epoch_train_loss)

        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)

        # Testing Phase
        print('testing phase')
        model.eval()
        running_test_loss = 0.0
        with torch.set_grad_enabled(True):
        # with torch.no_grad():
            for i, (inputs, targets, masks) in enumerate(test_loader):
                # inputs = inputs.to(device, non_blocking=True).float()# inputs.to(device).float()
                # targets = targets.to(device, non_blocking=True).float()#targets.to(device).float()
                # masks = masks.to(device, non_blocking=True).bool()#masks.to(device).bool()
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()
                if inputs.shape != targets.shape:
                    inputs = inputs[:, :, 0, :, :]

                for j in range(0, bs, sub_batch_size):
                    # Select sub-batch from inputs and targets (still on CPU)
                    inputs_sub = inputs[j:j + sub_batch_size, :, :, :]
                    targets_sub = targets[j:j + sub_batch_size, :, :, :]
                    masks_sub = masks[j:j + sub_batch_size, :, :, :] 
                    # print('input sub shape', inputs_sub.shape)
                    # print('input sub type', inputs_sub.type())
                    inputs_sub = inputs_sub.to(device, non_blocking=True)
                    targets_sub = targets_sub.to(device, non_blocking=True)
                    masks_sub = masks_sub.to(device, non_blocking=True)
                
                    # print('input sub size', sys.getsizeof(inputs_sub.storage()))
                    # print('target sub size', sys.getsizeof(targets_sub.storage()))
                    # print('mask sub size', sys.getsizeof(masks_sub.storage()))                     

                    outputs = model(inputs_sub)
                
                    # print('outputs nan:', torch.isnan(outputs).any())
                    if targets_sub.shape != outputs.shape:
                        targets_sub = targets_sub[:,0,:,:]
                    if masks_sub.shape != outputs.shape:
                        masks_sub = masks_sub[0,0,:,:,:]
        
                    # loss = rmse(outputs, targets, masks)
                    loss = rmse_new(outputs, targets_sub, masks_sub)# / accumulation_steps
                    # print('test loss item:', loss.item())
                    # sys.stdout.flush()
                
                    if not torch.isnan(loss):
                        running_test_loss += loss.item()
                        len_test_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
                
            # epoch_test_loss = running_test_loss / len(test_loader)
            epoch_test_loss = running_test_loss / len_test_loader #if I skip nan values I have to adapt the calc
            if writer:
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Testing Loss: {epoch_test_loss:.4f}")
            sys.stdout.flush()
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print("Best model saved!")
            else:
                epochs_without_improvement += 1
                print('no improvement in test loss for', epochs_without_improvement, 'epochs')
            if epochs_without_improvement >= patience:
                    print("Early stopping!")
                    break
        # Check memory usage before cleanup
        print_memory_usage("Before cleanup")
        # Clear cache and run garbage collection after each epoch
        torch.cuda.empty_cache()  # Free up unallocated cached memory
        gc.collect()              # Run garbage collection to clear Python references
        # Check memory usage after cleanup
        print_memory_usage("After cleanup")
    print('training and testing done')
    return
    
def RS_val_model_newrmse(model, validation_loader, device, bs, sub_batch_size, writer=None):
    model.eval()
    running_val_loss = 0.0
    len_validation_loader = 0
    with torch.no_grad():
        for inputs, targets, masks in validation_loader:
            # inputs = inputs.float()# inputs.to(device).float()
            # targets = targets.float()#targets.to(device).float()
            # masks = masks.bool()#masks.to(device).bool()
            inputs = inputs.to(device).float()# inputs.to(device).float()
            targets = targets.to(device).float()#targets.to(device).float()
            masks = masks.to(device).bool()#masks.to(device).bool()

            # print('inputs shape:', inputs.shape)
            # print('targets shape:', targets.shape)
            # print('masks shape:', masks.shape)

            if inputs.shape != targets.shape:
                inputs = inputs[:, :, 0, :, :]

            for j in range(0, bs, sub_batch_size):
                # Select sub-batch from inputs and targets (still on CPU)
                inputs_sub = inputs[j:j + sub_batch_size, :, :, :]
                targets_sub = targets[j:j + sub_batch_size, :, :, :]
                masks_sub = masks[j:j + sub_batch_size, :, :, :] 
                # print('input sub shape', inputs_sub.shape)
                # print('input sub type', inputs_sub.type())
                inputs_sub = inputs_sub.to(device, non_blocking=True)
                targets_sub = targets_sub.to(device, non_blocking=True)
                masks_sub = masks_sub.to(device, non_blocking=True)
            
                # print('input sub size', sys.getsizeof(inputs_sub.storage()))
                # print('target sub size', sys.getsizeof(targets_sub.storage()))
                # print('mask sub size', sys.getsizeof(masks_sub.storage()))                     

                outputs = model(inputs_sub)
            
                # print('outputs nan:', torch.isnan(outputs).any())
                if targets_sub.shape != outputs.shape:
                    targets_sub = targets_sub[:,0,:,:]
                if masks_sub.shape != outputs.shape:
                    masks_sub = masks_sub[0,0,:,:,:]
    
                # loss = rmse(outputs, targets, masks)
                loss = rmse_new(outputs, targets_sub, masks_sub)# / accumulation_steps
                # print('val loss item:', loss.item())
            
                if not torch.isnan(loss):
                    running_val_loss += loss.item()
                    len_validation_loader += 1
                # else:
                #     print("NaN encountered in loss calculation. Skipping this instance.")
        val_loss = running_val_loss / len_validation_loader
        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/val_epoch', val_loss)
        # Print test results
        print(f"Val Loss: {val_loss:.4f}")
        sys.stdout.flush()
    print('validation done')
    return 

'''random samples cpu'''
def RS_cpu_train_test_model(model, train_loader, test_loader, lr_rate, num_epochs, save_path, patience, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    best_test_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0
        len_train_loader = 0
        len_test_loader = 0

        for inputs, targets, masks in train_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()

            if inputs.shape != targets.shape:
                inputs = inputs[:, :, 0, :, :]

            optimizer.zero_grad()
            outputs = model(inputs)

            if targets.shape != outputs.shape:
                targets = targets.squeeze(1)
                print('targets dim', targets.dim(), targets.shape)
            if masks.shape != outputs.shape:
                masks = masks[0,0,:,:,:]
                print('mask dim', masks.dim(), masks.shape)

            loss = rmse(outputs, targets, masks)
            # print('rmse train loss calc:', loss)
            loss.backward()
            optimizer.step()
            print('loss item:', loss.item())    
            
            if not torch.isnan(loss):
                running_train_loss += loss.item()
                len_train_loader += 1
            else:
                print("NaN encountered in loss calculation. Skipping this instance.")

        # epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_loss = running_train_loss / len_train_loader #if I skip nan values I have to adapt the calc
    
        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)
            
        # Testing Phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks in test_loader:
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()

                if inputs.shape != targets.shape:
                    inputs = inputs[:, :, 0, :, :]

                outputs = model(inputs)
                # print('outputs nan:', torch.isnan(outputs).any())
                if targets.shape != outputs.shape:
                     targets = targets.squeeze(1)
                if masks.shape != outputs.shape:
                     masks = masks[0,0,:,:,:]
    
                loss = rmse(outputs, targets, masks)
                # print('rmse test loss calc:', loss)
                print('loss item:', loss.item())
                
                if not torch.isnan(loss):
                    running_test_loss += loss.item()
                    len_test_loader += 1
                else:
                    print("NaN encountered in loss calculation. Skipping this instance.")
                # running_test_loss += loss.item()

            # epoch_test_loss = running_test_loss / len(test_loader)
            epoch_test_loss = running_test_loss / len_test_loader #if I skip nan values I have to adapt the calc

            if writer:
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Testing Loss: {epoch_test_loss:.4f}")
            sys.stdout.flush()
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print("Best model saved!")
            else:
                epochs_without_improvement += 1
                print('no improvement in test loss for', epochs_without_improvement, 'epochs')
            if epochs_without_improvement >= patience:
                    print("Early stopping!")
                    break
    return
    
def RS_cpu_val_model(model, validation_loader, writer=None):

    model.eval()
    running_val_loss = 0.0
    len_validation_loader = 0

    with torch.no_grad():
        for inputs, targets, masks in validation_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()
            if inputs.shape != targets.shape:
                inputs = inputs[:, :, 0, :, :]

            outputs = model(inputs)
            if targets.shape != outputs.shape:
                targets = targets.squeeze(1)
            if masks.shape != outputs.shape:
                masks = masks[0,0,:,:,:]
            loss = rmse(outputs, targets, masks)
            # print('rmse val loss calc:', loss)
            print('loss item:', loss.item())
            if not torch.isnan(loss):
                running_val_loss += loss.item()
                len_validation_loader += 1
            else:
                print("NaN encountered in loss calculation. Skipping this instance.")
            running_val_loss += loss.item()

        # val_loss = running_val_loss / len(validation_loader)
        val_loss = running_val_loss / len_validation_loader

        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/val_epoch', val_loss)

        # Print test results
        print(f"Val Loss: {val_loss:.4f}")

    return 

''' RS target wtd'''
def rmse_new(outputs, targets, mas):
    diff = (targets[mas] - outputs[mas])
    # if torch.isnan(diff).any():
    #     print("NaN in diff")
    squared_diff = diff ** 2
    # if torch.isnan(squared_diff).any():
    #     print("NaN in squared_diff")
    mse = torch.mean(squared_diff)
    # if torch.isnan(mse).any():
    #     print("NaN in mse")
    rmse_value = torch.sqrt(mse)
    return rmse_value

def RS_train_test_model_newrmse_wtdtarget(model, train_loader, test_loader, lr_rate, num_epochs, save_path, patience, device, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    best_test_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        sys.stdout.flush()
        model.train()
        running_train_loss = 0.0
        len_train_loader = 0
        len_test_loader = 0
        print(next(iter(train_loader))[0].shape)
    
        for i, (inputs, targets, masks) in enumerate(train_loader):
            print(i, 'out of', len(train_loader))
            sys.stdout.flush()
            inputs = inputs.to(device).float()# inputs.to(device).float()
            targets = targets.to(device).float()#targets.to(device).float()
            masks = masks.to(device).bool()#masks.to(device).bool()
            # inputs = inputs.float()# inputs.to(device).float()
            # targets = targets.float()#targets.to(device).float()
            # masks = masks.bool()#masks.to(device).bool()
            print('input shape', inputs.shape)
            print('target shape', targets.shape)
            print('mask shape', masks.shape)
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            if inputs.shape != targets.shape:
                print('input shape not the same as target shape', inputs.shape, targets.shape)
                sys.stdout.flush()
                inputs = inputs[:, :, 0, :, :]
            print('input shape after reshape', inputs.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs nan:', torch.isnan(outputs).any())
            if targets.shape != outputs.shape:
                targets = targets[:,0,:,:]
                print('targets dim', targets.dim(), targets.shape)
            if masks.shape != outputs.shape:
                masks = masks[0,0,:,:,:]
                print('mask dim', masks.dim(), masks.shape)
            # print('calc rmse')
            loss = rmse_new(outputs, targets, masks) 
            loss.backward()
            optimizer.step()
            # print('train loss item:', loss.item())
            # sys.stdout.flush()    
            if not torch.isnan(loss):
                running_train_loss += loss.item()# * accumulation_steps  # Multiply back for tracking
                len_train_loader += 1
               
        epoch_train_loss = running_train_loss / len_train_loader #if I skip nan values I have to adapt the calc
        # print('epoch train loss:', epoch_train_loss)
        # sys.stdout.flush()

        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)

        # Testing Phase
        print('testing phase')
        sys.stdout.flush()
        model.eval()
        running_test_loss = 0.0
        # with torch.set_grad_enabled(True):
        with torch.no_grad():
            for i, (inputs, targets, masks) in enumerate(test_loader):
                inputs = inputs.to(device).float()# inputs.to(device).float()
                targets = targets.to(device).float()#targets.to(device).float()
                masks = masks.to(device).bool()#masks.to(device).bool()
                print('input test shape', inputs.shape)
                # print('target test shape', targets.shape)
                # print('mask test shape', masks.shape)

                if inputs.shape != targets.shape:
                    inputs = inputs[:, :, 0, :, :]
                print('input shape', inputs.shape)

                outputs = model(inputs)
            
                # print('outputs nan:', torch.isnan(outputs).any())
                if targets.shape != outputs.shape:
                    targets = targets[:,0,:,:]
                if masks.shape != outputs.shape:
                    masks = masks[0,0,:,:,:]
    
                # loss = rmse(outputs, targets, masks)
                loss = rmse_new(outputs, targets, masks)# / accumulation_steps
                # print('test loss item:', loss.item())
                # sys.stdout.flush()
            
                if not torch.isnan(loss):
                    running_test_loss += loss.item()
                    len_test_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
            epoch_test_loss = running_test_loss / len_test_loader #if I skip nan values I have to adapt the calc

            if writer:
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Testing Loss: {epoch_test_loss:.4f}")
            sys.stdout.flush()
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print("Best model saved!")
            else:
                epochs_without_improvement += 1
                print('no improvement in test loss for', epochs_without_improvement, 'epochs')
            if epochs_without_improvement >= patience:
                    print("Early stopping!")
                    break
        # Check memory usage before cleanup
        print_memory_usage("Before cleanup")
        # Clear cache and run garbage collection after each epoch
        torch.cuda.empty_cache()  # Free up unallocated cached memory
        gc.collect()              # Run garbage collection to clear Python references
        # Check memory usage after cleanup
        print_memory_usage("After cleanup")
    print('training and testing done')
    return
    
def RS_val_model_newrmse_wtdtarget(model, validation_loader, device, writer=None):
    model.eval()
    running_val_loss = 0.0
    len_validation_loader = 0
    with torch.no_grad():
        for inputs, targets, masks in validation_loader:
            # inputs = inputs.float()# inputs.to(device).float()
            # targets = targets.float()#targets.to(device).float()
            # masks = masks.bool()#masks.to(device).bool()
            inputs = inputs.to(device).float()# inputs.to(device).float()
            targets = targets.to(device).float()#targets.to(device).float()
            masks = masks.to(device).bool()#masks.to(device).bool()

            # print('inputs val shape:', inputs.shape)
            # print('targets val shape:', targets.shape)
            # print('masks val shape:', masks.shape)

            if inputs.shape != targets.shape:
                inputs = inputs[:, :, 0, :, :]

            outputs = model(inputs)
        
            # print('outputs nan:', torch.isnan(outputs).any())
            if targets.shape != outputs.shape:
                targets = targets[:,0,:,:]
            if masks.shape != outputs.shape:
                masks = masks[0,0,:,:,:]

            # loss = rmse(outputs, targets, masks)
            loss = rmse_new(outputs, targets, masks)# / accumulation_steps
            print('val loss item:', loss.item())
        
            if not torch.isnan(loss):
                running_val_loss += loss.item()
                len_validation_loader += 1
                # else:
                #     print("NaN encountered in loss calculation. Skipping this instance.")
        val_loss = running_val_loss / len_validation_loader
        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/val_epoch', val_loss)
        # Print test results
        print(f"Val Loss: {val_loss:.4f}")
        sys.stdout.flush()
    print('validation done')
    return 
