
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim


def rmse(outputs, targets, masks):
    return torch.sqrt(F.mse_loss(outputs[masks], targets[masks]))

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
        all_outputs = np.concatenate(all_outputs, axis=0)

        # Print test results
        print(f"Test Loss: {test_loss:.4f}")

    return all_outputs

