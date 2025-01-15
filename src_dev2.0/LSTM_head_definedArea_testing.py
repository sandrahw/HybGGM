import os
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

''' Set random seed for reproducibility also for different torch applications'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False     # Turn off optimization that introduces randomness


def data_prep(f, lat_bounds, lon_bounds, data_length):
    '''function to prepare the data for input by:
     - cropping the data to the specified lat and lon bounds for test regions, 
     - transforming the data to numpy arrays,
     - and additionally dealing with nan and inf values by setting them to 0 
     #TODO find a better way to deal with nan and inf values
    '''
    param = f.split('\\')[-1].split('.')[0]
    data = xr.open_dataset(f)
    data_cut = data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
    if param in params_monthly:
        data_arr = data_cut.to_array().values
        data_arr = data_arr[0, :, :, :] 
        data_arr = np.nan_to_num(data_arr, copy=False, nan=0)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)
        lat = data_cut.lat.values
        lon = data_cut.lon.values
        time = data_cut.time.values
       

    if param in params_initial: #repeat the initial values for each month as they are static
        data_arr = np.repeat(data_cut.to_array().values, data_length, axis=0)
        data_arr = np.where(data_arr==np.nan, 0, data_arr)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)
        lat = data_cut.lat.values
        lon = data_cut.lon.values
        time = np.arange(data_length)
    np.save(r'%s\%s.npy'%(log_directory, param), data_arr)
    return data_arr, lat, lon, time


# Set seed before any training or data loading happens
set_seed(10)


'''define general path, data length, case area, number of epochs, learning rate and batch size'''
# define the number of months in the dataset and the number of epochs
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
hidden_size = 16
num_layers = 2
batchSize = 10
lr_rate = 0.001
def_epochs = 50
targetvar = 'head'
patience = 5
trainsize = 0.1
testsize= 0.1
valsize = 1 - trainsize - testsize
# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs_dev2\%s_%s_%s_%s_%s_%s_LSTM_tr0.1_newlstm' %(targetvar, def_epochs, lr_rate ,batchSize, hidden_size, num_layers)
log_dir_fig = r'..\training\logs_dev2\%s_%s_%s_%s_%s_%s_LSTM_tr0.1_newlstm\figures' %(targetvar, def_epochs, lr_rate ,batchSize, hidden_size, num_layers)
#create folder in case not there yet
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)

'''load the modflow files and prepare the data for input'''
inFiles = glob.glob(r'..\data\temp\*.nc') #load all input files in the folder
# modflow files that are saved monthly
params_monthly = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'surface_water_bed_elevation_used',
 'surface_water_elevation', 'net_RCH', 'wtd']
# other modflow files that seem to be static parameters
params_initial = ['bottom_lowermost_layer', 'bottom_uppermost_layer', 
 'drain_conductance', 
 'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
 'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
 'top_uppermost_layer',
 'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer']

params_sel = ['abstraction_uppermost_layer',
              'bed_conductance_used',
              'bottom_uppermost_layer',
              'drain_conductance',
              'drain_elevation_uppermost_layer',
              'horizontal_conductivity_uppermost_layer',
              'initial_head_uppermost_layer',
              'net_RCH', 
              'primary_storage_coefficient_uppermost_layer',
              'surface_water_bed_elevation_used',
              'surface_water_elevation',
              'top_uppermost_layer',
              'vertical_conductivity_uppermost_layer',
              'wtd'
            ]   
# select the files that are needed for the input
selInFiles = [f for f in inFiles if f.split('\\')[-1].split('.')[0] in params_sel] 
'''prepare the data for input by cropping the data to the specified lat and lon bounds for test regions'''
datacut = []
for f in selInFiles[:]:
    print(f.split('\\')[-1].split('.')[0])
    param = f.split('\\')[-1].split('.')[0]
    data, lat, lon, time = data_prep(f, lat_bounds, lon_bounds, data_length)
    if param == 'wtd':
        wtd = data
        continue
    if param == 'top_uppermost_layer':
        top_upper = data
        datacut.append(data)
    else:
        datacut.append(data)

''''calculate the head for each month - define target (y) and input (X) arrays for the CNN'''
target_head = top_upper - wtd #calculate the head for each month
X_all = np.stack(datacut, axis=1)
X = X_all
y = target_head[:, np.newaxis, :, :] 
np.save(r'%s\X.npy'%log_directory, X)
np.save(r'%s\y.npy'%log_directory, y)

'''normalising the data for every array and save mean and std for denormalisation'''
inp_var_mean = [] # list to store normalisation information for denormalisation later
inp_var_std = []
X_norm = []
for i in range(X.shape[1]):
    mean = X[:, i, :, :].mean() # calculate mean and std for each array
    std = X[:, i, :, :].std() # calculate mean and std for each array
    # print('min',X[:, i, :, :].min(), 'max', X[:, i, :, :].max())
    # print('mean', mean, 'std', std)

    # mean = X[:, i, :, :].mean(axis=0)  # Mean per grid cell
    # std = X[:, i, :, :].std(axis=0)  # Std per grid cell
    # std[std == 0] = 1 
    # check if every value in array is 0, if so, skip normalisation
    if X[:, i, :, :].max() == 0 and X[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        X_temp = X[:, i, :, :]
    else:
        X_temp = (X[:, i, :, :] - mean) / std
    # print(mean, std, X_temp)
    # print('min',X_temp.min(), 'max', X_temp.max())
    X_norm.append(X_temp)
    inp_var_mean.append(mean)
    inp_var_std.append(std)
X_norm_arr = np.array(X_norm)
X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3)
np.save(r'%s\X_norm_arr.npy'%log_directory, X_norm_arr)
np.save(r'%s\inp_var_mean.npy'%log_directory, inp_var_mean)
np.save(r'%s\inp_var_std.npy'%log_directory, inp_var_std)

out_var_mean = []
out_var_std = []
y_norm = []
for i in range(y.shape[1]):
    y_sqrt = np.sqrt(y[:, i, :, :])
    mean = y_sqrt.mean() # calculate mean and std for each array
    std = y_sqrt.std() # calculate mean and std for each array
    # mean = y[:, i, :, :].mean(axis=0)  # Mean per grid cell
    # std = y[:, i, :, :].std(axis=0)  # Std per grid cell
    # std[std == 0] = 1 
    # check if every value in array is 0, if so, skip normalisation
    # print('min',y[:, i, :, :].min(), 'max', y[:, i, :, :].max())
    # print('mean', mean, 'std', std)
    if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        y_temp = y_sqrt
    else:
        y_temp = (y_sqrt - mean) / std
    y_temp = (y_sqrt - mean) / std
    # print('min aftern',y_temp.min(), 'max aftern', y_temp.max())
    y_norm.append(y_temp)
    out_var_mean.append(mean)
    out_var_std.append(std)
y_norm_arr = np.array(y_norm)
y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
np.save(r'%s\y_norm_arr.npy'%log_directory, y_norm_arr)
np.save(r'%s\out_var_mean.npy'%log_directory, out_var_mean)
np.save(r'%s\out_var_std.npy'%log_directory, out_var_std)


'''split the data into training, validation and test sets by choosing random indices'''
num_locations = X.shape[2] * X.shape[3]
location_indices = np.arange(num_locations)
np.random.shuffle(location_indices)

train_size = int(trainsize * num_locations)
test_size = int(testsize * num_locations)

train_indices = location_indices[:train_size]
test_indices = location_indices[train_size:train_size + test_size]
val_indices = location_indices[train_size + test_size:]


def create_masks(indices, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[np.unravel_index(indices, shape)] = True
    return mask

train_mask = create_masks(train_indices, (X.shape[2], X.shape[3]))
test_mask = create_masks(test_indices, (X.shape[2], X.shape[3]))
val_mask = create_masks(val_indices, (X.shape[2], X.shape[3]))

full_mask = np.zeros((X.shape[2], X.shape[3]), dtype=bool)
full_mask[:] = True

# Dataset class
class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, target, mask):
        self.mask = mask
        self.inputs = inputs[:,:, mask]
        self.target = target[:,:, mask]

    def __len__(self):
        return self.mask.sum()

    def __getitem__(self, idx):
        inputs_idx = self.inputs.transpose(2,0,1)[idx]
        target_idx = self.target.T[idx]
        return torch.tensor(inputs_idx, dtype=torch.float32), \
               torch.tensor(target_idx, dtype=torch.float32)

train_dataset = TemporalDataset(X_norm_arr, y_norm_arr, train_mask)
test_dataset = TemporalDataset(X_norm_arr, y_norm_arr, test_mask)
val_dataset = TemporalDataset(X_norm_arr, y_norm_arr, val_mask)
all_dataset = TemporalDataset(X_norm_arr, y_norm_arr, full_mask)

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=batchSize, shuffle=False)
print(next(iter(train_loader))[0].shape)

# # LSTM model
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, inputs):
#         batch_size = inputs.size(0)
#         inputs = inputs.permute(0, 1, 2)  # [batch_size, seq_len, features]
#         lstm_out, _ = self.lstm(inputs)
#         out = self.fc(lstm_out)
#         return out

# writer = SummaryWriter(log_dir=log_directory)
# model = LSTM(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)

class ImprovedFullSequenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(ImprovedFullSequenceLSTM, self).__init__()
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    batch_first=True, 
                                    dropout=dropout_rate)
        
        # Batch Normalization for stabilization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Decoder LSTM (optional, can use the same LSTM for simplicity)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    batch_first=True, 
                                    dropout=dropout_rate)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, inputs):
        # Encoder LSTM
        encoder_out, _ = self.encoder_lstm(inputs)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Batch normalization
        encoder_out = self.batch_norm(encoder_out.permute(0, 2, 1)).permute(0, 2, 1)  # Normalizing across features
        
        # Decoder LSTM
        decoder_out, _ = self.decoder_lstm(encoder_out)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Fully connected layers for sequence prediction
        output = self.fc(decoder_out)  # Shape: [batch_size, seq_len, output_size]
        
        return output    

writer = SummaryWriter(log_dir=log_directory)
model = ImprovedFullSequenceLSTM(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
''' RS target wtd'''
def rmse_lstm(outputs, targets):
    diff = (targets - outputs)
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

# Training loop
def LSTM_train_test_model(model, train_loader, test_loader, epochs):
    best_test_loss = float('inf')
    # patience_counter = 0

    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        # train_loss = 0
        running_train_loss = 0.0
        len_train_loader = 0
        len_test_loader = 0
        print(next(iter(train_loader))[0].shape)
        for i, (inputs, target) in enumerate(train_loader):
            # print(i, 'out of', len(train_loader))
            inputs, targets = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # target = target.unsqueeze(-1)  # Add a last dimension if needed, shape: [batch_size, seq_len, 1]
            
            # #loss = criterion(outputs, target)
            # print('target', target.shape)   
            outputs = outputs.permute(0, 2, 1)
            # print('outputs', outputs.shape)
            loss = rmse_lstm(outputs, targets)
            loss.backward()
            optimizer.step()
            if not torch.isnan(loss):
                running_train_loss += loss.item()# * accumulation_steps  # Multiply back for tracking
                len_train_loader += 1
            # train_loss += loss.item()
        epoch_train_loss = running_train_loss / len_train_loader #if I skip nan values I have to adapt the calc
        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)
        # Test the model
        print('testing phase')
        # train_loss /= len(train_loader)
        model.eval()
        # test_loss = 0
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, target in test_loader:
                inputs, targets = inputs.to(device), target.to(device)
                outputs = model(inputs)
                # target = target.unsqueeze(-1)
                # print('target', target.shape)
                # print('outputs', outputs.shape)
                outputs = outputs.permute(0, 2, 1)
                # print('outputs', outputs.shape)
                # test_loss += criterion(outputs, target).item()
                test_loss = rmse_lstm(outputs, targets)
                if not torch.isnan(test_loss):
                    running_test_loss += test_loss.item()
                    len_test_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
            epoch_test_loss = running_test_loss / len_test_loader #if I skip nan values I have to adapt the calc
            if writer:
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)

            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_train_loss:.4f}, Testing Loss: {epoch_test_loss:.4f}")
  
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(log_directory, 'best_model.pth'))
                print("Best model saved!")
            else:
                epochs_without_improvement += 1
                print('no improvement in test loss for', epochs_without_improvement, 'epochs')
            if epochs_without_improvement >= patience:
                    print("Early stopping!")
                    break
    print('training and testing done')
    return

def LSTM_val_model(model, data_loader, device):
    """
    Perform predictions using a PyTorch model.

    Args:
        model (torch.nn.Module): Trained PyTorch LSTM model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform predictions on (CPU or GPU).

    Returns:
        numpy.ndarray: Predicted values.
    """
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    len_validation_loader = 0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in data_loader:
            # Move inputs to the device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            # print('target', targets.shape)
            # print('outputs', outputs.shape)
            outputs = outputs.permute(0, 2, 1)
            # print('outputs', outputs.shape)
            loss = rmse_lstm(outputs, targets)
            if not torch.isnan(loss):
                running_val_loss += loss.item()
                len_validation_loader += 1

        val_loss = running_val_loss / len_validation_loader
        if writer:
            writer.add_scalar('Loss/val_epoch', val_loss)
        # Print test results
        print(f"Val Loss: {val_loss:.4f}")
    # Concatenate predictions from all batches
    return 

LSTM_train_test_model(model, train_loader, test_loader, def_epochs)
LSTM_val_model(model, val_loader, device)  # Use the predict function

def map_predictions_to_full_grid(predictions, val_mask, grid_shape):
    """
    Map predictions to their original spatial positions using a validation mask.

    Args:
        predictions (numpy.ndarray): Predicted values, shape [n_samples, time].
        val_mask (numpy.ndarray): Boolean mask indicating validation locations, shape [lat, lon].
        grid_shape (tuple): Full grid shape as (time, lat, lon).

    Returns:
        numpy.ndarray: Full grid with predictions placed at locations specified by val_mask.
    """
    time_steps, lat, lon = grid_shape
    full_grid = np.full((time_steps, lat, lon), np.nan)  # Initialize with NaNs

    # Map predictions to the full grid
    flat_mask = val_mask.ravel()  # Flatten the validation mask
    flat_grid = full_grid.reshape(time_steps, -1)  # Flatten spatial dimensions of the full grid

    flat_grid[:, flat_mask] = predictions.T  # Assign predictions to the masked positions
    return full_grid

def LSTM_run_model(model, data_loader, device):
    """
    Perform predictions using a PyTorch model.

    Args:
        model (torch.nn.Module): Trained PyTorch LSTM model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform predictions on (CPU or GPU).

    Returns:
        numpy.ndarray: Predicted values.
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    with torch.no_grad():  # Disable gradient computation
        for inputs, _ in data_loader:
            # Move inputs to the device
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            # Collect predictions
            predictions.append(outputs.cpu().numpy())
    # Concatenate predictions from all batches
    return np.concatenate(predictions, axis=0)

# y_pred_val =LSTM_run_model(model, val_loader, device)  # Use the predict function
# grid_shape = (y_pred_val.shape[1], val_mask.shape[0], val_mask.shape[1])  # (time, lat, lon)
# val_pred_grid = map_predictions_to_full_grid(y_pred_val, val_mask, grid_shape)
# #denormalise the validation predictions
# val_pred_denorm= val_pred_grid.reshape(71, val_mask.shape[0], val_mask.shape[1])*out_var_std[0] + out_var_mean[0]

# # transform the denormalised predictions to an xarray DataArray, with the correct dimensions and coordinates based on
# val_pred = xr.DataArray(val_pred_denorm, dims=['time', 'lat', 'lon'], coords={'time': np.arange(1, data_length), 'lat': lat, 'lon': lon})
# val_pred.to_netcdf(r'%s\val_pred_denorm.nc'%log_directory)
# val_pred[0].plot()

#full prediction
y_pred_full = LSTM_run_model(model, all_loader, device)  # Use the predict function

grid_shape = (y_pred_full.shape[1], full_mask.shape[0], full_mask.shape[1])  # (time, lat, lon)
y_pred_full_grid = map_predictions_to_full_grid(y_pred_full, full_mask, grid_shape)
y_pred_full_denorm= ((y_pred_full_grid * out_var_std[0]) + out_var_mean[0])**2
y_pred_full_denorm = xr.DataArray(y_pred_full_denorm, dims=['time', 'lat', 'lon'], coords={'time': np.arange(0, data_length), 'lat': lat, 'lon': lon})  
y_pred_full_denorm.to_netcdf(r'%s\full_pred.nc'%log_directory)

#plot the full prediction
y_pred_full_denorm[0].plot()

min_val = np.min([np.nanmin(y), np.nanmin(y_pred_full_denorm)])
max_val = np.max([np.nanmax(y), np.nanmax(y_pred_full_denorm)])
lim = np.max([np.abs(min_val), np.abs(max_val)])

diff = y[0, 0, :, :] - y_pred_full_denorm[0, :, :]
mindiff = np.nanmin(np.percentile(diff, 1))
maxdiff = np.nanmax(np.percentile(diff, 99))
limdiff = np.max([np.abs(mindiff), np.abs(maxdiff)])
plt.figure(figsize=(15,10))
plt.subplot(1, 3, 1)
plt.imshow(y_pred_full_denorm[0, :, :], cmap='viridis', label='predicted',vmin=min_val, vmax=max_val)    
plt.colorbar(shrink=0.5)
plt.title('predicted')
plt.subplot(1, 3, 2)
plt.imshow(y[0, 0, :, :], cmap='viridis', label='target',vmin=min_val, vmax=max_val)
plt.title('target')
plt.colorbar(shrink=0.5)
plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='RdBu', label='diff',vmin=-limdiff, vmax=limdiff)
plt.title('difference')
plt.colorbar(shrink=0.5)
plt.savefig(r'%s\prediction.png'%log_dir_fig)


plt.figure(figsize=(10,10))
plt.scatter(y[0, 0, :, :], y_pred_full_denorm[0, :, :])
plt.xlabel('target')
plt.ylabel('predicted')
plt.savefig(r'%s\scatterplot.png'%log_dir_fig)