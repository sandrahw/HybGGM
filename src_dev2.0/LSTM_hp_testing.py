from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import xarray as xr
import matplotlib.pyplot as plt


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
# hidden_size = 16
# num_layers = 2
batchSize = 16
# lr_rate = 0.001
# def_epochs = 10
targetvar = 'head'
patience = 5
trainsize = 0.3
testsize= 0.3
# dropoutrate = 0.1
valsize = 1 - trainsize - testsize
# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs_dev2\%s_LSTM_hp' %(targetvar)
log_dir_fig = r'..\training\logs_dev2\%s_LSTM_hp\figures' %(targetvar)

#create folder in case not there yet
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)

'''create mask (for land/ocean)'''
map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
map_cut = map_tile.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
plt.imshow(map_cut.Band1[0, :, :])
mask = map_cut.to_array().values
# mask where everything that is nan is 0 and everything else is 1
mask = np.nan_to_num(mask, copy=False, nan=0)
mask = np.where(mask==0, 0, 1)
mask = mask[0, :, :]
# plt.imshow(mask[0, :, :])
np.save(r'%s\mask.npy'%log_directory, mask)

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
for i in range(X.shape[1])[:]:
    mean = X[:, i, :, :].mean()
    std = X[:, i, :, :].std()
    # check if every value in array is 0, if so, skip normalisation
    if X[:, i, :, :].max() == 0 and X[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        X_temp = X[:, i, :, :]
    else:
        X_temp = (X[:, i, :, :] - mean) / std
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
    mean = y[:, i, :, :].mean()
    std = y[:, i, :, :].std()
    # check if every value in array is 0, if so, skip normalisation
    if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        y_temp = y[:, i, :, :]
    else:
        y_temp = (y[:, i, :, :] - mean) / std
    y_temp = (y[:, i, :, :] - mean) / std
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
# np.random.shuffle(location_indices)

#create a list of indices like num_locations for the mask, then map out which indices correspond to 0 values in the mask, so that those values can be dropped from num_indices
num_locations_mask = mask.shape[1] * mask.shape[2]  
num_locations_mask_indices = np.arange(num_locations_mask)
mask_indices = np.where(mask.ravel() == 0)[0]
location_indices_mask = np.delete(num_locations_mask_indices, mask_indices)

np.random.shuffle(location_indices_mask)
train_size = int(trainsize * num_locations)
test_size = int(testsize * num_locations)

train_indices = location_indices_mask[:train_size]
test_indices = location_indices_mask[train_size:train_size + test_size]
val_indices = location_indices_mask[train_size + test_size:]


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
print(next(iter(val_loader))[0].shape)

# # LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTM, self).__init__()
        # LSTM with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        # Fully connected layers
        self.fc = nn.Linear(hidden_size, output_size)
        # # Dropout after LSTM output
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs = inputs.permute(0, 1, 2)  # [batch_size, seq_len, features]
        # LSTM layer
        lstm_out, _ = self.lstm(inputs)
        # Apply dropout to the LSTM output
        # lstm_out = self.dropout(lstm_out)
        # Fully connected layer for sequence prediction
        out = self.fc(lstm_out)
        return out

# writer = SummaryWriter(log_dir=log_directory)
# model = LSTM(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout_rate=dropoutrate).to(device)

# Create a wrapper for the LSTM model
class LSTMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_size=X.shape[1], hidden_size=64, num_layers=1, output_size=1, dropout_rate=0.3, learning_rate=1e-3, epochs=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None

    def fit(self, X_t, y_t, mask_t=None):
        self.model = LSTM(self.input_size, self.hidden_size, self.num_layers, self.output_size, self.dropout_rate)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_dataset = TemporalDataset(X_t, y_t, mask_t)
        dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(X, dtype=torch.float32))
        return predictions.numpy()


# Define parameter grid
param_grid = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout_rate": [0.1, 0.3, 0.5],
    "learning_rate": [1e-3, 1e-4],
}

# Run GridSearchCV
grid_search = GridSearchCV(LSTMWrapper(input_size=X_norm_arr.shape[1]), param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_norm_arr, y_norm_arr, train_mask)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# Extract results as a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Save results to a CSV file
results.to_csv("grid_search_results.csv", index=False)

# Display the top results
print(results.sort_values("mean_test_score", ascending=False).head())