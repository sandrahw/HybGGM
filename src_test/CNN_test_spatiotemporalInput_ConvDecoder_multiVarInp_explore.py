#CNN_test

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

input_data = xr.open_dataset(r'..\data\temp\input_rch.nc') # in this case, groundwater recharge
target_data = xr.open_dataset(r'..\data\temp\target.nc') # in this case, water table depth
input_data_top = xr.open_dataset(r'..\data\temp\top_uppermost_layer.nc') # upper most layer, topography?

# Select a region of the data for now
lon_bounds = (5, 10) #NL bounds(3,6)
lat_bounds = (45, 50)#NL bounds(50,54)

input_data = input_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
target_data = target_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
input_data_top = input_data_top.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds)) 



## Plot the region data
# test_i = input_data['Band1'].isel(time=1)
# test_o = target_data['Band1'].isel(time=1)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
# test_i.plot(ax=ax1)
# test_o.plot(ax=ax2)
# input_data_top['Band1'].plot(ax=ax3)
# ax1.set_title('recharge')
# ax2.set_title('wtd')
# ax3.set_title('topography')

# Convert xarray DataArrays to numpy arrays
input_array = input_data.to_array().values
input2_array = np.repeat(input_data_top.to_array().values, input_array.shape[1], axis=0) #reshape input2_array to match input_array by repeating the same array for every timestep in input_array
input3_array = target_data.to_array().values
target_array = target_data.to_array().values
mask = np.isnan(target_array) #mask of ocean/land


# create a mask for nan values of target_array and use on input arrays (all nan should be converted to zero)
# mask = np.isnan(target_array) #mask of ocean/land
# mask_binary = np.where(mask == True, 0, 1)
# mask_torch = torch.from_numpy(mask).float()
#Current idea is to introduce extra masked loss fct to account for the fact that the model should not learn from the ocean values
# most 0 values are already in 'the ocean' or as nan 
# nan's are transformed to 0

# use mask to replace values in input arrays with nan where target_array has nan
# input_array = np.where(mask, np.nan, input_array)
# input2_array = np.where(mask, np.nan, input2_array)
# input3_array = np.where(mask, np.nan, input3_array)

# reshape arrays for input to CNN
X1 = input_array[0, 1:, :, :] # recharge from 2nd month till last (because we want to include lagged values from wtd)
X2 = input2_array [1: , : , :]# topography
X3 = input3_array[0, :-1, :, :]# wtd from 1st month till second last
y = target_array[0, 1:, :, :] # wtd from 2nd month till last

# create input and target arrays for CNN
X = np.stack([X1, X2, X3], axis=1) #stack input arrays along the channel axis
y = y[:, np.newaxis, :, :]


# Step 1: Random Temporal Split
timespan = np.arange(0, X.shape[0]) #create a timespan array to include as a feature in the input array
train_months, test_months = train_test_split(timespan, test_size=0.3, random_state=42)

# Extract training and testing data for the temporal split
Xtrain_data_temporal = X[train_months, :, :, :]  # Shape: (train_months, 3, 360, 480)
Xtest_data_temporal = X[test_months, :, :, :]    # Shape: (test_months, 3, 360, 480)

ytrain_data_temporal = y[train_months, :, :, :]  # Shape: (train_months, 1, 360, 480)
ytest_data_temporal = y[test_months, :, :, :]    # Shape: (test_months, 1, 360, 480)

# Step 2: Random Spatial Split
lat_indices = np.arange(X.shape[2])
lon_indices = np.arange(X.shape[3])

# Randomly select grid points
train_lat_indices, test_lat_indices = train_test_split(lat_indices, test_size=0.3, random_state=42)
train_lon_indices, test_lon_indices = train_test_split(lon_indices, test_size=0.3, random_state=42)


# Extract training and testing data for the spatial split
Xtrain_data_spatial = Xtrain_data_temporal[:, :, train_lat_indices, :][:, :, :, train_lon_indices]
Xtest_data_spatial = Xtest_data_temporal[:, :, test_lat_indices, :][:, :, :, test_lon_indices]

ytrain_data_spatial = ytrain_data_temporal[:, :, train_lat_indices, :][:, :, :, train_lon_indices]
ytest_data_spatial = ytest_data_temporal[:, :, test_lat_indices, :][:, :, :, test_lon_indices]


# #idea for random sampling: reshape the array into a 2D array 
# # and then randomly sample from the valid indices 
# # (which would be based on the mask that still needs to be implemented)
# # this sampling is not yet temporal though I think
# X_flat = X.reshape(X.shape[0], X.shape[1], -1)
# y_flat = y.reshape(y.shape[0], y.shape[1], -1)

# # Indices of non-NaN cells
# valid_indices = np.where(mask)[0]  # Indices of valid cells
# # Randomly shuffle the indices
# np.random.shuffle(valid_indices)
# # Determine the split index (e.g., 80% for training, 20% for testing)
# split_idx = int(len(valid_indices) * 0.8)
# # Select train and test indices
# train_indices = valid_indices[:split_idx]
# test_indices = valid_indices[split_idx:]
# # Training data
# X_train = X_flat[:, :, train_indices]  # Shape: (11, 3, num_train_cells)
# y_train = y_flat[:, :, train_indices]  # Shape: (11, 1, num_train_cells)
# # Test data
# X_test = X_flat[:, :, test_indices]  # Shape: (11, 3, num_test_cells)
# y_test = y_flat[:, :, test_indices]  # Shape: (11, 1, num_test_cells)


# # # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_mask = ~np.isnan(X_train)
# test_mask = ~np.isnan(X_test)

# fill_value = 0  # or -9999, depending on your preference

# # Replace NaNs with the fill value
# X_train = np.nan_to_num(X_train, nan=fill_value)
# X_test = np.nan_to_num(X_test, nan=fill_value)
# y_train = np.nan_to_num(y_train, nan=fill_value)
# y_test = np.nan_to_num(y_test, nan=fill_value)

# Normalize data
def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    tensor -= mean 
    tensor /= std
    return   mean, std

inp_var_mean_train = [] # list to store normalisation information for denormalisation later
inp_var_std_train = []
inp_var_std_test = []
inp_var_mean_test = []
X_train = torch.from_numpy(Xtrain_data_spatial).float() #tranform into torch tensor
X_test = torch.from_numpy(Xtest_data_spatial).float()
for i in range(X_train.shape[1]):
    mean, std = normalize(X_train[:, i, :, :]) #normalise each variable separately
    inp_var_mean_train.append(mean)
    inp_var_std_train.append(std)

    mean, std = normalize(X_test[:, i, :, :]) #normalise each variable separately
    inp_var_mean_test.append(mean) # append mean and std of each variable to the list
    inp_var_std_test.append(std)

out_var_mean_train = []
out_var_std_train = []
out_var_std_test = []
out_var_mean_test = []
y_train = torch.from_numpy(ytrain_data_spatial).float() #transform into torch tensor
y_test = torch.from_numpy(ytest_data_spatial).float()
for i in range(y_train.shape[1]):
    mean, std = normalize(y_train[:, i, :, :])#normalise each variable separately
    out_var_mean_train.append(mean) # append mean and std of each variable to the list
    out_var_std_train.append(std)

    mean, std = normalize(y_test[:, i, :,:])
    out_var_mean_test.append(mean)
    out_var_std_test.append(std)

# plt.figure() #check normalisation of input variables and the distribution
# [plt.hist(X_train[:, i, :, :].flatten(),bins=np.arange(-5,5,0.1),histtype='step', label=i) for i in range(X_train.shape[1])]
# plt.legend()


# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: 3 channels (recharge, levels, topography)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Decoder (upsampling layers)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Output: 1 channel (WTD)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Decoder
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)  # No activation here, let the loss function handle it
        
        return x


# Instantiate the model, define the loss function and the optimizer
model = SimpleCNN()
# Define the loss function without reduction to retain per-element losses
criterion = nn.MSELoss()

# def masked_loss(output, target, mask):
#     loss = criterion(output, target)
#     print('loss shape', loss.shape)  # Debugging: print loss shape
#     expanded_mask = mask_torch[:,:1,:,:]  # Shape: (1, 1, 240, 240)
#     masked_loss = loss * expanded_mask  # Zero out loss where mask is 0 (ocean)
#     # Compute mean loss only over land areas
#     mean_loss = masked_loss.sum() / expanded_mask.sum()
#     return mean_loss

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            # print(f"Training input shape: {inputs.shape}")  # Debugging: print input shape
            # print(f"Training target shape: {targets.shape}")  # Debugging: print target shape

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Model output shape: {outputs.shape}")  # Debugging: print output shape

            # Check shapes before the error line
            # print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
            loss = criterion(outputs, targets)
            # loss = masked_loss(outputs, targets, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            # masked_outputs = outputs * mask_torch
            # masked_targets = targets * mask_torch # Zero out ocean values
            loss = criterion(outputs, targets)
            # normalized_loss = loss.sum()/mask_torch.sum()
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs   # Denormalize the outputs

test_outputs = evaluate_model(model, test_loader, criterion)


'''running the model on original data'''
'''extra test different region'''
input_data = xr.open_dataset(r'..\data\temp\input_rch.nc') # in this case, groundwater recharge
target_data = xr.open_dataset(r'..\data\temp\target.nc') # in this case, water table depth
input_data_top = xr.open_dataset(r'..\data\temp\top_uppermost_layer.nc') # upper most layer, topography?
# Select a region of the data for now
# lon_bounds = (5, 10) #example swiss bounds
# lat_bounds = (45, 50)#example swiss bounds
lon_bounds = (9, 14) #NL bounds(3,6)
lat_bounds = (50, 55)#NL bounds(50,54)

input_data = input_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
target_data = target_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
input_data_top = input_data_top.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds)) 



## Plot the region data
test_i = input_data['Band1'].isel(time=1)
test_o = target_data['Band1'].isel(time=1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
test_i.plot(ax=ax1)
test_o.plot(ax=ax2)
input_data_top['Band1'].plot(ax=ax3)
ax1.set_title('recharge')
ax2.set_title('wtd')
ax3.set_title('topography')

# Convert xarray DataArrays to numpy arrays
input_array = input_data.to_array().values
input2_array = np.repeat(input_data_top.to_array().values, input_array.shape[1], axis=0) #reshape input2_array to match input_array by repeating the same array for every timestep in input_array
input3_array = target_data.to_array().values
target_array = target_data.to_array().values


# reshape arrays for input to CNN
X1 = input_array[0, 1:, :, :] # recharge from 2nd month till last (because we want to include lagged values from wtd)
X2 = input2_array [1: , : , :]# topography
X3 = input3_array[0, :-1, :, :]# wtd from 1st month till second last
y = target_array[0, 1:, :, :] # wtd from 2nd month till last

# create input and target arrays for CNN
X = np.stack([X1, X2, X3], axis=1) #stack input arrays along the channel axis
y = y[:, np.newaxis, :, :]

# replace nan with 0 in X and y
X = np.nan_to_num(X, copy=False, nan=0)
y = np.nan_to_num(y, copy=False, nan=0)


inp_var_mean = [] # list to store normalisation information for denormalisation later
inp_var_std = []
X_run = torch.from_numpy(X).float() #tranform into torch tensor
for i in range(X.shape[1]):
    mean, std = normalize(X_run[:, i, :, :]) #normalise each variable separately
    inp_var_mean.append(mean)
    inp_var_std.append(std)

out_var_mean = []
out_var_std = []
y_run = torch.from_numpy(y).float() #transform into torch tensor
for i in range(y.shape[1]):
    mean, std = normalize(y_run[:, i, :, :])#normalise each variable separately
    out_var_mean.append(mean) # append mean and std of each variable to the list
    out_var_std.append(std)

dataset = TensorDataset(X_run, y_run)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

def run_model_on_full_data(model, data_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inp, tar in data_loader:
            outputs = model(inp)
            all_outputs.append(outputs.cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs

# Running the model on the entire dataset
y_pred_full = run_model_on_full_data(model, data_loader)

y_run_denorm = y_run*out_var_std[0].float() + out_var_mean[0].float() #denormalise the test outputs
y_pred_denorm = y_pred_full*out_var_std[0].numpy() + out_var_mean[0].numpy() #denormalise the predicted outputs
y_pred_denorm = y_pred_denorm[:, 0, :, :]

y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
y_run_denorm = y_run_denorm.flip(2)


for i in range(y_pred_denorm.shape[0])[:]:
    vmax = max([y_run_denorm.max(),y_pred_denorm.max()])
    vmin = min([y_run_denorm.min(),y_pred_denorm.min()])
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title('Actual Groundwater Depth')
    plt.imshow(y_run_denorm[i, 0, :, :], cmap='viridis',vmin=vmin,vmax=vmax)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Predicted Groundwater Depth')
    plt.imshow(y_pred_denorm[i], cmap='viridis',vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.title(f"Model Output {i+1}")

    diff = y_run_denorm[i, 0, :, :] - y_pred_denorm[i]
    diff_del = y_pred_denorm[i] - y_run_denorm[i-1, 0, :, :] #delta between predicted and previous actual
    diff_act = y_run_denorm[i, 0, :, :] - y_run_denorm[i-1, 0, :, :]  
    #TODO scatterplot of y_pred_denorm vs y_run_denorm
    #TODO also plot diff_del
    vmax_diff = max([diff.max(),diff_act.max()])
    vmin_diff = min([diff.min(),diff_act.min()])
    
    plt.subplot(2, 2, 3)
    plt.title('Predicted Groundwater Depth')
    plt.text(0, 60, 'min=%s, max=%s' %(np.array(diff).min(),np.array(diff).max()), fontsize = 12)
    # pcm = plt.imshow(diff, cmap='RdBu', norm=colors.CenteredNorm())#, norm=SymLogNorm(linthresh=1))
    pcm = plt.imshow(diff, cmap='RdBu', norm=SymLogNorm(linthresh=1))
    plt.colorbar(pcm, orientation='vertical')
    plt.title(f"Difference OG-Pred {i+1}")
    plt.tight_layout()

    if i == 0:
        continue
    else:
        plt.subplot(2, 2, 4)
        plt.title('Actual Groundwater Depth')
        plt.text(0, 60, 'min=%s, max=%s' %(np.array(diff_act).min(),np.array(diff_act).max()), fontsize = 12)
        # pcm = plt.imshow(diff_act, cmap='RdBu', norm=colors.CenteredNorm())#, norm=SymLogNorm(linthresh=1))norm=SymLogNorm(linthresh=1))
        pcm = plt.imshow(diff_act, cmap='RdBu', norm=SymLogNorm(linthresh=1))# norm=SymLogNorm(linthresh=1))
        plt.colorbar(pcm, orientation='vertical')
        plt.title(f"Difference OG{i} - OG{i+1}")    
        plt.tight_layout()

        #plot diff_del
        


# highlight regions on original map by drawing a rectangle based on the lat and lon bounds
lon_bounds_old = (5, 10) #example swiss bounds
lat_bounds_old = (45, 50)#example swiss bounds
map = xr.open_dataset(r'..\data\temp\target.nc') # in this case, water table depth
# map = map.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
map = map['Band1'].isel(time=1)
fig, ax = plt.subplots()
map.plot(ax=ax)
ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
ax.add_patch(plt.Rectangle((lon_bounds_old[0], lat_bounds_old[0]), lon_bounds_old[1] - lon_bounds_old[0], lat_bounds_old[1] - lat_bounds_old[0], fill=None, color='blue'))
plt.show()