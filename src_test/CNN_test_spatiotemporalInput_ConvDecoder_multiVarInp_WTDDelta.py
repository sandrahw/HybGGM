#CNN_test

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
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors

random.seed(10)
print(random.random())

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

lon_bounds = (5, 10) #NL bounds(3,6)
lat_bounds = (45, 50)#NL bounds(50,54)

data_length = 12 # number of months in current dataset

inFiles = glob.glob(r'..\data\temp\*.nc')

params_monthly = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'surface_water_bed_elevation_used',
 'surface_water_elevation', 'input_rch', 'target']

params_initial = ['bottom_lowermost_layer', 'bottom_uppermost_layer', 
 'drain_conductance', 
 'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
 'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
 'top_uppermost_layer',
 'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer']

def data_prep(f, lat_bounds, lon_bounds, data_length):
    param = f.split('\\')[-1].split('.')[0]
    data = xr.open_dataset(f)
    data_cut = data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
    if param in params_monthly:
        data_arr = data_cut.to_array().values
        data_arr = data_arr[0, :, :, :]
        data_arr = np.nan_to_num(data_arr, copy=False, nan=0)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)

    if param in params_initial:
        data_arr = np.repeat(data_cut.to_array().values, data_length, axis=0)
        data_arr = np.where(data_arr==np.nan, 0, data_arr)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)
    return data_arr

abs_lower = data_prep(inFiles[0], lat_bounds, lon_bounds, data_length)
abs_upper = data_prep(inFiles[1], lat_bounds, lon_bounds, data_length)
bed_cond = data_prep(inFiles[2], lat_bounds, lon_bounds, data_length)
bottom_lower = data_prep(inFiles[3], lat_bounds, lon_bounds, data_length)
bottom_upper = data_prep(inFiles[4], lat_bounds, lon_bounds, data_length)
drain_cond = data_prep(inFiles[5], lat_bounds, lon_bounds, data_length)
drain_elev_lower = data_prep(inFiles[6], lat_bounds, lon_bounds, data_length)
drain_elev_upper = data_prep(inFiles[7], lat_bounds, lon_bounds, data_length)
hor_cond_lower = data_prep(inFiles[8], lat_bounds, lon_bounds, data_length)
hor_cond_upper = data_prep(inFiles[9], lat_bounds, lon_bounds, data_length)
init_head_lower = data_prep(inFiles[10], lat_bounds, lon_bounds, data_length)
init_head_upper = data_prep(inFiles[11], lat_bounds, lon_bounds, data_length)
recharge = data_prep(inFiles[12], lat_bounds, lon_bounds, data_length)
prim_stor_coeff_lower = data_prep(inFiles[13], lat_bounds, lon_bounds, data_length)
prim_stor_coeff_upper = data_prep(inFiles[14], lat_bounds, lon_bounds, data_length)
surf_wat_bed_elev = data_prep(inFiles[15], lat_bounds, lon_bounds, data_length)
surf_wat_elev = data_prep(inFiles[16], lat_bounds, lon_bounds, data_length)
wtd = data_prep(inFiles[17], lat_bounds, lon_bounds, data_length)
top_upper = data_prep(inFiles[18], lat_bounds, lon_bounds, data_length)
vert_cond_lower = data_prep(inFiles[19], lat_bounds, lon_bounds, data_length)
vert_cond_upper = data_prep(inFiles[20], lat_bounds, lon_bounds, data_length)

# plot wtd for one month, indicating range of values
plt.imshow(wtd[0, :, :], cmap='viridis')
plt.colorbar()
plt.title('WTD for one month')

# calculate the delta wtd for each month
delta_wtd = np.diff(wtd, axis=0)

y = delta_wtd[:, np.newaxis, :, :]
X = np.stack([abs_lower[:-1,:,:], abs_upper[:-1,:,:], 
              bed_cond[:-1,:,:], 
              bottom_lower[:-1,:,:], bottom_upper[:-1,:,:], 
              drain_cond[:-1,:,:], drain_elev_lower[:-1,:,:], drain_elev_upper[:-1,:,:], 
              hor_cond_lower[:-1,:,:], hor_cond_upper[:-1,:,:], 
              init_head_lower[:-1,:,:], init_head_upper[:-1,:,:], 
              recharge[:-1,:,:], 
              prim_stor_coeff_lower[:-1,:,:], prim_stor_coeff_upper[:-1,:,:], 
              surf_wat_bed_elev[:-1,:,:], surf_wat_elev[:-1,:,:], 
              top_upper[:-1,:,:], 
              vert_cond_lower[:-1,:,:], vert_cond_upper[:-1,:,:], #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
              wtd[:-1, :, :]
              ], axis=1)


#check if nan values in X
# for i in range(X.shape[1]):
#     print(f"Number of NaN values in variable {i}: {np.isnan(X[:, i, :, :]).sum()}")
# for i in range(X.shape[1]):   
#     print(f"Number of inf values in variable {i}: {np.isinf(X[:, i, :, :]).sum()}")
# for i in range(y.shape[1]):
#     print(f"Number of NaN values in variable {i}: {np.isnan(y[:, i, :, :]).sum()}")
#     print(f"Number of inf values in variable {i}: {np.isinf(y[:, i, :, :]).sum()}")


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
        self.conv1 = nn.Conv2d(21, 16, kernel_size=3, padding=1)  # Input: 3 channels (recharge, levels, topography)
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


# '''extra test different region'''
# input_data = xr.open_dataset(r'..\data\temp\input_rch.nc') # in this case, groundwater recharge
# target_data = xr.open_dataset(r'..\data\temp\target.nc') # in this case, water table depth
# input_data_top = xr.open_dataset(r'..\data\temp\top_uppermost_layer.nc') # upper most layer, topography?
# # Select a region of the data for now
# # lon_bounds = (5, 10) #example swiss bounds
# # lat_bounds = (45, 50)#example swiss bounds
# lon_bounds = (9, 14) #NL bounds(3,6)
# lat_bounds = (50, 55)#NL bounds(50,54)

# input_data = input_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
# target_data = target_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
# input_data_top = input_data_top.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds)) 



# ## Plot the region data
# test_i = input_data['Band1'].isel(time=1)
# test_o = target_data['Band1'].isel(time=1)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
# test_i.plot(ax=ax1)
# test_o.plot(ax=ax2)
# input_data_top['Band1'].plot(ax=ax3)
# ax1.set_title('recharge')
# ax2.set_title('wtd')
# ax3.set_title('topography')

# # Convert xarray DataArrays to numpy arrays
# input_array = input_data.to_array().values
# input2_array = np.repeat(input_data_top.to_array().values, input_array.shape[1], axis=0) #reshape input2_array to match input_array by repeating the same array for every timestep in input_array
# input3_array = target_data.to_array().values
# target_array = target_data.to_array().values


# # reshape arrays for input to CNN
# X1 = input_array[0, 1:, :, :] # recharge from 2nd month till last (because we want to include lagged values from wtd)
# X2 = input2_array [1: , : , :]# topography
# X3 = input3_array[0, :-1, :, :]# wtd from 1st month till second last
# y = target_array[0, 1:, :, :] # wtd from 2nd month till last

# # create input and target arrays for CNN
# X = np.stack([X1, X2, X3], axis=1) #stack input arrays along the channel axis
# y = y[:, np.newaxis, :, :]

# # replace nan with 0 in X and y
# X = np.nan_to_num(X, copy=False, nan=0)
# y = np.nan_to_num(y, copy=False, nan=0)


y = delta_wtd[:, np.newaxis, :, :]
X = np.stack([abs_lower[:-1,:,:], abs_upper[:-1,:,:], 
              bed_cond[:-1,:,:], 
              bottom_lower[:-1,:,:], bottom_upper[:-1,:,:], 
              drain_cond[:-1,:,:], drain_elev_lower[:-1,:,:], drain_elev_upper[:-1,:,:], 
              hor_cond_lower[:-1,:,:], hor_cond_upper[:-1,:,:], 
              init_head_lower[:-1,:,:], init_head_upper[:-1,:,:], 
              recharge[:-1,:,:], 
              prim_stor_coeff_lower[:-1,:,:], prim_stor_coeff_upper[:-1,:,:], 
              surf_wat_bed_elev[:-1,:,:], surf_wat_elev[:-1,:,:], 
              top_upper[:-1,:,:], 
              vert_cond_lower[:-1,:,:], vert_cond_upper[:-1,:,:], #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
              wtd[:-1, :, :]
              ], axis=1)
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
y_pred_full = run_model_on_full_data(model, data_loader) #this is now the delta wtd

y_run_denorm = y_run*out_var_std[0].float() + out_var_mean[0].float() #denormalise the original delta wtd
y_pred_denorm = y_pred_full*out_var_std[0].numpy() + out_var_mean[0].numpy() #denormalise the predicted delta wtd

y_pred_denorm = y_pred_denorm[:, 0, :, :]
y_run_denorm = y_run_denorm[:, 0, :, :].numpy()

y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
y_run_denorm = np.flip(y_run_denorm, axis=1)#.numpy() #not necessarily needed to plot?
X_wtd = np.flip(wtd[:-1, :, :],axis=1)

for i in range(y_pred_denorm.shape[0])[:-1]:
    # print(i)

    pred_wtd = X_wtd[i] + y_pred_denorm[i]

    vmax = max([pred_wtd.max(),X_wtd[i+1, :, :].max()])
    vmin = min([pred_wtd.min(),X_wtd[i+1, :, :].min()])
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.title('Actual WTD (OG)')
    plt.imshow(X_wtd[i+1, :, :], cmap='viridis',vmin=vmin,vmax=vmax)
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title('Predicted WTD')
    plt.imshow(pred_wtd, cmap='viridis',vmin=vmin,vmax=vmax)
    plt.colorbar()

    diff = X_wtd[i+1, :, :] - pred_wtd
    plt.subplot(2, 3, 3)
    plt.title('diff actual vs sim wtd')
    pcm = plt.imshow(diff, cmap='RdBu', norm=colors.CenteredNorm())
    plt.colorbar(pcm, orientation='vertical')
    plt.tight_layout()

    plt.subplot(2, 3, 4)
    pcm = plt.imshow(y_run_denorm[i,:,:], cmap='RdBu', norm=SymLogNorm(linthresh=1))
    plt.colorbar(pcm, orientation='vertical')
    plt.title(f"original delta WTD")
    plt.tight_layout()

    plt.subplot(2, 3, 5)
    pcm = plt.imshow(y_pred_denorm[i,:,:], cmap='RdBu', vmax=0.1, vmin=-0.1)#, norm=colors.CenteredNorm())# norm=SymLogNorm(linthresh=1))
    plt.colorbar(pcm, orientation='vertical')
    plt.title(f"predicted delta WTD")    
    plt.tight_layout()

    vmax = max([X_wtd[i, :, :].max(),pred_wtd.max()])
    vmin = min([X_wtd[i, :, :].min(),pred_wtd.min()])
    plt.subplot(2, 3, 6)
    plt.scatter(X_wtd[i, :, :].flatten(), pred_wtd.flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('Actual WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated WTD')
    plt.title(f"Actual vs Sim wtd")
    plt.tight_layout()

    plt.savefig(r'..\data\temp\plots\plot_%s.png' %(i))

        


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