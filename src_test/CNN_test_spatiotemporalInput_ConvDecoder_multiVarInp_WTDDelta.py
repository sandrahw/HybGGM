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

#TODO more epochs

random.seed(10)
print(random.random())

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

lon_bounds = (5, 10) #NL bounds(3,6)
lat_bounds = (45, 50)#NL bounds(50,54)

data_length = 12 # number of months in current dataset

inFiles = glob.glob(r'..\data\temp\*.nc')

# modflow files that are saved monthly
params_monthly = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'surface_water_bed_elevation_used',
 'surface_water_elevation', 'input_rch', 'target']

# other modflow files that seem to be static parameters
params_initial = ['bottom_lowermost_layer', 'bottom_uppermost_layer', 
 'drain_conductance', 
 'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
 'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
 'top_uppermost_layer',
 'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer']

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

    if param in params_initial: #repeat the initial values for each month as they are static
        data_arr = np.repeat(data_cut.to_array().values, data_length, axis=0)
        data_arr = np.where(data_arr==np.nan, 0, data_arr)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)
    return data_arr

# load the different modflow files (#TODO: find a way to load all files at once)
def load_cut_data(inFiles, lat_bounds, lon_bounds, data_length):
    abs_lower = data_prep(inFiles[0], lat_bounds, lon_bounds, data_length) #these include negative values
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
    wtd = data_prep(inFiles[17], lat_bounds, lon_bounds, data_length) # wtd has nan values
    top_upper = data_prep(inFiles[18], lat_bounds, lon_bounds, data_length)
    vert_cond_lower = data_prep(inFiles[19], lat_bounds, lon_bounds, data_length) #vert_cond_lower has inf values 
    vert_cond_upper = data_prep(inFiles[20], lat_bounds, lon_bounds, data_length)
    return abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper

abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper = load_cut_data(inFiles, lat_bounds, lon_bounds, data_length)
# plot wtd for one month, indicating range of values
plt.imshow(wtd[0, :, :], cmap='viridis')
plt.colorbar()
plt.title('WTD for one month') #arrays are flipped so plotting might look weird

# calculate the delta wtd for each month
delta_wtd = np.diff(wtd, axis=0) #wtd is always for end of the month so here for example delta wtd between jan and feb means the delta for feb
wtd0 = wtd[0, :, :]
wtd1 = wtd[1, :, :]
dwtd = wtd1 - wtd0 #this is the same as delta_wtd[0, :, :] #so e.g. delta between feb and jan

# define target (y) and input (X) arrays for the CNN
#TODO: double check if datashift makes sense for delta wtd as target and corresponding input
y = delta_wtd[:, np.newaxis, :, :]
X_All = np.stack([abs_lower, abs_upper, 
              bed_cond, 
              bottom_lower, bottom_upper, 
              drain_cond, drain_elev_lower, drain_elev_upper, 
              hor_cond_lower, hor_cond_upper, 
              init_head_lower, init_head_upper, 
              recharge, 
              prim_stor_coeff_lower, prim_stor_coeff_upper, 
              surf_wat_bed_elev, surf_wat_elev, 
              top_upper, 
              vert_cond_lower, vert_cond_upper, #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
              wtd
              ], axis=1)
X = X_All[1:,:,:,:] #remove first month to match the delta wtd data

#check if nan or inf values in X
for i in range(X.shape[1]):
    print(f"Number of NaN values in variable {i}: {np.isnan(X[:, i, :, :]).sum()}")
for i in range(X.shape[1]):   
    print(f"Number of inf values in variable {i}: {np.isinf(X[:, i, :, :]).sum()}")
for i in range(y.shape[1]):
    print(f"Number of NaN values in variable {i}: {np.isnan(y[:, i, :, :]).sum()}")
    print(f"Number of inf values in variable {i}: {np.isinf(y[:, i, :, :]).sum()}")


def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    tensor -= mean 
    tensor /= std
    return   mean, std

inp_var_mean = [] # list to store normalisation information for denormalisation later
inp_var_std = []
X = torch.from_numpy(X).float() #tranform into torch tensor
for i in range(X.shape[1]):
    mean, std = normalize(X[:, i, :, :]) #normalise each variable separately
    inp_var_mean.append(mean)
    inp_var_std.append(std)

out_var_mean = []
out_var_std = []
y = torch.from_numpy(y).float() #transform into torch tensor
for i in range(y.shape[1]):
    mean, std = normalize(y[:, i, :, :])#normalise each variable separately
    out_var_mean.append(mean) # append mean and std of each variable to the list
    out_var_std.append(std)

# Random Temporal Split
timespan = np.arange(0, X.shape[0]) #create a timespan array to include as a feature in the input array
train_months, test_months = train_test_split(timespan, test_size=0.3, random_state=42)

# Extract training and testing data for the temporal split
Xtrain_data_temporal = X[train_months, :, :, :]  
Xtest_data_temporal = X[test_months, :, :, :]    

ytrain_data_temporal = y[train_months, :, :, :]  
ytest_data_temporal = y[test_months, :, :, :] 

#  Random Spatial Split
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

X_train = Xtrain_data_spatial
X_test = Xtest_data_spatial
y_train = ytrain_data_spatial
y_test = ytest_data_spatial

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
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
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
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

#run pretrained model from above on the original data
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

# Denormalize the wtd data
# y_run_denorm = y*out_var_std[0].float() + out_var_mean[0].float() #denormalise the original delta wtd
y_run_og = delta_wtd[:, np.newaxis, :, :]
y_pred_denorm = y_pred_full*out_var_std[0].numpy() + out_var_mean[0].numpy() #denormalise the predicted delta wtd

# reshape and make sure data are arrays
y_pred_denorm = y_pred_denorm[:, 0, :, :]
y_run_og = y_run_og[:, 0, :, :]

#flip data for plotting
y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
y_run_og = np.flip(y_run_og, axis=1)#.numpy() #not necessarily needed to plot?
X_wtd = np.flip(wtd[:, :, :],axis=1) #also get wtd data for plotting
map_tile = xr.open_dataset(r'..\data\temp\target.nc')
map_tile = map_tile['Band1'].isel(time=1)


for i in range(y_pred_denorm.shape[0])[:1]:
    # print(i)
    #X_wtd is the full 12 months, while y_pred_denorm is shifted by 1 month resulting in 11 months, 
    # here calculate the wtd which is the first month of X_wtd + the predicted delta wtd and then 
    # compare to X_wtd[i+1] which is the actual wtd for the same month as delta wtd was predicted
    pred_wtd = X_wtd[i] + y_pred_denorm[i] 

    vmax = max([pred_wtd.max(),X_wtd[i:, :, :].max()])
    vmin = min([pred_wtd.min(),X_wtd[i:, :, :].min()])
    # lim = np.max([np.abs(vmax), np.abs(vmin)])
    
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 4, 1)
    plt.title('Actual WTD (OG) month %s' %(i+2))
    plt.imshow(X_wtd[i+1, :, :], cmap='viridis', vmin=vmin, vmax=vmax) #plot the actual wtd that to compare wtd+delta wtd
    plt.colorbar(shrink=0.8)

    plt.subplot(2, 4, 2)
    plt.title('Predicted WTD')
    plt.imshow(pred_wtd, cmap='viridis', vmin=vmin,vmax=vmax)#,vmin=vmin,vmax=vmax)
    plt.colorbar(shrink=0.8)

    vmax = max([X_wtd[i+1, :, :].max(),pred_wtd.max()])
    vmin = min([X_wtd[i+1, :, :].min(),pred_wtd.min()])
    plt.subplot(2, 4, 3)
    plt.scatter(X_wtd[i+1, :, :].flatten(), pred_wtd.flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('OG WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated WTD')
    plt.title(f"OG vs Sim WTD")
    plt.tight_layout()

    diff = X_wtd[i+1, :, :] - pred_wtd #difference between wtd and calculated wtd
    vmax = diff.max()
    vmin = diff.min()
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.subplot(2, 4, 4)
    plt.title('diff actual vs sim wtd')
    pcm = plt.imshow(diff, cmap='RdBu', vmin=-lim, vmax=lim)#norm=SymLogNorm(linthresh=1))#norm=colors.CenteredNorm()) #difference between wtd and calculated wtd
    plt.colorbar(pcm, orientation='vertical',shrink=0.8)
    plt.tight_layout()

    vmax = y_run_og[i,:,:].max()
    vmin = y_run_og[i,:,:].min()
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.subplot(2, 4, 5)
    pcm = plt.imshow(y_run_og[i,:,:], cmap='RdBu', vmin=-lim, vmax=lim)# norm=SymLogNorm(linthresh=1)) # delta wtd that was target
    plt.colorbar(pcm, orientation='vertical', shrink=0.8)
    plt.title(f"OG delta WTD")
    plt.tight_layout()

    vmax = y_pred_denorm[i,:,:].max()
    vmin = y_pred_denorm[i,:,:].min()
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.subplot(2, 4, 6)
    pcm = plt.imshow(y_pred_denorm[i,:,:], cmap='RdBu', vmin=-lim, vmax=lim)#, norm=SymLogNorm(linthresh=1))#norm=colors.CenteredNorm())# norm=SymLogNorm(linthresh=1)) #delta wtd that was predicted
    plt.colorbar(pcm, orientation='vertical', shrink=0.8)
    plt.title(f"predicted delta WTD")    
    plt.tight_layout()

    vmax = max([y_run_og[i,:,:].max(),y_pred_denorm[i,:,:].max()])
    vmin = min([y_run_og[i,:,:].min(),y_pred_denorm[i,:,:].min()])
    plt.subplot(2, 4, 7)
    plt.scatter(y_run_og[i,:,:], y_pred_denorm[i,:,:],alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('OG delta WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated delta WTD')
    plt.title(f"OG vs Sim delta WTD")

    ax = plt.subplot(2, 4, 8)
    map_tile.plot(ax=ax)
    ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
    plt.tight_layout()
    
    plt.savefig(r'..\data\temp\plots\plot_%s.png' %(i))

        

''' running it on a different region from the same tile'''
lon_bounds = (9, 14) #NL bounds(3,6)
lat_bounds = (50, 55)#NL bounds(50,54)

abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper = load_cut_data(inFiles, lat_bounds, lon_bounds, data_length)

plt.imshow(wtd[0, :, :], cmap='viridis')
plt.colorbar()
plt.title('WTD for one month') #arrays are flipped so plotting might look weird

# calculate the delta wtd for each month
delta_wtd = np.diff(wtd, axis=0)

y = delta_wtd[:, np.newaxis, :, :]
X_All = np.stack([abs_lower, abs_upper, 
              bed_cond, 
              bottom_lower, bottom_upper, 
              drain_cond, drain_elev_lower, drain_elev_upper, 
              hor_cond_lower, hor_cond_upper, 
              init_head_lower, init_head_upper, 
              recharge, 
              prim_stor_coeff_lower, prim_stor_coeff_upper, 
              surf_wat_bed_elev, surf_wat_elev, 
              top_upper, 
              vert_cond_lower, vert_cond_upper, #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
              wtd
              ], axis=1)
X = X_All[1:,:,:,:]
# # replace nan with 0 in X and y
X = np.nan_to_num(X, copy=False, nan=0)
y = np.nan_to_num(y, copy=False, nan=0)

#normalise data
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

#run pretrained model from above on the original data
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

# Denormalize the wtd data
y_run_denorm = y_run*out_var_std[0].float() + out_var_mean[0].float() #denormalise the original delta wtd
y_pred_denorm = y_pred_full*out_var_std[0].numpy() + out_var_mean[0].numpy() #denormalise the predicted delta wtd

# reshape and make sure data are arrays
y_pred_denorm = y_pred_denorm[:, 0, :, :]
y_run_denorm = y_run_denorm[:, 0, :, :].numpy()

#flip data for plotting
y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
y_run_denorm = np.flip(y_run_denorm, axis=1)#.numpy() #not necessarily needed to plot?
X_wtd = np.flip(wtd[:, :, :],axis=1) #also get wtd data for plotting
map_tile = xr.open_dataset(r'..\data\temp\target.nc')
map_tile = map_tile['Band1'].isel(time=1)


for i in range(y_pred_denorm.shape[0])[:]:
    # print(i)
    pred_wtd = X_wtd[i] + y_pred_denorm[i] 
    #X_wtd is the full 12 months, while y_pred_denorm is shifted by 1 month resulting in 11 months, 
    # here calculate the wtd which is the first month of X_wtd + the predicted delta wtd and then 
    # compare to X_wtd[i+1] which is the actual wtd for the same month as delta wtd was predicted

    vmax = max([pred_wtd.max(),X_wtd[:, :, :].max()])
    vmin = min([pred_wtd.min(),X_wtd[:, :, :].min()])
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 4, 1)
    plt.title('Actual WTD (OG) month %s' %(i+2))
    plt.imshow(X_wtd[i+1, :, :], cmap='viridis',vmin=-lim,vmax=lim) #plot the actual wtd that to compare wtd+delta wtd
    plt.colorbar(shrink=0.8)

    plt.subplot(2, 4, 2)
    plt.title('Predicted WTD')
    plt.imshow(pred_wtd, cmap='viridis')#,vmin=vmin,vmax=vmax)
    plt.colorbar(shrink=0.8)

    vmax = max([X_wtd[i, :, :].max(),pred_wtd.max()])
    vmin = min([X_wtd[i, :, :].min(),pred_wtd.min()])
    plt.subplot(2, 4, 3)
    plt.scatter(X_wtd[i+1, :, :].flatten(), pred_wtd.flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('OG WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated WTD')
    plt.title(f"OG vs Sim WTD")
    plt.tight_layout()

    diff = X_wtd[i+1, :, :] - pred_wtd #difference between wtd and calculated wtd
    plt.subplot(2, 4, 4)
    plt.title('diff actual vs sim wtd')
    pcm = plt.imshow(diff, cmap='RdBu', norm=colors.CenteredNorm()) #difference between wtd and calculated wtd
    plt.colorbar(pcm, orientation='vertical',shrink=0.8)
    plt.tight_layout()

    plt.subplot(2, 4, 5)
    pcm = plt.imshow(y_run_denorm[i,:,:], cmap='RdBu', norm=SymLogNorm(linthresh=1)) # delta wtd that was target
    plt.colorbar(pcm, orientation='vertical', shrink=0.8)
    plt.title(f"OG delta WTD")
    plt.tight_layout()

    plt.subplot(2, 4, 6)
    pcm = plt.imshow(y_pred_denorm[i,:,:], cmap='RdBu', norm=colors.CenteredNorm())# norm=SymLogNorm(linthresh=1)) #delta wtd that was predicted
    plt.colorbar(pcm, orientation='vertical', shrink=0.8)
    plt.title(f"predicted delta WTD")    
    plt.tight_layout()

    vmax = max([y_run_denorm[i,:,:].max(),y_pred_denorm[i,:,:].max()])
    vmin = min([y_run_denorm[i,:,:].min(),y_pred_denorm[i,:,:].min()])
    plt.subplot(2, 4, 7)
    plt.scatter(y_run_denorm[i,:,:], y_pred_denorm[i,:,:],alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('OG delta WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated delta WTD')
    plt.title(f"OG vs Sim delta WTD")

    ax = plt.subplot(2, 4, 8)
    map_tile.plot(ax=ax)
    ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
    plt.tight_layout()
    
    plt.savefig(r'..\data\temp\plots\plot_different_region_%s.png' %(i))

        














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