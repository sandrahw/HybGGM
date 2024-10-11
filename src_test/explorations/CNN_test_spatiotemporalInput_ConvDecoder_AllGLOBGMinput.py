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
import glob

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

# lon_bounds = (6, 8) #NL bounds(3,6)
# lat_bounds = (48, 50)#NL bounds(50,54)

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

#initial conditions
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
        # data_arr = np.nan_to_num(data_arr, copy=False, nan=0)
        # # data_arr = np.where(data_arr==np.nan, 0, data_arr)
        # test = np.isnan(data_arr)
        # if True in test:
        #     print('nan replacement did not work') 
        data_arr = np.where(data_arr==np.inf, 0, data_arr)

    if param in params_initial:
        data_arr = np.repeat(data_cut.to_array().values, data_length, axis=0)
        # data_arr = np.where(data_arr==np.nan, 0, data_arr)
        # data_arr = np.nan_to_num(data_arr, copy=False, nan=0)
        # test = np.isnan(data_arr)
        # if True in test:
        #     print('nan replacement did not work')
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


y = wtd[1:, np.newaxis, :, :]
X = np.stack([#abs_lower[1:,:,:], abs_upper[1:,:,:], 
              #bed_cond[1:,:,:], 
              #bottom_lower[1:,:,:], bottom_upper[1:,:,:], 
              #drain_cond[1:,:,:], drain_elev_lower[1:,:,:], drain_elev_upper[1:,:,:], 
              #hor_cond_lower[1:,:,:], hor_cond_upper[1:,:,:], 
              #init_head_lower[1:,:,:], init_head_upper[1:,:,:], 
              recharge[1:,:,:], 
              #prim_stor_coeff_lower[1:,:,:], prim_stor_coeff_upper[1:,:,:], 
              #surf_wat_bed_elev[1:,:,:], surf_wat_elev[1:,:,:], 
              top_upper[1:,:,:], 
              #vert_cond_lower[1:,:,:], vert_cond_upper[1:,:,:], #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
              wtd[:-1, :, :]
              ], axis=1)




#check if nan values in X
for i in range(X.shape[1]):
    print(f"Number of NaN values in variable {i}: {np.isnan(X[:, i, :, :]).sum()}")
for i in range(X.shape[1]):   
    print(f"Number of inf values in variable {i}: {np.isinf(X[:, i, :, :]).sum()}")
# for i in range(y.shape[1]):
#     # print(f"Number of NaN values in variable {i}: {np.isnan(y[:, i, :, :]).sum()}")
#     print(f"Number of inf values in variable {i}: {np.isinf(y[:, i, :, :]).sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
X_train = torch.from_numpy(X_train).float() #tranform into torch tensor
X_test = torch.from_numpy(X_test).float()
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
y_train = torch.from_numpy(y_train).float() #transform into torch tensor
y_test = torch.from_numpy(y_test).float()
for i in range(y_train.shape[1]):
    mean, std = normalize(y_train[:, i, :, :])#normalise each variable separately
    out_var_mean_train.append(mean) # append mean and std of each variable to the list
    out_var_std_train.append(std)

    mean, std = normalize(y_test[:, i, :, :])
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
        self.conv1 = nn.Conv2d(20, 16, kernel_size=3, padding=1)  # Input: 3 channels (recharge, levels, topography)
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
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs   # Denormalize the outputs

test_outputs = evaluate_model(model, test_loader, criterion)
test_outputs_reshaped = test_outputs[:, 0, :, :]

y_test_denorm = y_test*out_var_std_train[0].float() + out_var_mean_train[0].float() #denormalise the test outputs
y_pred_denorm = test_outputs*out_var_std_train[0].numpy() + out_var_mean_train[0].numpy() #denormalise the predicted outputs
y_pred_denorm = y_pred_denorm[:, 0, :, :]

y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
y_test_denorm = y_test_denorm.flip(2)

from matplotlib.colors import SymLogNorm
for i in range(y_pred_denorm.shape[0]):
    vmax = max([y_test_denorm.max(),y_pred_denorm.max()])
    vmin = min([y_test_denorm.min(),y_pred_denorm.min()])
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title('Actual Groundwater Depth')
    plt.imshow(y_test_denorm[i, 0, :, :], cmap='viridis',vmin=vmin,vmax=vmax)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Predicted Groundwater Depth')
    plt.imshow(y_pred_denorm[i], cmap='viridis',vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.title(f"Model Output {i+1}")

    diff = y_test_denorm[i, 0, :, :] - y_pred_denorm[i]
    plt.subplot(2, 2, 3)
    plt.title('Predicted Groundwater Depth')
    plt.imshow(diff, cmap='coolwarm_r', norm=SymLogNorm(linthresh=1))
    plt.colorbar()
    plt.title(f"Difference OG{i} - Pred{i+1}")
    plt.tight_layout()

    #create actuall diff between actual groundwater depth the timestep before and current groundwater depth
    diff_act = y_test_denorm[i, 0, :, :] - y_test_denorm[i-1, 0, :, :]  
    plt.subplot(2, 2, 4)
    plt.title('Actual Groundwater Depth')
    plt.imshow(diff_act, cmap='coolwarm_r', norm=SymLogNorm(linthresh=1))
    plt.colorbar()
    plt.title(f"Difference OG{i} - OG{i+1}")    
    plt.tight_layout()



