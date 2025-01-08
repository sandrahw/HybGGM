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

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

input_data = xr.open_dataset(r'..\data\temp\input_rch.nc') # in this case, groundwater recharge
target_data = xr.open_dataset(r'..\data\temp\target.nc') # in this case, water table depth
input_data_top = xr.open_dataset(r'..\data\temp\top_uppermost_layer.nc') # upper most layer, topography?

# test_i = input_data['Band1'].isel(time=1)
# test_o = target_data['Band1'].isel(time=1)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# test_i.plot(ax=ax1)
# test_o.plot(ax=ax2)

lon_bounds = (6, 8) #NL bounds(3,6)
lat_bounds = (48, 50)#NL bounds(50,54)

input_data = input_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
target_data = target_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
input_data_top = input_data_top.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds)) 

# Convert xarray DataArrays to numpy arrays
input_array = input_data.to_array().values
#reshape input2_array to match input_array by repeating the same array for every timestep in input_array
input2_array = np.repeat(input_data_top.to_array().values, input_array.shape[1], axis=0)
input3_array = target_data.to_array().values
target_array = target_data.to_array().values


X1 = input_array[0, 1:, :, :] # recharge from 2nd month till last (because we want to include lagged values from wtd)
X2 = input2_array [1: , : , :]# topography
X3 = input3_array[0, :-1, :, :]# wtd from 1st month till second last
y = target_array[0, 1:, :, :] # wtd from 2nd month till last

X = np.stack([X1, X2, X3], axis=1)
y = y[:, np.newaxis, :, :]

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            print(f"Training input shape: {inputs.shape}")  # Debugging: print input shape
            print(f"Training target shape: {targets.shape}")  # Debugging: print target shape

            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Model output shape: {outputs.shape}")  # Debugging: print output shape

            # Check shapes before the error line
            print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")

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
    plt.imshow(diff, cmap='coolwarm', norm=SymLogNorm(linthresh=1))
    plt.colorbar()
    plt.title(f"Difference OG-Pred {i+1}")
    plt.tight_layout()



