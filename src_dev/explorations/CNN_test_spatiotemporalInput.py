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

input_data = xr.open_dataset(r'..\data\temp\input_rch.nc')
target_data = xr.open_dataset(r'..\data\temp\target.nc')

test_i = input_data['Band1'].isel(time=1)
test_o = target_data['Band1'].isel(time=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
test_i.plot(ax=ax1)
test_o.plot(ax=ax2)

lon_bounds = (3, 6)
lat_bounds = (50, 54)

input_data = input_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
target_data = target_data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

# Convert xarray DataArrays to numpy arrays
input_array = input_data.to_array().values
target_array = target_data.to_array().values

input_array = np.where(input_array==0, np.nan, input_array)

# Impute NaN values with the mean of the array
def impute_nan(arr):
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 999999
    return arr

input_array = impute_nan(input_array)
target_array = impute_nan(target_array)

X = input_array[0, :, :, :]
y = target_array[0, :, :, :]


X = X[:, np.newaxis, :, :] 
y = y[:, np.newaxis, :, :]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std, mean, std

X_train, X_train_mean, X_train_std = normalize(torch.from_numpy(X_train).float())
X_test, X_test_mean, X_test_std = normalize(torch.from_numpy(X_test).float())
y_train, y_train_mean, y_train_std = normalize(torch.from_numpy(y_train).float())
y_test, y_test_mean, y_test_std = normalize(torch.from_numpy(y_test).float())

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for X, y in train_loader:
    print(f"Input shape: {X.shape}")  # Should be [batch_size, 720, 1800]
    print(f"Target shape: {y.shape}")  # Should be [batch_size, 720, 1800]
    break  # Just check one batch

for X, y in test_loader:
    print(f"Shape of X [B, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel, 32 output channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after conv1 and pooling
        # Input size (1, 1800, 1080) -> Conv1 + Pool (32, 900, 540)
        conv_output_size = 32 * 240 * 180   # Corrected flattened size
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 480 * 360)  # 720 * 1800 = 1296000

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(f"Shape after conv1 and pool: {x.shape}")  # Debugging: print the shape
        x = x.view(x.size(0), -1)  # Flatten the tensor correctly
        # print(f"Shape after flattening: {x.shape}")  # Debugging: print the shape
        x = F.relu(self.fc1(x))
        # print(f"Shape after fc1: {x.shape}")  # Debugging: print the shape
        x = self.fc2(x)
        # print(f"Shape after fc2: {x.shape}")  # Debugging: print the shape
        x = x.view(x.size(0), 1, 480, 360)  # Reshape to the target dimensions
        # print(f"Shape after final reshape: {x.shape}")  # Debugging: print the shape
        return x

# Instantiate the model, define the loss function and the optimizer
model = CNNModel()
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


# Evaluation function with denormalization
def evaluate_model(model, test_loader, criterion, mean, std):
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
    mean = mean.numpy()  # Convert to numpy
    std = std.numpy()  # Convert to numpy
    return (all_outputs * std) + mean  # Denormalize the outputs

# Evaluate the model and denormalize the outputs
test_outputs = evaluate_model(model, test_loader, criterion, y_test_mean, y_test_std)

test_outputs_reshaped = test_outputs[:, 0, :, :]


for i in range(test_outputs_reshaped.shape[0]):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1)
    plt.title('Actual Groundwater Depth')
    plt.imshow(y_test[i, 0, :, :], cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Predicted Groundwater Depth')
    plt.imshow(test_outputs_reshaped[i], cmap='viridis')
    plt.colorbar()
    plt.title(f"Model Output {i+1}")
    plt.show()