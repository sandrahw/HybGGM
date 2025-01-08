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



# Convert xarray DataArrays to numpy arrays
input_array = input_data.to_array().values
target_array = target_data.to_array().values

input_array = np.where(input_array==0, np.nan, input_array)
plt.imshow(input_array[0, 1, :, :])
plt.imshow(target_array[0, 1, :, :])

# Impute NaN values with the mean of the array
def impute_nan(arr):
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 999999
    return arr

input_array = impute_nan(input_array)
target_array = impute_nan(target_array)


#just take first slice for now
X = input_array[:, 0:1, :, :]
y = target_array[:, 0:1, :, :]

# Split data into training and testing sets 
train_size = int(0.5 * X.shape[2])


X_train_og, X_test_og = X[:, :, :, :train_size], X[:, :, :, train_size:]  # Adjust spatial split axis
y_train_og, y_test_og = y[:, :, :, :train_size], y[:, :, :, train_size:]

plt.imshow(X_train_og[0, 0, :, :])
plt.imshow(X_test_og[0, 0, :, :])

plt.imshow(y_train_og[0, 0, :, :])
plt.imshow(y_test_og[0, 0, :, :])

# Normalize data
def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()
# Convert data to PyTorch tensors
X_train = normalize(torch.from_numpy(X_train_og).float())
X_test = normalize(torch.from_numpy(X_test_og).float())
y_train = normalize(torch.from_numpy(y_train_og).float())
y_test = normalize(torch.from_numpy(y_test_og).float())


# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for X, y in train_loader:
    print(f"Input shape: {X.shape}")  # Should be [batch_size, 1, 720, 1800]
    print(f"Target shape: {y.shape}")  # Should be [batch_size, 720, 1800]
    break  # Just check one batch

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel, 32 output channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after conv1 and pooling
        # Input size (1, 1800, 1080) -> Conv1 + Pool (32, 900, 540)
        conv_output_size = 32 * 900 * 450   # Corrected flattened size
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 1800 * 900)  # 720 * 1800 = 1296000

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(f"Shape after conv1 and pool: {x.shape}")  # Debugging: print the shape
        x = x.view(x.size(0), -1)  # Flatten the tensor correctly
        print(f"Shape after flattening: {x.shape}")  # Debugging: print the shape
        x = F.relu(self.fc1(x))
        print(f"Shape after fc1: {x.shape}")  # Debugging: print the shape
        x = self.fc2(x)
        print(f"Shape after fc2: {x.shape}")  # Debugging: print the shape
        x = x.view(x.size(0), 1, 1800, 900)  # Reshape to the target dimensions
        print(f"Shape after final reshape: {x.shape}")  # Debugging: print the shape
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


# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            print(f"Training input shape: {inputs.shape}")  # Debugging: print input shape
            print(f"Training target shape: {targets.shape}")  # Debugging: print target shape
            outputs = model(inputs)
            print(f"Model output shape: {outputs.shape}")  # Debugging: print output shape

            # Check shapes before the error line
            print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# Evaluate the model
evaluate_model(model, test_loader, criterion)

# Predict on test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()

def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()
def denormalize(tensor, mean, std):
    return tensor * std + mean

y_test_trans = denormalize(y_test, y_test_og.mean(), y_test_og.std())
y_pred_trans = denormalize(torch.tensor(y_pred), y_test_og.mean(), y_test_og.std())

# Compare predicted vs actual spatial maps
for i in range(len(y_test)):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Actual Groundwater Depth')
    plt.imshow(y_test_trans[0, 0, :, :], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Groundwater Depth')
    plt.imshow(y_pred_trans[0, 0, :, :], cmap='viridis')
    plt.colorbar()
    
    plt.show()


















'''example with using the timestep as train/test split'''
# Reshape data to include the temporal dimension
X = groundwater_recharge.reshape((12, groundwater_recharge.shape[2], groundwater_recharge.shape[3]))
y = groundwater_depth.reshape((12, groundwater_depth.shape[2], groundwater_depth.shape[3]))

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split data into training and testing sets 
train_size = int(0.8 * X.shape[1])
X_train, X_test = X[:, :train_size], X[:, train_size:] #this seems to do spatial splits
y_train, y_test = y[:, :train_size], y[:, train_size:]

train_size = int(0.8 * X.shape[0])
X_train, X_test = X[:train_size, :, :], X[train_size:, :, :] #this seems to do time splits
y_train, y_test = y[:train_size, :, :], y[train_size:, :, :]

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class ConvLSTMModel(nn.Module):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        self.convlstm = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        out, _ = self.convlstm(x)
        out = self.conv1(out)
        return out
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# Instantiate the model, define the loss function and the optimizer
model = ConvLSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluate the model
evaluate_model(model, test_loader, criterion)

# Predict on test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()


# Compare predicted vs actual spatial maps
for i in range(y_test.shape[1]):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Actual Groundwater Depth')
    plt.imshow(y_test[0, i, :, :, 0], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Groundwater Depth')
    plt.imshow(y_pred[0, i, :, :, 0], cmap='viridis')
    plt.colorbar()
    
    plt.show()