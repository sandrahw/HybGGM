import glob
import netCDF4 as netCDF
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Parameters
data_dir = "input/"
target_dir = "target/"
output_path = "output/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ratio = 0.1
test_ratio = 0.1
hidden_size = 16
num_layers = 2
batch_size = 10
learning_rate = 0.001
epochs = 50

### 7,10 lon
### 47, 50 lat

target_file = glob.glob(f"{target_dir}/*.nc")[0]  # Assume there's only one target file
with netCDF.Dataset(target_file, mode="r") as nc:
    lon = nc["lon"][:]
    lat = nc["lat"][:]

lon_cells = np.arange(0,len(lon))
selected_lon = lon_cells[(lon > 7.) & (lon < 10.)]

xmin = np.where(lon[:] > 7)[0][0]
xmax = np.where(lon[:] < 10)[0][-1] + 1
ymin = np.where(lat[:] > 47)[0][0]
ymax = np.where(lat[:] < 50)[0][-1] + 1

lat = lat[ymin:ymax]
lon = lon[xmin:xmax]

def load_data(data_dir, target_dir):
    # Find all NetCDF files in the data directory
    files = glob.glob(f"{data_dir}/*.nc")
    static_features = []
    dynamic_features = []
    staticVarNames = []
    dynamicVarNames = []
    for file in files:
        print(file)
        with netCDF.Dataset(file, mode="r") as nc:
            # Check dimensions of the dataset
            dims = nc.dimensions.keys()
            vars = list(nc.variables.keys())
            # Identify static and dynamic features
            if "time" not in dims:  # Static feature with 2D dimensions
                data = nc[vars[-1]][ymin:ymax,xmin:xmax]  # Assume the last variable is the feature of interest
                static_features.append(data)
                staticVarNames.append(file)
            elif nc.dimensions["time"].size == 1:  # Single-timestep static
                data = nc[vars[-1]][0, ymin:ymax,xmin:xmax]  # Take the first time step
                static_features.append(data)
                staticVarNames.append(file)
            else:  # Dynamic feature
                data = nc[vars[-1]][:, ymin:ymax,xmin:xmax]  # Assume time, lat, lon ordering
                dynamic_features.append(data)
                dynamicVarNames.append(file)
    file = "predictions_CNN.nc"           
    print(file)
    with netCDF.Dataset(file, mode="r") as nc:
        # Check dimensions of the dataset
        dims = nc.dimensions.keys()
        vars = list(nc.variables.keys())
        # Identify static and dynamic features
        data = nc[vars[-1]][:]  # Assume time, lat, lon ordering
        dynamic_features.append(data)
        dynamicVarNames.append(file)
                
    # Stack static and dynamic features into NumPy arrays
    if static_features:
        static_features = np.repeat(np.stack(static_features, axis=0)[np.newaxis,:], 72, axis=0)  # Shape: [features, lat, lon]
    else:
        raise ValueError("No static features found in the dataset.")
    
    if dynamic_features:
        dynamic_features = np.stack(dynamic_features, axis=1)  # Shape: [time, features, lat, lon]
    else:
        raise ValueError("No dynamic features found in the dataset.")
    
    # Load the target file
    target_file = glob.glob(f"{target_dir}/*.nc")[0]  # Assume there's only one target file
    with netCDF.Dataset(target_file, mode="r") as nc:
        vars = list(nc.variables.keys())
        target = nc[vars[-1]][:,ymin:ymax,xmin:xmax]  # Assume the last variable is the target
    
    return static_features, dynamic_features, target, [staticVarNames, dynamicVarNames]

def normalize_data(input_features, target):
    """
    Normalize static features, dynamic features, and target data.
    
    Static features: feature-wise normalization.
    Dynamic features: time-wise normalization.
    Target: normalized across all time and spatial dimensions.
    """
    # Static features normalization
    # Dynamic features normalization
    time_steps, num_features, lat, lon = input_features.shape
    input_features_reshaped = input_features.transpose(1, 0, 2, 3).reshape(num_features, -1)
    input_scaler = StandardScaler()
    #input_features_normalized = np.zeros_like(input_features_reshaped)
    input_features_normalized = input_scaler.fit_transform(input_features_reshaped.T).T
    input_features_normalized = input_features_normalized.reshape(num_features, time_steps, lat, lon).transpose(1, 0, 2 , 3)

    # Target normalization
    target_reshaped = target.reshape(1,-1)
    target_scaler = StandardScaler()
    target_normalized = target_scaler.fit_transform(target_reshaped.T).T
    target_normalized = target_normalized.reshape(target.shape)
    
    return input_features_normalized, target_normalized, input_scaler, target_scaler


# Normalize the data
static, dynamic, target, varNames = load_data(data_dir, target_dir)
inputs = np.concatenate((static, dynamic), axis=1)
inputs_normalized, target_normalized, inputs_scaler, target_scaler = normalize_data(inputs, target)

print("Features loaded")
# Generate train, test, and validation masks
num_locations = inputs.shape[2] * inputs.shape[3]
location_indices = np.arange(num_locations)
np.random.shuffle(location_indices)

train_size = int(train_ratio * num_locations)
test_size = int(test_ratio * num_locations)

train_indices = location_indices[:train_size]
test_indices = location_indices[train_size:train_size + test_size]
val_indices = location_indices[train_size + test_size:]

def create_masks(indices, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[np.unravel_index(indices, shape)] = True
    return mask

train_mask = create_masks(train_indices, (static.shape[2], static.shape[3]))
test_mask = create_masks(test_indices, (static.shape[2], static.shape[3]))
val_mask = create_masks(val_indices, (static.shape[2], static.shape[3]))

full_mask = np.zeros((inputs.shape[2], inputs.shape[3]), dtype=bool)
full_mask[:] = True

# Dataset class
class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, target, mask):
        self.mask = mask
        self.inputs = inputs[:,:,mask]
        self.target = target[:,mask]

    def __len__(self):
        return self.mask.sum()

    def __getitem__(self, idx):
        inputs_idx = self.inputs.transpose(2,0,1)[idx]
        target_idx = self.target.T[idx]
        return torch.tensor(inputs_idx, dtype=torch.float32), \
               torch.tensor(target_idx, dtype=torch.float32)

train_dataset = TemporalDataset(inputs_normalized, target_normalized, train_mask)
test_dataset = TemporalDataset(inputs_normalized, target_normalized, test_mask)
val_dataset = TemporalDataset(inputs_normalized, target_normalized, val_mask)
all_dataset = TemporalDataset(inputs_normalized, target_normalized, full_mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs = inputs.permute(0, 1, 2)  # [batch_size, seq_len, features]
        lstm_out, _ = self.lstm(inputs)
        out = self.fc(lstm_out)
        return out


# Temporal Gradient Loss
def temporal_gradient_loss(predictions, targets):
    if predictions.dim() == 3:  # Shape: [batch_size, seq_len, features]
        pred_grad = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_grad = targets[:, 1:, :] - targets[:, :-1, :]
    elif predictions.dim() == 4:  # Shape: [batch_size, seq_len, spatial_dim_1, spatial_dim_2]
        pred_grad = predictions[:, 1:, :, :] - predictions[:, :-1, :, :]
        target_grad = targets[:, 1:, :, :] - targets[:, :-1, :, :]
    else:
        raise ValueError("Unexpected tensor shape for predictions and targets.")
    
    return nn.MSELoss()(pred_grad, target_grad)


# Total Loss Function
def total_loss(predictions, targets, lambda_grad=0.1):
    # Reconstruction loss
    reconstruction_loss = nn.MSELoss()(predictions, targets)
    # Temporal gradient loss
    grad_loss = temporal_gradient_loss(predictions, targets)
    # Weighted total loss
    return reconstruction_loss + lambda_grad * grad_loss


# Initialize model
model = LSTM(input_size=inputs.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)

# Training loop
def train_model(model, train_loader, test_loader, epochs):
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lambda_grad = 0.1  # Weight for temporal gradient loss

    for epoch in range(epochs):
        print(epoch)
        model.train()
        train_loss = 0
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            target = target.unsqueeze(-1)  # Add a last dimension if needed, shape: [batch_size, seq_len, 1]

            # Compute total loss
            loss = total_loss(outputs, target, lambda_grad=lambda_grad)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Test the model
        train_loss /= len(train_loader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, target in test_loader:
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                target = target.unsqueeze(-1)

                # Compute total loss
                test_loss += total_loss(outputs, target, lambda_grad=lambda_grad).item()

        test_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}, Test Loss: {test_loss}")

        # Early stopping logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()  # Save the best model state
            patience_counter = 0  # Reset patience counter
            print("Validation loss improved, saving model...")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights
    model.load_state_dict(best_model_state)

train_model(model, train_loader, test_loader, epochs)

def save_predictions_to_netcdf(predictions, output_file, time, lat, lon):
    """
    Save predictions to a NetCDF file.

    Args:
        predictions (numpy.ndarray): Predicted data with shape [time, lat, lon].
        output_file (str): Path to the output NetCDF file.
        time (numpy.ndarray): Array of time steps.
        lat (numpy.ndarray): Array of latitude values.
        lon (numpy.ndarray): Array of longitude values.
    """
    with netCDF.Dataset(output_file, "w", format="NETCDF4") as nc:
        # Create dimensions
        nc.createDimension("time", len(time))
        nc.createDimension("lat", len(lat))
        nc.createDimension("lon", len(lon))
        
        # Create variables
        times = nc.createVariable("time", "f8", ("time",))
        lats = nc.createVariable("lat", "f8", ("lat",))
        lons = nc.createVariable("lon", "f8", ("lon",))
        pred_var = nc.createVariable("predictions", "f8", ("time", "lat", "lon"))
        
        # Add attributes (optional)
        times.units = "months since 2000-01-01 00:00:00"
        times.calendar = "gregorian"
        lats.units = "degrees_north"
        lons.units = "degrees_east"
        pred_var.units = "predicted_units"  # Change to your specific units
        
        # Write data
        times[:] = time
        lats[:] = lat
        lons[:] = lon
        pred_var[:, :, :] = predictions

        print(f"Predictions saved to {output_file}")

# Example Usage
# Assume `predictions` is the predicted data in the shape [time, lat, lon]
# and we have `time`, `lat`, and `lon` arrays.
def predict(model, data_loader, device):
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

# Example Usage

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

# Example usage
# Assuming `val_mask` has shape [lat, lon] and `predictions` has shape [n_samples, time_steps]

validation_predictions = predict(model, val_loader, device)  # Use the predict function

grid_shape = (validation_predictions.shape[1], val_mask.shape[0], val_mask.shape[1])  # (time, lat, lon)
validation_predictions_grid = map_predictions_to_full_grid(
    validation_predictions, val_mask, grid_shape
)


# Step 2: Inverse transform predictions if normalized
validation_predictions_original = target_scaler.inverse_transform(
    validation_predictions_grid.reshape(1,-1))  # Back to original range

predictions = validation_predictions_original.reshape(72, val_mask.shape[0], val_mask.shape[1])

# Step 3: Map predictions to full grid
# Assuming val_mask has shape [lat, lon] and validation_predictions_original has shape [n_samples, time_steps]

# Perform predictions on validation data
time = np.arange(72)  # Replace with your time data

# Save to NetCDF
output_file = "predictions_CNNLSTM_Temporal.nc"
save_predictions_to_netcdf(predictions, output_file, time, lat, lon)
output_file = "target_CNNLSTM_Temporal.nc"
save_predictions_to_netcdf(target, output_file, time, lat, lon)

########### Full region ##############

predictions = predict(model, all_loader, device)  # Use the predict function

grid_shape = (predictions.shape[1], full_mask.shape[0], full_mask.shape[1])  # (time, lat, lon)
all_predictions_grid = map_predictions_to_full_grid(
    predictions, full_mask, grid_shape
)


# Step 2: Inverse transform predictions if normalized
all_predictions_original = target_scaler.inverse_transform(
    all_predictions_grid.reshape(1,-1))  # Back to original range

all_predictions = all_predictions_original.reshape(72, full_mask.shape[0], full_mask.shape[1])

# Step 3: Map predictions to full grid
# Assuming val_mask has shape [lat, lon] and validation_predictions_original has shape [n_samples, time_steps]

# Perform predictions on validation data
time = np.arange(72)  # Replace with your time data

# Save to NetCDF
output_file = "all_predictions_CNNLSTM_Temporal.nc"
save_predictions_to_netcdf(all_predictions, output_file, time, lat, lon)


# output_file = "scalar.nc"
# test = np.ones((1,200*200))
# validation_predictions_original = target_scaler.inverse_transform(test)  # Back to original range

# test2 = validation_predictions_original.reshape(1, val_mask.shape[0], val_mask.shape[1])

# save_predictions_to_netcdf(test2, output_file, [1], lat, lon)


