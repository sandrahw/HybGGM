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
window_size = 5
hidden_size = window_size * window_size
num_layers = 2
batch_size = 1
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
train_size = int(12)
test_size = int(12)

###### New function for random sampling

def makePatches(inputs):
        time_steps, num_features, x_dim, y_dim = inputs.shape

        # Compute all possible non-overlapping patch indices
        x_indices = np.arange(0, x_dim, window_size)
        y_indices = np.arange(0, y_dim, window_size)

        # Ensure there are enough patches to select from
        all_patches = [(x, y) for x in x_indices for y in y_indices]
        return all_patches

def splitSample(patches, train_ratio, test_ratio):
    np.random.shuffle(patches)
    num_locations = len(patches)
    train_size = int(train_ratio * num_locations)
    test_size = int(test_ratio * num_locations)
    train_indices = patches[:train_size]
    test_indices = patches[train_size:train_size + test_size]
    val_indices = patches[train_size + test_size:]
    return(train_indices, test_indices, val_indices)

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, target, window_size, selected_patches):
        time_steps, num_features, x_dim, y_dim = inputs.shape
        num_samples = len(selected_patches)

        # Extract non-overlapping patches
        inputs_samples = np.zeros((num_samples, time_steps, num_features, window_size, window_size))
        target_samples = np.zeros((num_samples, time_steps, window_size, window_size))

        for i, (x_start, y_start) in enumerate(selected_patches):
            inputs_samples[i] = inputs[:, :, x_start:x_start+window_size, y_start:y_start+window_size]
            target_samples[i] = target[:, x_start:x_start+window_size, y_start:y_start+window_size]

        # Store the processed data
        self.inputs = inputs_samples
        self.target = target_samples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs_idx = self.inputs.transpose(0,1,3,4,2)[idx]
        target_idx = self.target[idx]
        return torch.tensor(inputs_idx, dtype=torch.float32), torch.tensor(target_idx, dtype=torch.float32)

all_patches = makePatches(inputs_normalized)
train_patch, test_patch, val_patch = splitSample(all_patches, 0.1, 0.1)

train_dataset = TemporalDataset(inputs_normalized, target_normalized, window_size, train_patch)
test_dataset = TemporalDataset(inputs_normalized, target_normalized, window_size, test_patch)
val_dataset = TemporalDataset(inputs_normalized, target_normalized, window_size, val_patch)
all_dataset = TemporalDataset(inputs_normalized, target_normalized, window_size, all_patches)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)


# # Dataset class
# class TemporalDataset(torch.utils.data.Dataset):
#     def __init__(self, inputs, target, mask):
#         self.mask = mask
#         self.inputs = inputs[mask,:,:window_size,:window_size]
#         self.target = target[mask,:window_size,:window_size]

#     def __len__(self):
#         return len(self.mask)

#     def __getitem__(self, idx):
#         inputs_idx = self.inputs.transpose(0,2,3,1)[idx]
#         target_idx = self.target.transpose(0,1,2)[idx]
#         return torch.tensor(inputs_idx, dtype=torch.float32), \
#                torch.tensor(target_idx, dtype=torch.float32)

# #TODO work on batch size = 1 and different then timestep

# train_dataset = TemporalDataset(inputs_normalized, target_normalized, range(train_size))
# test_dataset = TemporalDataset(inputs_normalized, target_normalized, range(train_size,train_size+test_size))
# val_dataset = TemporalDataset(inputs_normalized, target_normalized, range(train_size+test_size,72))
# all_dataset = TemporalDataset(inputs_normalized, target_normalized, range(0,72))

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=num_layers, out_channels=window_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=window_size, out_channels=num_layers, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_steps=inputs.size(1)
        #inputs = inputs.view(batch_size, self.time_steps, -1)  # Flatten spatial dims for LSTM
        inputs = inputs.view(batch_size, time_steps, -1)  # [batch_size, seq_len, features]
        h_n = torch.zeros(num_layers, batch_size, hidden_size)
        c_n = torch.zeros(num_layers, batch_size, hidden_size)
        output = torch.zeros(batch_size, time_steps, window_size, window_size)
        for step in range(time_steps):
            #print("Shapes")
            #print(inputs.shape)        
            #print(h_n.shape)        
            #print(c_n.shape)        
            #print(inputs[:,step,:].shape)        
            # Pass through LSTM
            lstm_out, (h_n, c_n) = self.lstm(inputs[:,step,:].unsqueeze(1), (h_n, c_n))  # Use cell state (c_n) as output
            #print("LSTM out shape")
            #print(lstm_out.shape)
            #print("Shape c_n in")
            #print(c_n.shape)
    
            # Reshape LSTM cell state for CNN input
            cnn_input = c_n.squeeze(0).view(batch_size, num_layers, window_size, window_size)
            #print("Shape c_n out")
            #print(cnn_input.shape)
    
            # Pass through CNN
            cnn_out = self.conv1(cnn_input)
            cnn_out = self.relu(cnn_out)
            cnn_out = self.conv2(cnn_out)
            cnn_out = self.sigmoid(cnn_out)
            
            c_n = cnn_out.view(num_layers, batch_size, window_size*window_size)
            #print("Shape c_n out")
            #print(c_n.shape)
            output[:,step,:,:] = lstm_out.view(batch_size, window_size, window_size)

        return output

# Initialize model
model = LSTM(input_size=(window_size*window_size*13), hidden_size=(window_size*window_size), num_layers=num_layers, output_size=(window_size, window_size)).to(device)
train_model(model, train_loader, test_loader, epochs)

# Training loop
def train_model(model, train_loader, test_loader, epochs):
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 5
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(epoch)
        model.train()
        train_loss = 0
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            target = target  # Add a last dimension if needed, shape: [batch_size, seq_len, 1]
            #print("Training")
            #print(inputs.shape)
            #print(outputs.shape)
            #print(target.shape)
            loss = criterion(outputs, target)
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
                target = target #.unsqueeze(-1)
                test_loss += criterion(outputs, target).item()
        test_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Test Loss: {test_loss / len(test_loader)}")

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

#grid_shape = (validation_predictions.shape[1], val_mask.shape[0], val_mask.shape[1])  # (time, lat, lon)
#validation_predictions_grid = map_predictions_to_full_grid(
#    validation_predictions, val_mask, grid_shape
#)


# Step 2: Inverse transform predictions if normalized
validation_predictions_original = target_scaler.inverse_transform(
    validation_predictions.reshape(1,-1))  # Back to original range

predictions = validation_predictions_original
predictions = validation_predictions_original.reshape(-1, window_size, window_size)

# Step 3: Map predictions to full grid
# Assuming val_mask has shape [lat, lon] and validation_predictions_original has shape [n_samples, time_steps]

# Perform predictions on validation data
time = range(48) # Replace with your time data

# Save to NetCDF
new_lat = lat[:window_size]
new_lon = lon[:window_size]
output_file = "predictions_LSTMCNN.nc"
save_predictions_to_netcdf(predictions, output_file, time, new_lat, new_lon)
output_file = "target_LSTMCNN.nc"
new_target = target[range(train_size+test_size,72),:window_size,:window_size]
save_predictions_to_netcdf(new_target, output_file, time, new_lat, new_lon)

########### Full region ##############

#predictions = predict(model, all_loader, device)  # Use the predict function

#grid_shape = (predictions.shape[1], full_mask.shape[0], full_mask.shape[1])  # (time, lat, lon)
#all_predictions_grid = map_predictions_to_full_grid(
#    predictions, full_mask, grid_shape
#)


# Step 2: Inverse transform predictions if normalized
#all_predictions_original = target_scaler.inverse_transform(
#    all_predictions_grid.reshape(1,-1))  # Back to original range

#all_predictions = all_predictions_original.reshape(72, full_mask.shape[0], full_mask.shape[1])

# Step 3: Map predictions to full grid
# Assuming val_mask has shape [lat, lon] and validation_predictions_original has shape [n_samples, time_steps]

# Perform predictions on validation data
#time = np.arange(72)  # Replace with your time data

# Save to NetCDF
#output_file = "all_predictions.nc"
#save_predictions_to_netcdf(all_predictions, output_file, time, lat, lon)


# output_file = "scalar.nc"
# test = np.ones((1,200*200))
# validation_predictions_original = target_scaler.inverse_transform(test)  # Back to original range

# test2 = validation_predictions_original.reshape(1, val_mask.shape[0], val_mask.shape[1])

# save_predictions_to_netcdf(test2, output_file, [1], lat, lon)


############ Plotting ############


def plot_predictions_targets_maps(predictions, target, lon, lat, time_step=0):
    """
    Create a four-panel plot: scatter plot, predictions map, targets map, and error map.
    
    Parameters:
    - predictions: 3D array [time, lat, lon], predicted values.
    - target: 3D array [time, lat, lon], true values.
    - lon: 1D array, longitude values.
    - lat: 1D array, latitude values.
    - time_step: int, the time step to plot for the maps.
    """
    # Flatten data for the scatter plot
    pred_flat = predictions[time_step,:,:].flatten()
    target_flat = target[time_step,:,:].flatten()
    
    # Calculate error for the error map
    error = predictions[time_step] - target[time_step]
    
    # Create the multi-panel figure
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # Panel 1: Scatter plot (Predictions vs. Targets)
    ax[0, 0].scatter(target_flat, pred_flat, alpha=0.5, edgecolor='k', s=20)
    ax[0, 0].plot([target_flat.min(), target_flat.max()], [target_flat.min(), target_flat.max()], 
                  color='red', linestyle='--', label='1:1 Line')
    ax[0, 0].set_title("Predictions vs Targets")
    ax[0, 0].set_xlabel("Target")
    ax[0, 0].set_ylabel("Prediction")
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # Panel 2: Predictions Map
    p1 = ax[1, 1].pcolormesh(lon, lat, predictions[time_step], cmap='viridis', shading='auto')
    fig.colorbar(p1, ax=ax[1, 1], label="Predictions")
    ax[1, 1].set_title(f"Predictions Map (Time Step: {time_step})")
    ax[1, 1].set_xlabel("Longitude")
    ax[1, 1].set_ylabel("Latitude")
    
    # Panel 3: Targets Map
    p2 = ax[1, 0].pcolormesh(lon, lat, target[time_step], cmap='viridis', shading='auto')
    fig.colorbar(p2, ax=ax[1, 0], label="Targets")
    ax[1, 0].set_title(f"Targets Map (Time Step: {time_step})")
    ax[1, 0].set_xlabel("Longitude")
    ax[1, 0].set_ylabel("Latitude")
    
    # Panel 4: Error Map
    p3 = ax[0, 1].pcolormesh(lon, lat, error, cmap='coolwarm', shading='auto')
    fig.colorbar(p3, ax=ax[0, 1], label="Prediction Error")
    ax[0, 1].set_title(f"Prediction Error Map (Time Step: {time_step})")
    ax[0, 1].set_xlabel("Longitude")
    ax[0, 1].set_ylabel("Latitude")
    
    # Show the plot
    plt.suptitle("Model Predictions and Error Analysis", fontsize=16, y=1.02)
    plt.show()

# Example usage
# Assuming predictions, target, lon, and lat are already defined as NumPy arrays
# predictions and target are [time, lat, lon], and lon/lat are 1D arrays.
time_step_to_plot = 4  # Choose a specific time step to visualize
plot_predictions_targets_maps(predictions, new_target, new_lon, new_lat, time_step=time_step_to_plot)

############

def plot_time_series(predictions, target, lon, lat, location_lon, location_lat):
    """
    Create a time series plot for a single location of predictions and target.
    
    Parameters:
    - predictions: 3D array [time, lat, lon], predicted values.
    - target: 3D array [time, lat, lon], true values.
    - lon: 1D array, longitude values.
    - lat: 1D array, latitude values.
    - location_lon: float, longitude of the location to plot.
    - location_lat: float, latitude of the location to plot.
    """
    # Find the closest grid point for the specified location
    lon_idx = np.argmin(np.abs(lon - location_lon))
    lat_idx = np.argmin(np.abs(lat - location_lat))
    
    # Extract time series for the chosen location
    pred_series = predictions[:, lat_idx, lon_idx]
    target_series = target[:, lat_idx, lon_idx]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(pred_series, label="Predictions", marker="o", linestyle="-")
    plt.plot(target_series, label="Target", marker="x", linestyle="--")
    plt.title(f"Time Series at Location (Lon: {lon[lon_idx]}, Lat: {lat[lat_idx]})")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming predictions, target, lon, and lat are already defined as NumPy arrays
# predictions and target are [time, lat, lon], and lon/lat are 1D arrays.
location_lon = 7.02  # Example longitude
location_lat = 47.02  # Example latitude
plot_time_series(predictions, new_target, new_lon, new_lat, location_lon, location_lat)

##############

def calculate_rmse_over_time(predictions, target):
    """
    Calculate RMSE over time between predictions and target.
    
    Parameters:
    - predictions: 3D array [time, lat, lon], predicted values.
    - target: 3D array [time, lat, lon], true values.
    
    Returns:
    - rmse_over_time: 1D array [time], RMSE values for each time step.
    """
    # Flatten lat/lon dimensions and compute RMSE for each time step
    rmse_over_time = np.sqrt(np.mean((predictions - target) ** 2, axis=(1, 2)))
    return rmse_over_time

def plot_rmse_over_time(rmse_over_time):
    """
    Plot RMSE over time.
    
    Parameters:
    - rmse_over_time: 1D array [time], RMSE values for each time step.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_over_time, label="RMSE", marker="o", linestyle="-", color="blue")
    plt.title("RMSE Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Root Mean Squared Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming predictions and target are already defined as NumPy arrays
# predictions and target are [time, lat, lon].
rmse_over_time = calculate_rmse_over_time(predictions, new_target)
plot_rmse_over_time(rmse_over_time)


###############

def compute_temporal_correlation(target, predictions, lat, lon):
    """
    Compute temporal correlation for each spatial location between target and predictions.
    
    Parameters:
    - target: Actual target values (shape: [time_steps, lat, lon])
    - predictions: Model predictions (shape: [time_steps, lat, lon])
    
    Returns:
    - correlation_map: 2D array of shape [lat, lon] with temporal correlations.
    """
    time_steps, lat, lon = target.shape
    correlation_map = np.zeros((lat, lon))
    
    for i in range(lat):
        for j in range(lon):
            target_series = target[:, i, j]
            predictions_series = predictions[:, i, j]
            
            # Compute Pearson correlation
            if np.std(target_series) > 0 and np.std(predictions_series) > 0:
                correlation_map[i, j] = np.corrcoef(target_series, predictions_series)[0, 1]
            else:
                correlation_map[i, j] = np.nan  # Handle cases with no variation
    print(correlation_map.shape)
    return correlation_map


temporal_correlation_map = compute_temporal_correlation(new_target, predictions, new_lat, new_lon)
plt.figure(figsize=(10, 8))
p = plt.pcolormesh(new_lon, new_lat, temporal_correlation_map, cmap='coolwarm', shading='auto')
cbar = plt.colorbar(p, label="Prediction Error")
plt.title(f"Prediction Correlation Map", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

