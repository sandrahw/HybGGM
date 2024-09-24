#CNN_test
import os
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
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

random.seed(10)
print(random.random())

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

# define the number of months in the dataset and the number of epochs
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
def_epochs = 5

log_directory = r'..\training\logs\%s' %(def_epochs)
log_dir_fig = r'..\training\logs\%s\spatial_eval_plots' %(def_epochs)
#create folder in case not there yet
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)

# define the lat and lon bounds for the test region
# lon_bounds = (5, 10) #CH bounds(5,10)
# lat_bounds = (45, 50)#CH bounds(45,50)

# lon_bounds = (3,6) #NL bounds(3,6)
# lat_bounds = (50,54)#NL bounds(50,54)

lat_bounds = (52, 53)
lon_bounds = (4, 5)
#create mask (for land/ocean)
map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
map_cut = map_tile.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
mask = map_cut.to_array().values
# mask where everything that is nan is 0 and everything else is 1
mask = np.nan_to_num(mask, copy=False, nan=0)
mask = np.where(mask==0, 0, 1)
mask = mask[0, :, :]
plt.imshow(mask[0, :, :])


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
    top_upper = data_prep(inFiles[17], lat_bounds, lon_bounds, data_length)
    vert_cond_lower = data_prep(inFiles[18], lat_bounds, lon_bounds, data_length) #vert_cond_lower has inf values 
    vert_cond_upper = data_prep(inFiles[19], lat_bounds, lon_bounds, data_length)
    wtd = data_prep(inFiles[20], lat_bounds, lon_bounds, data_length) # wtd has nan values
    return abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper

abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper = load_cut_data(inFiles, lat_bounds, lon_bounds, data_length)

# plot wtd for one month, indicating range of values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('WTD for one month')
plt.imshow(wtd[0, :, :], cmap='viridis') #the array version of the input data is flipped
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('Mask')
plt.imshow(mask[0,:,:])


# calculate the delta wtd for each month
delta_wtd = np.diff(wtd, axis=0) #wtd is always for end of the month so here for example delta wtd between jan and feb means the delta for feb

# define target (y) and input (X) arrays for the CNN
y = delta_wtd[:, np.newaxis, :, :] #shape[timesteps, channels, lat, lon]
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
              wtd, 
              mask #TODO check if mask should be used in the input data
              ], axis=1)
X = X_All[1:,:,:,:] #remove first month to match the delta wtd data


# normalising the data for every array and save mean and std for denormalisation
inp_var_mean = [] # list to store normalisation information for denormalisation later
inp_var_std = []
X_norm = []

for i in range(X.shape[1]):
    mean = X[:, i, :, :].mean()
    std = X[:, i, :, :].std()
    # check if every value in array is 0, if so, skip normalisation
    if X[:, i, :, :].max() == 0 and X[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        X_temp = X[:, i, :, :]
    else:
        X_temp = (X[:, i, :, :] - mean) / std
    # print(mean, std, X_temp)
    X_norm.append(X_temp)
    inp_var_mean.append(mean)
    inp_var_std.append(std)
X_norm_arr = np.array(X_norm)
X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3)


out_var_mean = []
out_var_std = []
y_norm = []
for i in range(y.shape[1]):
    mean = y[:, i, :, :].mean()
    std = y[:, i, :, :].std()
    y_temp = (y[:, i, :, :] - mean) / std
    y_norm.append(y_temp)
    out_var_mean.append(mean)
    out_var_std.append(std)
y_norm_arr = np.array(y_norm)
y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)


# def normalize(tensor):
#     mean = tensor.mean()
#     std = tensor.std()
#     tensor -= mean 
#     tensor /= std
#     return   mean, std
# inp_var_mean_new = [] # list to store normalisation information for denormalisation later
# inp_var_std_new = []
# X_norm_new= torch.from_numpy(X).float() #
# for i in range(X.shape[1]):
#     mean, std = normalize(X_norm_new[:, i, :, :]) #normalise each variable separately
#     inp_var_mean_new.append(mean)
#     inp_var_std_new.append(std)

# out_var_mean_new = []
# out_var_std_new = []
# y_norm_new = torch.from_numpy(y).float() #transform into torch tensor
# for i in range(y.shape[1]):
#     mean, std = normalize(y_norm_new[:, i, :, :])#normalise each variable separately
#     out_var_mean_new.append(mean) # append mean and std of each variable to the list
#     out_var_std_new.append(std)


#transform ndarrays of training and test data into torch tensors

# ## Create DataLoader
# train_dataset = TensorDataset(transformArrayToTensor(X_train_all), transformArrayToTensor(y_train_all))
# test_dataset = TensorDataset(transformArrayToTensor(X_test_all), transformArrayToTensor(y_test_all))
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

'''TODO: create selection of training,testing and validation data
 where I use areas of e.g. 50x50 pixels and then select them randomly'''



#This is the old splitting method
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_norm_arr, y_norm_arr, test_size=0.4, random_state=42)
# remove mask from input data train and test
X_train = X_train_all[:, :-1, :, :]
X_test = X_test_all[:, :-1, :, :]
y_train = y_train_all.copy()
y_test = y_test_all.copy()

# mask extracted from train test split
def transformArrayToTensor(array):
    return torch.from_numpy(array).float()


def train_test_mask(data):
    data_mask = data[:, -1, :, :]
    data_mask = data_mask[:, np.newaxis, :, :]
    # data_mask_ex = np.repeat(data_mask, data.shape[1]-1, axis=1)
    data_mask_tensor = transformArrayToTensor(data_mask)
    data_mask_binary = data_mask_tensor.type(torch.ByteTensor)
    data_mask_bool = data_mask_binary.bool()
    return data_mask_bool
X_train_mask = train_test_mask(X_train_all)
X_test_mask = train_test_mask(X_test_all)
y_train_mask = train_test_mask(X_train_all) #does it make sense to use the Xtrain mask as target as well? 
y_test_mask = train_test_mask(X_test_all)


#create new dataset and dataloader for combined loss function incl mask
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data, labels, masks, transform=None):
        """
        Args:
            data (torch.Tensor or numpy array): Input data (e.g., images).
            labels (torch.Tensor or numpy array): Corresponding labels for the input data.
            masks (torch.Tensor or numpy array): Masks corresponding to each input data.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., for data augmentation).
        """
        self.data = data
        self.labels = labels
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input data, label, and mask for the given index
        input_data = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        # Apply any transformations if specified
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, label, mask
# Create dataset instances
train_dataset = CustomDataset(X_train, y_train, X_train_mask)
test_dataset = CustomDataset(X_test, y_test, y_test_mask)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of two Conv2D layers,
    each followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # Conv -> BatchNorm -> ReLU
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a ConvBlock followed by MaxPooling.
    The output of the ConvBlock is stored for the skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)  # Apply the convolutional block
        x_pool = self.pool(x_conv) # Apply max pooling
        return x_conv, x_pool # Return both for the skip connection

# class DecoderBlock(nn.Module):
#     """
#     Decoder block consisting of an upsampling (ConvTranspose2d) and a ConvBlock.
#     It takes the skip connection from the corresponding encoder block.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(out_channels * 2, out_channels)  # Double input channels to account for skip connection
#     def forward(self, x, skip):
#         x = self.up(x)  # Upsample the input
#         # Crop skip connection if the sizes don't match
#         if x.shape != skip.shape:
#             skip = self.center_crop(skip, x.shape[2], x.shape[3])
#         x = torch.cat([x, skip], dim=1)  # Concatenate with the corresponding skip connection
#         x = self.conv_block(x)  # Apply convolutional block
#         return x
#     def center_crop(self, feature_map, target_height, target_width):
#             """
#             Center crops the feature_map to match the target_height and target_width.
#             """
#             _, _, h, w = feature_map.size()
#             delta_h = (h - target_height) // 2
#             delta_w = (w - target_width) // 2
#             return feature_map[:, :, delta_h:(delta_h + target_height), delta_w:(delta_w + target_width)]

class DecoderBlock(nn.Module):
    """
    Decoder block consisting of an upsampling (ConvTranspose2d) and a ConvBlock.
    It takes the skip connection from the corresponding encoder block.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)# Double input channels to account for skip connection

    def forward(self, x, skip):
        x = self.up(x)
        # Center crop the skip connection tensor to match the size of the upsampled tensor
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)  # Concatenate along channel axis
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        
        # Encoder: Downsampling path
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck layer (middle part of the U-Net)
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder: Upsampling path
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        x2, p2 = self.encoder2(p1) # Second block
        x3, p3 = self.encoder3(p2) # Third block
        x4, p4 = self.encoder4(p3) # Fourth block

        # Bottleneck (middle)
        bottleneck = self.bottleneck(p4)

        # Decoder
        d1 = self.decoder1(bottleneck, x4)  # Upsample from bottleneck and add skip connection from encoder4
        d2 = self.decoder2(d1, x3)          # Continue with decoder and corresponding skip connection from encoder3
        d3 = self.decoder3(d2, x2)          # Continue with decoder and corresponding skip connection from encoder2
        d4 = self.decoder4(d3, x1)          # Continue with decoder and corresponding skip connection from encoder1

        # Final output layer
        output = self.final_conv(d4)        # Reduce to the number of output channels (e.g., 1 for groundwater head)

        return output



# class UNet(nn.Module):
#     """
#     The complete U-Net architecture with an encoder-decoder structure.
#     It uses skip connections from the encoder to the decoder.
#     """
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.enc1 = EncoderBlock(21, 64)   # Input channels = 21, Output channels = 64
#         self.enc2 = EncoderBlock(64, 128)
#         # self.enc3 = EncoderBlock(128, 256)
#         # self.enc4 = EncoderBlock(256, 512)
        
#         # self.bottleneck = ConvBlock(512, 1024)  # Bottleneck layer
#         self.bottleneck = ConvBlock(128, 256) 

#         # self.dec4 = DecoderBlock(1024, 512)
#         # self.dec3 = DecoderBlock(512, 256)
#         self.dec2 = DecoderBlock(256, 128)
#         self.dec1 = DecoderBlock(128, 64)

#         self.output_conv = nn.Conv2d(64, 1, kernel_size=1)  # Output layer with 1 channel

#     def forward(self, x):
#         # Encoder path
#         x1, p1 = self.enc1(x)
#         x2, p2 = self.enc2(p1)
#         # x3, p3 = self.enc3(p2)
#         # x4, p4 = self.enc4(p3)

#         # Bottleneck
#         # b = self.bottleneck(p4)
#         b = self.bottleneck(p2)

#         # Decoder path
#         # d4 = self.dec4(b, x4)
#         # d3 = self.dec3(d4, x3)
#         d2 = self.dec2(b, x2)
#         d1 = self.dec1(d2, x1)

#         # Output
#         output = self.output_conv(d1)
#         return output


# Instantiate the model, define the loss function and the optimizer
writer = SummaryWriter(log_dir=log_directory)
model = UNet(input_channels=21, output_channels=1)
torch.save(model.state_dict(), os.path.join(log_directory, 'model_untrained.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, writer=None):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets, masks in train_loader:
            inputs = inputs.float()
            targets = targets.float()  

            optimizer.zero_grad()
            outputs = model(inputs)
  
            # Compute the combined loss
            loss = criterion(outputs[masks], targets[masks])
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        if writer:
           writer.add_scalar('Loss/train_epoch', running_loss/len(train_loader), epoch)
    return 

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=def_epochs, writer=writer)

torch.save(model.state_dict(), os.path.join(log_directory, 'model_trained.pth'))

def evaluate_model(model, test_loader, criterion, writer=None):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)
            loss = criterion(outputs[masks], targets[masks])
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())

        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        # Log the average test loss for this epoch
        if writer:
            writer.add_scalar('Loss/test_epoch', test_loss/len(test_loader))
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_outputs   # Denormalize the outputs

test_outputs = evaluate_model(model, test_loader, criterion, writer=writer)

def plot_tensorboard_logs(log_dir):
    # List all event files in the log directory
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
    print(event_files)
    # Initialize lists to store the data
    train_loss = []
    test_loss = []

    stepstr = []
    stepste = []

    # Iterate through all event files and extract data
    # for event_file in event_files[:]:
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()

    # Extract scalars
    loss_train = event_acc.Scalars('Loss/train_epoch')
    # print('loss train', loss_train)

    # event_acc = EventAccumulator(event_files[1])
    # event_acc.Reload()

    loss_test = event_acc.Scalars('Loss/test_epoch')
    print('loss test', loss_test)

    # Append to the lists
    for i in range(len(loss_train)):
        stepstr.append(loss_train[i].step)
        train_loss.append(loss_train[i].value)
            
    for i in range(len(loss_test)):
        stepste.append(loss_train[i].step)
        test_loss.append(loss_test[i].value)

    # Plot the training and test losses
    fig, ax1 = plt.subplots()
    ax1.plot(stepstr, train_loss, label='Train Loss', color='blue')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Training Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.scatter(stepste, test_loss, label='Test Loss', color='orange')
    ax2.set_ylabel('Test Loss')
    plt.title('Training and Test Loss')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(r'..\training\logs\%s\training_loss.png' %(def_epochs))

plot_tensorboard_logs(log_directory)


'''running the model on original data'''
#if mask has to be cut out from the input data
def transformArrayToTensor(array):
    return torch.from_numpy(array).float()
X_norm_nomask = X_norm_arr[:, :-1, :, :]
mask = map_cut.to_array().values
# mask where everything that is nan is 0 and everything else is 1
mask = np.nan_to_num(mask, copy=False, nan=0)
mask_full = mask[0, :, :]
mask_full = mask_full[:, np.newaxis, :, :]

# dataset = TensorDataset(transformArrayToTensor(X_norm_arr), transformArrayToTensor(y_norm_arr)) 
dataset = CustomDataset(transformArrayToTensor(X_norm_nomask), transformArrayToTensor(y_norm_arr), transformArrayToTensor(mask_full)) 
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# model_reload = SimpleCNN()
model_reload = UNet(input_channels=21, output_channels=1)
model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'model_trained.pth')))
#run pretrained model from above on the original data
def run_model_on_full_data(model, data_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inp, tar, mask in data_loader:
            outputs = model(inp)
            all_outputs.append(outputs.cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs

# Running the model on the entire dataset
y_pred_full = run_model_on_full_data(model_reload, data_loader) #this is now the delta wtd

# Denormalize the wtd data
# y_run_denorm = y*out_var_std[0].float() + out_var_mean[0].float() #denormalise the original delta wtd
y_run_og = delta_wtd[:, np.newaxis, :, :]
y_pred_denorm = y_pred_full*out_var_std[0] + out_var_mean[0] #denormalise the predicted delta wtd

# reshape and make sure data are arrays
y_pred_denorm = y_pred_denorm[:, 0, :, :]
y_run_og = y_run_og[:, 0, :, :]

#flip data for plotting
y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
y_run_og = np.flip(y_run_og, axis=1)#.numpy() #not necessarily needed to plot?
X_wtd = np.flip(wtd[:, :, :],axis=1) #also get wtd data for plotting
map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
map_tile = map_tile['Band1'].isel(time=1)
#TODO: mask ocean/land from wtd
mask_na = np.where(mask==0, np.nan, 1) #
mask_na = np.flip(mask_na, axis=1)
mask_na = mask_na[0,1,:,:]

for i in range(y_pred_denorm.shape[0])[:1]:
    print(i)
    #X_wtd is the full 12 months, while y_pred_denorm is shifted by 1 month resulting in 11 months, 
    # here calculate the wtd which is the first month of X_wtd + the predicted delta wtd and then 
    # compare to X_wtd[i+1] which is the actual wtd for the same month as delta wtd was predicted
    pred_wtd = X_wtd[i] + y_pred_denorm[i] 

    vmax = max([pred_wtd.max(),X_wtd[i:, :, :].max()])
    vmin = min([pred_wtd.min(),X_wtd[i:, :, :].min()])
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 4, 1)
    plt.title('Actual WTD (OG) month %s' %(i+2))
    plt.imshow(X_wtd[i+1, :, :]*mask_na, cmap='viridis', vmin=vmin, vmax=vmax) #plot the actual wtd that to compare wtd+delta wtd
    plt.colorbar(shrink=0.8)
    plt.tight_layout()

    plt.subplot(2, 4, 2)
    plt.title('Predicted WTD')
    plt.imshow(pred_wtd*mask_na, cmap='viridis', vmin=vmin,vmax=vmax)#,vmin=vmin,vmax=vmax)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()

    # vmax = max([X_wtd[i+1, :, :]*.max(),pred_wtd.max()])
    # vmin = min([X_wtd[i+1, :, :].min(),pred_wtd.min()])
    X_wtd_scatter = X_wtd[i+1, :, :]*mask_na
    pred_wtd_scatter = pred_wtd*mask_na
    vmin = min([np.nanmin(X_wtd_scatter),np.nanmin(pred_wtd_scatter)])
    vmax = max([np.nanmax(X_wtd_scatter),np.nanmax(pred_wtd_scatter)])
    plt.subplot(2, 4, 3)
    plt.scatter(X_wtd_scatter.flatten(), pred_wtd_scatter.flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('OG WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated WTD')
    plt.title(f"OG vs Sim WTD")
    plt.tight_layout()

    diff = (X_wtd[i+1, :, :]*mask_na)- (pred_wtd*mask_na) #difference between wtd and calculated wtd
    vmax = np.nanmax(diff)
    vmin = np.nanmin(diff)
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.subplot(2, 4, 4)
    plt.title('diff actual vs sim wtd')
    pcm = plt.imshow(diff, cmap='RdBu',vmin=-lim, vmax=lim)#norm=SymLogNorm(linthresh=1))#norm=colors.CenteredNorm()) #difference between wtd and calculated wtd
    plt.colorbar(pcm, orientation='vertical',shrink=0.8)
    plt.tight_layout()

    vmax = np.nanmax(y_run_og[i,:,:]*mask_na)
    vmin = np.nanmin(y_run_og[i,:,:]*mask_na)
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.subplot(2, 4, 5)
    pcm = plt.imshow(y_run_og[i,:,:]*mask_na, cmap='RdBu',  vmin=-lim, vmax=lim)# norm=SymLogNorm(linthresh=1)) # delta wtd that was target
    plt.colorbar(pcm, orientation='vertical', shrink=0.8)
    plt.title(f"OG delta WTD")
    plt.tight_layout()

    vmax = np.nanmax(y_pred_denorm[i,:,:]*mask_na)
    vmin = np.nanmin(y_pred_denorm[i,:,:]*mask_na)
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.subplot(2, 4, 6)
    pcm = plt.imshow(y_pred_denorm[i,:,:]*mask_na, cmap='RdBu', vmin=-lim, vmax=lim)#, norm=SymLogNorm(linthresh=1))#norm=colors.CenteredNorm())# norm=SymLogNorm(linthresh=1)) #delta wtd that was predicted
    plt.colorbar(pcm, orientation='vertical', shrink=0.8)
    plt.title(f"predicted delta WTD")    
    plt.tight_layout()

    # vmax = max([y_run_og[i,:,:].max(),y_pred_denorm[i,:,:].max()])
    # vmin = min([y_run_og[i,:,:].min(),y_pred_denorm[i,:,:].min()])
    y_run_scatter =y_run_og[i,:,:]*mask_na
    y_pred_scatter = y_pred_denorm[i,:,:]*mask_na
    vmin = min([np.nanmin(y_run_scatter),np.nanmin(y_pred_scatter)])
    vmax = max([np.nanmax(y_run_scatter),np.nanmax(y_pred_scatter)])
    plt.subplot(2, 4, 7)
    plt.scatter(y_run_scatter, y_pred_scatter,alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlabel('OG delta WTD')
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Simulated delta WTD')
    plt.title(f"OG vs Sim delta WTD")
    plt.tight_layout()

    ax = plt.subplot(2, 4, 8)
    map_tile.plot(ax=ax)
    ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
    plt.tight_layout()
    
    plt.savefig(r'%s\plot_timesplit_%s.png' %(log_dir_fig, i))



        








# ''' running it on a different region from the same tile'''
# lon_bounds = (9, 14) #NL bounds(3,6)
# lat_bounds = (50, 55)#NL bounds(50,54)

# abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper = load_cut_data(inFiles, lat_bounds, lon_bounds, data_length)

# plt.imshow(wtd[0, :, :], cmap='viridis')
# plt.colorbar()
# plt.title('WTD for one month') #arrays are flipped so plotting might look weird

# # calculate the delta wtd for each month
# delta_wtd = np.diff(wtd, axis=0)

# y = delta_wtd[:, np.newaxis, :, :]
# X_All = np.stack([abs_lower, abs_upper, 
#               bed_cond, 
#               bottom_lower, bottom_upper, 
#               drain_cond, drain_elev_lower, drain_elev_upper, 
#               hor_cond_lower, hor_cond_upper, 
#               init_head_lower, init_head_upper, 
#               recharge, 
#               prim_stor_coeff_lower, prim_stor_coeff_upper, 
#               surf_wat_bed_elev, surf_wat_elev, 
#               top_upper, 
#               vert_cond_lower, vert_cond_upper, #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
#               wtd
#               ], axis=1)
# X = X_All[1:,:,:,:]
# # # replace nan with 0 in X and y
# X = np.nan_to_num(X, copy=False, nan=0)
# y = np.nan_to_num(y, copy=False, nan=0)

# #normalise data
# inp_var_mean = [] # list to store normalisation information for denormalisation later
# inp_var_std = []
# X_run = torch.from_numpy(X).float() #tranform into torch tensor
# for i in range(X.shape[1]):
#     mean, std = normalize(X_run[:, i, :, :]) #normalise each variable separately
#     inp_var_mean.append(mean)
#     inp_var_std.append(std)

# out_var_mean = []
# out_var_std = []
# y_run = torch.from_numpy(y).float() #transform into torch tensor
# for i in range(y.shape[1]):
#     mean, std = normalize(y_run[:, i, :, :])#normalise each variable separately
#     out_var_mean.append(mean) # append mean and std of each variable to the list
#     out_var_std.append(std)

# dataset = TensorDataset(X_run, y_run)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
# model_reload = SimpleCNN()
# model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'model_trained.pth')))
# #run pretrained model from above on the original data
# def run_model_on_full_data(model, data_loader):
#     model.eval()
#     all_outputs = []
#     with torch.no_grad():
#         for inp, tar in data_loader:
#             outputs = model(inp)
#             all_outputs.append(outputs.cpu().numpy())
#     all_outputs = np.concatenate(all_outputs, axis=0)
#     return all_outputs

# # Running the model on the entire dataset
# y_pred_full = run_model_on_full_data(model, data_loader) #this is now the delta wtd

# # Denormalize the wtd data
# y_run_denorm = y_run*out_var_std[0].float() + out_var_mean[0].float() #denormalise the original delta wtd
# y_pred_denorm = y_pred_full*out_var_std[0].numpy() + out_var_mean[0].numpy() #denormalise the predicted delta wtd

# # reshape and make sure data are arrays
# y_pred_denorm = y_pred_denorm[:, 0, :, :]
# y_run_denorm = y_run_denorm[:, 0, :, :].numpy()

# #flip data for plotting
# y_pred_denorm = np.flip(y_pred_denorm, axis=1) 
# y_run_denorm = np.flip(y_run_denorm, axis=1)#.numpy() #not necessarily needed to plot?
# X_wtd = np.flip(wtd[:, :, :],axis=1) #also get wtd data for plotting
# map_tile = xr.open_dataset(r'..\data\temp\target.nc')
# map_tile = map_tile['Band1'].isel(time=1)


# for i in range(y_pred_denorm.shape[0])[:]:
#     # print(i)
#     pred_wtd = X_wtd[i] + y_pred_denorm[i] 
#     #X_wtd is the full 12 months, while y_pred_denorm is shifted by 1 month resulting in 11 months, 
#     # here calculate the wtd which is the first month of X_wtd + the predicted delta wtd and then 
#     # compare to X_wtd[i+1] which is the actual wtd for the same month as delta wtd was predicted

#     vmax = max([pred_wtd.max(),X_wtd[:, :, :].max()])
#     vmin = min([pred_wtd.min(),X_wtd[:, :, :].min()])
#     lim = np.max([np.abs(vmax), np.abs(vmin)])
    
#     plt.figure(figsize=(20, 8))
#     plt.subplot(2, 4, 1)
#     plt.title('Actual WTD (OG) month %s' %(i+2))
#     plt.imshow(X_wtd[i+1, :, :], cmap='viridis',vmin=-lim,vmax=lim) #plot the actual wtd that to compare wtd+delta wtd
#     plt.colorbar(shrink=0.8)

#     plt.subplot(2, 4, 2)
#     plt.title('Predicted WTD')
#     plt.imshow(pred_wtd, cmap='viridis')#,vmin=vmin,vmax=vmax)
#     plt.colorbar(shrink=0.8)

#     vmax = max([X_wtd[i, :, :].max(),pred_wtd.max()])
#     vmin = min([X_wtd[i, :, :].min(),pred_wtd.min()])
#     plt.subplot(2, 4, 3)
#     plt.scatter(X_wtd[i+1, :, :].flatten(), pred_wtd.flatten(),alpha=0.5, facecolors='none', edgecolors='r')
#     plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
#     plt.xlabel('OG WTD')
#     plt.xlim(vmin,vmax)
#     plt.ylim(vmin,vmax)
#     plt.ylabel('Simulated WTD')
#     plt.title(f"OG vs Sim WTD")
#     plt.tight_layout()

#     diff = X_wtd[i+1, :, :] - pred_wtd #difference between wtd and calculated wtd
#     plt.subplot(2, 4, 4)
#     plt.title('diff actual vs sim wtd')
#     pcm = plt.imshow(diff, cmap='RdBu', norm=colors.CenteredNorm()) #difference between wtd and calculated wtd
#     plt.colorbar(pcm, orientation='vertical',shrink=0.8)
#     plt.tight_layout()

#     plt.subplot(2, 4, 5)
#     pcm = plt.imshow(y_run_denorm[i,:,:], cmap='RdBu', norm=SymLogNorm(linthresh=1)) # delta wtd that was target
#     plt.colorbar(pcm, orientation='vertical', shrink=0.8)
#     plt.title(f"OG delta WTD")
#     plt.tight_layout()

#     plt.subplot(2, 4, 6)
#     pcm = plt.imshow(y_pred_denorm[i,:,:], cmap='RdBu', norm=colors.CenteredNorm())# norm=SymLogNorm(linthresh=1)) #delta wtd that was predicted
#     plt.colorbar(pcm, orientation='vertical', shrink=0.8)
#     plt.title(f"predicted delta WTD")    
#     plt.tight_layout()

#     vmax = max([y_run_denorm[i,:,:].max(),y_pred_denorm[i,:,:].max()])
#     vmin = min([y_run_denorm[i,:,:].min(),y_pred_denorm[i,:,:].min()])
#     plt.subplot(2, 4, 7)
#     plt.scatter(y_run_denorm[i,:,:], y_pred_denorm[i,:,:],alpha=0.5, facecolors='none', edgecolors='r')
#     plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
#     plt.xlabel('OG delta WTD')
#     plt.xlim(vmin,vmax)
#     plt.ylim(vmin,vmax)
#     plt.ylabel('Simulated delta WTD')
#     plt.title(f"OG vs Sim delta WTD")

#     ax = plt.subplot(2, 4, 8)
#     map_tile.plot(ax=ax)
#     ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
#     plt.tight_layout()
    
#     plt.savefig(r'..\data\temp\plots\plot_different_region_%s.png' %(i))

        














# # highlight regions on original map by drawing a rectangle based on the lat and lon bounds
# lon_bounds_old = (5, 10) #example swiss bounds
# lat_bounds_old = (45, 50)#example swiss bounds
# map = xr.open_dataset(r'..\data\temp\target.nc') # in this case, water table depth
# # map = map.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
# map = map['Band1'].isel(time=1)

# fig, ax = plt.subplots()
# map.plot(ax=ax)
# ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
# ax.add_patch(plt.Rectangle((lon_bounds_old[0], lat_bounds_old[0]), lon_bounds_old[1] - lon_bounds_old[0], lat_bounds_old[1] - lat_bounds_old[0], fill=None, color='blue'))
# plt.show()