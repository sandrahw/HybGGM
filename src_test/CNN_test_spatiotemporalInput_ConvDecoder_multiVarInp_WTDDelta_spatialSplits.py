#CNN_test
import os
import gc
import tracemalloc
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
from torch.utils.data import random_split, DataLoader, TensorDataset
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

''' Set random seed for reproducibility also for different torch applications'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False     # Turn off optimization that introduces randomness

# Set seed before any training or data loading happens
set_seed(10)

'''define general path, data length, case area, number of epochs, learning rate and batch size'''
general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'
# define the number of months in the dataset and the number of epochs
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
def_epochs = 5
lr_rate = 0.001
batchSize = 1
lat_bounds = (45, 52.5) #roughly half the tile
lon_bounds = (0, 15)#roughly half the tile

'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs\%s_%s_%s' %(def_epochs, lr_rate ,batchSize)
log_dir_fig = r'..\training\logs\%s_%s_%s\spatial_eval_plots' %(def_epochs, lr_rate ,batchSize)
#create folder in case not there yet
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)



'''create mask (for land/ocean)'''
map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
map_cut = map_tile.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
mask = map_cut.to_array().values
# mask where everything that is nan is 0 and everything else is 1
mask = np.nan_to_num(mask, copy=False, nan=0)
mask = np.where(mask==0, 0, 1)
mask = mask[0, :, :]
plt.imshow(mask[0, :, :])
np.save(r'..\data\input_target_example\mask.npy', mask)


'''load the modflow files and prepare the data for input'''
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

''''calculate the delta wtd for each month - define target (y) and input (X) arrays for the CNN'''
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
np.save(r'..\data\input_target_example\X.npy', X)
np.save(r'..\data\input_target_example\y.npy', y)

# release memory by deleting large lists
del abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper
del delta_wtd
del map_tile, map_cut
del X_All
gc.collect()


'''normalising the data for every array and save mean and std for denormalisation'''
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
np.save(r'..\data\input_target_example\X_norm_arr.npy', X_norm_arr)
np.save(r'..\data\input_target_example\inp_var_mean.npy', inp_var_mean)
np.save(r'..\data\input_target_example\inp_var_std.npy', inp_var_std)

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

np.save(r'..\data\input_target_example\y_norm_arr.npy', y_norm_arr)
np.save(r'..\data\input_target_example\out_var_mean.npy', out_var_mean)
np.save(r'..\data\input_target_example\out_var_std.npy', out_var_std)

#different approach to normalise the data
# # Normalize data
# def normalize(tensor):
#     mean = tensor.mean()
#     std = tensor.std()
#     return (tensor - mean) / std, mean, std

# X_norm = X.copy()
# y_norm = y.copy()
# X_norm_arr_test, X_norm_mean, X_norm_std = normalize(torch.from_numpy(X_norm).float())
# y_norm_arr_test, y_norm_mean, y_norm_std = normalize(torch.from_numpy(y_norm).float())


del X, y 
del X_norm, y_norm
gc.collect()


'''split the data into patches for spatial evaluation'''
def split_spatial_data(data, lat_chunk_size, lon_chunk_size):
    """
    Split the spatial dimensions (lat, lon) of the data into smaller chunks.
    
    Args:
    - data: Input array of shape (months, variables, lat, lon), e.g., (71, 22, 900, 1800)
    - lat_chunk_size: Desired chunk size for the latitude dimension.
    - lon_chunk_size: Desired chunk size for the longitude dimension.

    Returns:
    - patches: List of smaller chunks with shape (months, variables, lat_chunk_size, lon_chunk_size).
    """
    months, variables, lat, lon = data.shape
    patches = []
    
    for i in range(0, lat, lat_chunk_size):
        for j in range(0, lon, lon_chunk_size):
            # Extract a patch (ensure it's within bounds)
            patch = data[:, :, i:i+lat_chunk_size, j:j+lon_chunk_size]
            patches.append(patch)
    
    return np.array(patches)

# Split normalized data into patches
input_patches = split_spatial_data(X_norm_arr, 100, 100)
target_patches = split_spatial_data(y_norm_arr, 100, 100)
print(f"Number of patches: {len(input_patches)}, Patch shape: {input_patches[0].shape}")
print(f"Number of patches: {len(target_patches)}, Patch shape: {target_patches[0].shape}")

np.save(r'..\data\input_target_example\input_patches.npy', input_patches)
np.save(r'..\data\input_target_example\target_patches.npy', target_patches)

del X_norm_arr, y_norm_arr
'''reload the data for the CNN and only select half of the patches for use in the CNN'''
input_patches = np.load(r'..\data\input_target_example\input_patches.npy')
target_patches = np.load(r'..\data\input_target_example\target_patches.npy')

# select half of the batches randomly and keep the other half for another case
input_half_patches, rest_half_patches = train_test_split(input_patches, test_size=0.8, random_state=10)
target_half_patches, rest_half_patches = train_test_split(target_patches, test_size=0.8, random_state=10)


'''split the patches into training, validation and test sets'''
train_patches, val_test_patches = train_test_split(input_half_patches, test_size=0.7, random_state=10)
val_patches, test_patches = train_test_split(val_test_patches, test_size=0.6, random_state=10)  # 20% test, 20% validation

y_train_patches, y_val_test_patches = train_test_split(target_half_patches, test_size=0.7, random_state=10)
y_val_patches, y_test_patches = train_test_split(y_val_test_patches, test_size=0.6, random_state=10)  # 20% test, 20% validation
# np.save(r'..\data\input_target_example\train_patches_half.npy', train_patches)
# np.save(r'..\data\input_target_example\val_patches_half.npy', val_patches)
# np.save(r'..\data\input_target_example\test_patches_half.npy', test_patches)

# np.save(r'..\data\input_target_example\y_train_patches_half.npy', y_train_patches)
# np.save(r'..\data\input_target_example\y_val_patches_half.npy', y_val_patches)
# np.save(r'..\data\input_target_example\y_test_patches_half.npy', y_test_patches)



del input_patches, target_patches
del input_half_patches, target_half_patches

# '''reload the data for the CNN'''
# y_train_patches = np.load(r'..\data\input_target_example\y_train_patches.npy')
# y_val_patches = np.load(r'..\data\input_target_example\y_val_patches.npy')
# y_test_patches = np.load(r'..\data\input_target_example\y_test_patches.npy')

# train_patches = np.load(r'..\data\input_target_example\train_patches.npy')
# val_patches = np.load(r'..\data\input_target_example\val_patches.npy')
# test_patches = np.load(r'..\data\input_target_example\test_patches.npy')

'''remove mask in every training, validation and test patch'''
def remove_mask_patch(r):
    r_train = r[:, :, :-1, :, :]
    r_mask = r[:, :, -1, :, :]
    r_mask = np.where(r_mask<=0, 0, 1)
    mask_bool = r_mask.astype(bool)
    mask_bool = mask_bool[:, :, np.newaxis, :, :]
    return r_train, mask_bool

X_train, mask_train = remove_mask_patch(train_patches)
X_val, mask_val = remove_mask_patch(val_patches)
X_test, mask_test = remove_mask_patch(test_patches)

'''transform the the data into cnn input format'''
X_train_reshaped = X_train.reshape(-1, X_train.shape[2], X_train.shape[3], X_train.shape[4])
X_val_reshaped = X_val.reshape(-1, X_val.shape[2], X_val.shape[3], X_val.shape[4])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2], X_test.shape[3], X_test.shape[4])

y_train_reshaped = y_train_patches.reshape(-1, y_train_patches.shape[2], y_train_patches.shape[3], y_train_patches.shape[4])
y_val_reshaped = y_val_patches.reshape(-1, y_val_patches.shape[2], y_val_patches.shape[3], y_val_patches.shape[4])
y_test_reshaped = y_test_patches.reshape(-1, y_test_patches.shape[2], y_test_patches.shape[3], y_test_patches.shape[4])

mask_train_reshaped = mask_train.reshape(-1, mask_train.shape[2], mask_train.shape[3], mask_train.shape[4])
mask_val_reshaped = mask_val.reshape(-1, mask_val.shape[2], mask_val.shape[3], mask_val.shape[4])
mask_test_reshaped = mask_test.reshape(-1, mask_test.shape[2], mask_test.shape[3], mask_test.shape[4])

# # Split the dataset into train, validation, and test sets -> temporal split
# train_size = int(0.3 * len(X_norm_arr))  # 30% for training
# val_size = int(0.3 * len(X_norm_arr))   # 30% for validation
# test_size = len(X_norm_arr) - train_size - val_size  # 40% for testing

# Split the data into training, validation, and test sets based on the calculated sizes
# X_train = X_norm_arr[:train_size]
# X_val = X_norm_arr[train_size:train_size+val_size]
# X_test = X_norm_arr[train_size+val_size:]
# y_train = y_norm_arr[:train_size]
# y_val = y_norm_arr[train_size:train_size+val_size]
# y_test = y_norm_arr[train_size+val_size:]

# take out the mask from the input data
# def remove_mask(r):
#     r_train = r[:, :-1, :, :]
#     r_mask = r[:, -1, :, :]
#     # redefine mask with 0 and 1 values
#     r_mask = np.where(r_mask<=0, 0, 1)
#     # create boolean mask based on mask values
#     mask_bool = r_mask.astype(bool)
#     mask_bool = mask_bool[:, np.newaxis, :, :]
#     return r_train, mask_bool
    

# X_train, mask_train = remove_mask(X_train)
# X_val, mask_val = remove_mask(X_val)
# X_test, mask_test = remove_mask(X_test)

'''transform the data into tensors'''
def transformArrayToTensor(array):
    return torch.from_numpy(array).float()
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
train_loader = DataLoader(CustomDataset(X_train_reshaped, y_train_reshaped, mask_train_reshaped), batch_size=batchSize, shuffle=False)
validation_loader = DataLoader(CustomDataset(X_val_reshaped, y_val_reshaped, mask_val_reshaped), batch_size=batchSize, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test_reshaped, y_test_reshaped, mask_test_reshaped), batch_size=batchSize, shuffle=False)

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
        # self.encoder2 = EncoderBlock(64, 128)
        # self.encoder3 = EncoderBlock(128, 256)
        # self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck layer (middle part of the U-Net)
        # self.bottleneck = ConvBlock(512, 1024)
        # self.bottleneck = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(64, 128)

        # Decoder: Upsampling path
        # self.decoder1 = DecoderBlock(1024, 512)
        # self.decoder2 = DecoderBlock(512, 256)
        # self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        # x2, p2 = self.encoder2(p1) # Second block
        # x3, p3 = self.encoder3(p2) # Third block
        # x4, p4 = self.encoder4(p3) # Fourth block

        # Bottleneck (middle)
        # bottleneck = self.bottleneck(p4)
        # bottleneck = self.bottleneck(p2)
        bottleneck = self.bottleneck(p1)

        # Decoder
        # d1 = self.decoder1(bottleneck, x4)  # Upsample from bottleneck and add skip connection from encoder4
        # d2 = self.decoder2(d1, x3)          # Continue with decoder and corresponding skip connection from encoder3
        # d3 = self.decoder3(d2, x2)  
        # d3 = self.decoder3(bottleneck, x2)        # Continue with decoder and corresponding skip connection from encoder2
        # d4 = self.decoder4(d3, x1)          # Continue with decoder and corresponding skip connection from encoder1

        d4 = self.decoder4(bottleneck, x1)          # Continue with decoder and corresponding skip connection from encoder1

        # Final output layer
        output = self.final_conv(d4)        # Reduce to the number of output channels (e.g., 1 for groundwater head)

        return output

# Instantiate the model, define the loss function and the optimizer
writer = SummaryWriter(log_dir=log_directory)

model = UNet(input_channels=21, output_channels=1)
torch.save(model.state_dict(), os.path.join(log_directory, 'model_untrained.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

# RMSE function
def rmse(outputs, targets, masks):
    return torch.sqrt(F.mse_loss(outputs[masks], targets[masks]))

# # MAE function
# def mae(outputs, targets, masks):
#     return F.l1_loss(outputs[masks], targets[masks])

# Training function
def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, writer=None):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0
        running_train_rmse = 0.0
        running_train_mae = 0.0

        for inputs, targets, masks in train_loader:
            inputs = inputs.float()
            targets = targets.float()

            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = criterion(outputs[masks], targets[masks])
            loss = rmse(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Calculate RMSE and MAE
            # running_train_rmse += rmse(outputs, targets, masks).item()
            # running_train_mae += mae(outputs, targets, masks).item()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        # epoch_train_rmse = running_train_rmse / len(train_loader.dataset)
        # epoch_train_mae = running_train_mae / len(train_loader.dataset)

        if writer:
            writer.add_scalar('Loss/train_epoch', running_train_loss / len(train_loader), epoch)
            # writer.add_scalar('RMSE/train_epoch', epoch_train_rmse, epoch)
            # writer.add_scalar('MAE/train_epoch', epoch_train_mae, epoch)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_rmse = 0.0
        running_val_mae = 0.0

        with torch.no_grad():
            for inputs, targets, masks in validation_loader:
                inputs = inputs.float()
                targets = targets.float()

                outputs = model(inputs)
                # loss = criterion(outputs[masks], targets[masks])
                loss = rmse(outputs, targets, masks)
                running_val_loss += loss.item() * inputs.size(0)

                # Calculate RMSE and MAE
                # running_val_rmse += rmse(outputs, targets, masks).item()
                # running_val_mae += mae(outputs, targets, masks).item()
        
        epoch_val_loss = running_val_loss / len(validation_loader.dataset)
        # epoch_val_rmse = running_val_rmse / len(validation_loader.dataset)
        # epoch_val_mae = running_val_mae / len(validation_loader.dataset)

        if writer:
            writer.add_scalar('Loss/validation_epoch', running_val_loss / len(validation_loader), epoch)
            # writer.add_scalar('RMSE/validation_epoch', epoch_val_rmse, epoch)
            # writer.add_scalar('MAE/validation_epoch', epoch_val_mae, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
            #   f"Training RMSE: {epoch_train_rmse:.4f}, Validation RMSE: {epoch_val_rmse:.4f}, "
            #   f"Training MAE: {epoch_train_mae:.4f}, Validation MAE: {epoch_val_mae:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(log_directory, 'best_model.pth'))
            print("Best model saved!")

    return

def test_model(model, test_loader, criterion, writer=None):
    model.eval()
    running_test_loss = 0.0
    running_test_rmse = 0.0
    running_test_mae = 0.0
    all_outputs = []

    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)

            # Compute the loss
            # loss = criterion(outputs[masks], targets[masks])
            loss = rmse(outputs, targets, masks)
            running_test_loss += loss.item() * inputs.size(0)

            # Compute RMSE and MAE for the current batch
            # running_test_rmse += rmse(outputs, targets, masks).item() * inputs.size(0)
            # running_test_mae += mae(outputs, targets, masks).item() * inputs.size(0)

            # Store outputs for further analysis or visualization if needed
            all_outputs.append(outputs.cpu().numpy())

        # Compute the average loss, RMSE, and MAE for the entire test dataset
        test_loss = running_test_loss / len(test_loader.dataset)
        # test_rmse = running_test_rmse / len(test_loader.dataset)
        # test_mae = running_test_mae / len(test_loader.dataset)

        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/test_epoch', test_loss)
            # writer.add_scalar('RMSE/test_epoch', test_rmse)
            # writer.add_scalar('MAE/test_epoch', test_mae)

        # Combine all output batches into a single array
        all_outputs = np.concatenate(all_outputs, axis=0)

        # Print test results
        print(f"Test Loss: {test_loss:.4f}")

    return all_outputs
# Train the model
train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=def_epochs, writer=writer)

test_outputs = test_model(model, test_loader, criterion, writer=writer)

def plot_tensorboard_logs(log_dir):
    # List all event files in the log directory
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
    # print(event_files)
    # Initialize lists to store the data
    train_loss = []
    val_loss = []
    test_loss = []
    # train_rmse = []
    # val_rmse = []
    # test_rmse = []
    # train_mae = []
    # val_mae = []
    # test_mae = []
    
    stepstr = []
    stepsva = []
    stepste = []
    # stepstrrmse = []
    # stepsvrmse = []
    # stepstermse = []
    # stepstrmae = []
    # stepsvmae = []
    # stepstemae = []

    # Iterate through all event files and extract data
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()

    # Extract scalars
    loss_train = event_acc.Scalars('Loss/train_epoch')
    loss_val = event_acc.Scalars('Loss/validation_epoch')
    loss_test = event_acc.Scalars('Loss/test_epoch')
    # rmse_train = event_acc.Scalars('RMSE/train_epoch')
    # rmse_val = event_acc.Scalars('RMSE/validation_epoch')
    # rmse_test = event_acc.Scalars('RMSE/test_epoch')
    # mae_train = event_acc.Scalars('MAE/train_epoch')
    # mae_val = event_acc.Scalars('MAE/validation_epoch')
    # mae_test = event_acc.Scalars('MAE/test_epoch')


    # Append to the lists
    for i in range(len(loss_train)):
        stepstr.append(loss_train[i].step)
        train_loss.append(loss_train[i].value)
    
    for i in range(len(loss_val)):
        stepsva.append(loss_val[i].step)
        val_loss.append(loss_val[i].value)
            
    for i in range(len(loss_test)):
        stepste.append(loss_test[i].step)
        test_loss.append(loss_test[i].value)
    
    # for i in range(len(rmse_train)):
    #     stepstrrmse.append(rmse_train[i].step)
    #     train_rmse.append(rmse_train[i].value)
    
    # for i in range(len(rmse_val)):
    #     stepsvrmse.append(rmse_val[i].step)
    #     val_rmse.append(rmse_val[i].value)
    
    # for i in range(len(rmse_test)):
    #     stepstermse.append(rmse_test[i].step)
    #     test_rmse.append(rmse_test[i].value)
    
    # for i in range(len(mae_train)):
    #     stepstrmae.append(mae_train[i].step)
    #     train_mae.append(mae_train[i].value)
    
    # for i in range(len(mae_val)):
    #     stepsvmae.append(mae_val[i].step)
    #     val_mae.append(mae_val[i].value)
    
    # for i in range(len(mae_test)):
    #     stepstemae.append(mae_test[i].step)
    #     test_mae.append(mae_test[i].value)

    # Plot the training and test losses
    fig, ax1 = plt.subplots()
    ax1.plot(stepstr, train_loss, label='Train Loss', color='blue')
    ax1.plot(stepsva, val_loss, label='Validation Loss', color='green')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Training/Validation Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.scatter(stepste, test_loss, label='Test Loss', color='orange')
    ax2.set_ylabel('Test Loss')
    plt.title('Training and Test Loss')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(r'..\training\logs\%s_%s_%s\training_loss.png' %(def_epochs, lr_rate, batchSize))

    # # Plot the training and test RMSE
    # fig, ax1 = plt.subplots()
    # ax1.plot(stepstrrmse, train_rmse, label='Train RMSE', color='blue')
    # ax1.plot(stepsvrmse, val_rmse, label='Validation RMSE', color='green')
    # ax1.set_xlabel('Steps')
    # ax1.set_ylabel('Training RMSE')
    # ax1.legend(loc='upper left')
    # ax2 = ax1.twinx()
    # ax2.scatter(stepstermse, test_rmse, label='Test RMSE', color='orange')
    # ax2.set_ylabel('Test RMSE')
    # plt.title('Training and Test RMSE')
    # ax2.legend(loc='upper right')   
    # plt.tight_layout()
    # plt.savefig(r'..\training\logs\%s\training_rmse.png' %(def_epochs))

    # # Plot the training and test MAE
    # fig, ax1 = plt.subplots()
    # ax1.plot(stepstrmae, train_mae, label='Train MAE', color='blue')
    # ax1.plot(stepsvmae, val_mae, label='Validation MAE', color='green') 
    # ax1.set_xlabel('Steps') 
    # ax1.set_ylabel('Training MAE')
    # ax1.legend(loc='upper left')
    # ax2 = ax1.twinx()
    # ax2.scatter(stepstemae, test_mae, label='Test MAE', color='orange')
    # ax2.set_ylabel('Test MAE')
    # plt.title('Training and Test MAE')
    # ax2.legend(loc='upper right')
    # plt.tight_layout()
    # plt.savefig(r'..\training\logs\%s\training_mae.png' %(def_epochs))

plot_tensorboard_logs(log_directory)

'''running the model on original data'''
model_reload = UNet(input_channels=21, output_channels=1)
model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
#run pretrained model from above on the original data
def run_model_on_full_data(model, data_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inp, tar, mask in data_loader:
            inp = inp.float()
            tar = tar.float()  
            outputs = model(inp)
            all_outputs.append(outputs.cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs

# Running the model on the entire dataset
y_pred_full = run_model_on_full_data(model_reload, test_loader) #this is now the delta wtd

# Denormalize the wtd data
y = np.load(r'..\data\input_target_example\y.npy')
target_patches_test = split_spatial_data(y, 100, 100)
target_half_patches, rest_half_patches = train_test_split(target_patches_test, test_size=0.8, random_state=10)
target_train_patches, target_val_test_patches = train_test_split(target_half_patches, test_size=0.7, random_state=10)
target_val_patches, target_test_patches = train_test_split(target_val_test_patches, test_size=0.6, random_state=10)  # 20% test, 20% validation

# target_patches_reload = np.load(r'..\data\input_target_example\target_patches.npy')
# target_half_patches, rest_half_patches = train_test_split(target_patches_reload, test_size=0.8, random_state=10)
# target_train_patches, target_val_test_patches = train_test_split(target_half_patches, test_size=0.7, random_state=10)
# target_val_patches, target_test_patches = train_test_split(target_val_test_patches, test_size=0.6, random_state=10)  # 20% test, 20% validation


# y_target = y[train_size+val_size:] #delta wtd
y_target = target_test_patches#.reshape(-1, target_test_patches.shape[2], target_test_patches.shape[3], target_test_patches.shape[4])
#reshape y_target back to original y_test_patches shape as a test
# test = y_target.reshape(target_test_patches.shape[0], target_test_patches.shape[1], target_test_patches.shape[2], target_test_patches.shape[3], target_test_patches.shape[4])
out_var_mean = np.load(r'..\data\input_target_example\out_var_mean.npy')
out_var_std = np.load(r'..\data\input_target_example\out_var_std.npy')
y_pred_reshape = y_pred_full.reshape(y_test_patches.shape[0], y_test_patches.shape[1], y_test_patches.shape[2], y_test_patches.shape[3], y_test_patches.shape[4])
y_pred_denorm = y_pred_reshape#*out_var_std[0] + out_var_mean[0] #denormalise the predicted delta wtd

mask_test_na = np.where(mask_test==0, np.nan, 1)
# Plot the first sample
plt.figure(figsize=(20, 8))
plt.subplot(1, 4, 1)
plt.imshow(y_target[0, 3, 0, :, :]*mask_test_na[0,3,0,:,:], cmap='viridis')
plt.colorbar(shrink=0.8)
plt.title('Actual delta')
plt.tight_layout()

plt.subplot(1, 4, 2)
plt.imshow(y_pred_denorm[0, 3, 0, :, :]*mask_test_na[0,3,0,:,:], cmap='viridis')
plt.colorbar(shrink=0.8)
plt.title('Predicted delta')
plt.tight_layout()

for i in range(y_pred_denorm.shape[0])[:]:
    print(i, range(y_pred_denorm.shape[0]))
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(y_target[i,1, 0, :, :]*mask_test_na[0,1,0,:,:], cmap='viridis')
    plt.colorbar(shrink=0.8)
    plt.title('Actual delta')
    plt.tight_layout()

    plt.subplot(1, 4, 2)
    plt.imshow(y_pred_denorm[i,1, 0, :, :]*mask_test_na[0,1,0,:,:], cmap='viridis')
    plt.colorbar(shrink=0.8)
    plt.title('Predicted delta')

    vmin = min([np.nanmin(y_target[i,1, 0, :, :]*mask_test_na[0,1,0,:,:]),np.nanmin(y_pred_denorm[i,1, 0, :, :]*mask_test_na[0,1,0,:,:])])
    vmax = max([np.nanmax(y_target[i,1, 0, :, :]*mask_test_na[0,1,0,:,:]),np.nanmax(y_pred_denorm[i,1, 0, :, :]*mask_test_na[0,1,0,:,:])])
    plt.subplot(1, 4, 3)
    plt.scatter((y_target[i,1, 0, :, :]*mask_test_na[0,1,0,:,:]).flatten(), (y_pred_denorm[i,1, 0, :, :]*mask_test_na[0,1,0,:,:]).flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Predicted delta') 
    plt.xlabel('Actual delta')

    plt.subplot(1, 4, 4)
    plt.imshow(y_pred_denorm[i,1, 0, :, :]*mask_test_na[0,1,0,:,:] - y_target[i, 1, 0, :, :]*mask_test_na[0,1,0,:,:], cmap='RdBu')
    plt.colorbar(shrink=0.8)
    plt.title('Difference')

    plt.savefig(r'%s\plot_timesplit_area_%s.png' %(log_dir_fig, i))



'''past classic time series split'''
# wtd_prior = wtd[:, :, :] #wtd for the month before the delta wtd
# mask_na = np.where(mask==0, np.nan, 1) 
# # mask_na = np.flip(mask_na, axis=1)
# mask_na = mask_na[:1,:,:]
# mask_na_wtd = mask_na[0, :, :]
# map_tile_plot = map_tile['Band1'].isel(time=1)

# for i in range(y_pred_denorm.shape[0])[:5]:
#     print(i)
#     pred_wtd = wtd_prior[i] + y_pred_denorm[i] 

#     vmax = max([pred_wtd.max(),wtd_prior[i:, :, :].max()])
#     vmin = min([pred_wtd.min(),wtd_prior[i:, :, :].min()])
#     lim = np.max([np.abs(vmax), np.abs(vmin)])
    
#     plt.figure(figsize=(20, 8))
#     plt.subplot(2, 4, 1)
#     plt.title('Actual WTD (OG) month %s' %(i+2))
#     plt.imshow(wtd_prior[i+1, :, :]*mask_na_wtd, cmap='viridis', vmin=vmin, vmax=vmax) #plot the actual wtd that to compare wtd+delta wtd
#     plt.colorbar(shrink=0.8)
#     plt.tight_layout()

#     pred_wtd_na = pred_wtd*mask_na
#     plt.subplot(2, 4, 2)
#     plt.title('Predicted WTD')
#     plt.imshow(pred_wtd_na[0], cmap='viridis', vmin=vmin,vmax=vmax)#,vmin=vmin,vmax=vmax)
#     plt.colorbar(shrink=0.8)
#     plt.tight_layout()

#     wtd_scatter = wtd_prior[i+1, :, :]*mask_na_wtd
#     pred_wtd_scatter = pred_wtd*mask_na
#     vmin = min([np.nanmin(wtd_scatter),np.nanmin(pred_wtd_scatter)])
#     vmax = max([np.nanmax(wtd_scatter),np.nanmax(pred_wtd_scatter)])
#     plt.subplot(2, 4, 3)
#     plt.scatter(wtd_scatter.flatten(), pred_wtd_scatter.flatten(),alpha=0.5, facecolors='none', edgecolors='r')
#     plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
#     plt.xlabel('OG WTD')
#     plt.xlim(vmin,vmax)
#     plt.ylim(vmin,vmax)
#     plt.ylabel('Simulated WTD')
#     plt.title(f"OG vs Sim WTD")
#     plt.tight_layout()

#     diff = (pred_wtd*mask_na) - (wtd_prior[i+1, :, :]*mask_na_wtd)  #difference between wtd and calculated wtd
#     vmax = np.nanmax(diff)
#     vmin = np.nanmin(diff)
#     lim = np.max([np.abs(vmax), np.abs(vmin)])
#     plt.subplot(2, 4, 4)
#     plt.title('diff sim vs actual wtd')
#     pcm = plt.imshow(diff[0], cmap='RdBu',vmin=-lim, vmax=lim)#norm=SymLogNorm(linthresh=1))#norm=colors.CenteredNorm()) #difference between wtd and calculated wtd
#     plt.colorbar(pcm, orientation='vertical',shrink=0.8)
#     plt.tight_layout()

#     vmax = np.nanmax(y_target[i,:,:]*mask_na)
#     vmin = np.nanmin(y_target[i,:,:]*mask_na)
#     lim = np.max([np.abs(vmax), np.abs(vmin)])
#     y_target_map = y_target[i,:,:]*mask_na
#     plt.subplot(2, 4, 5)
#     pcm = plt.imshow(y_target_map[0], cmap='RdBu',  vmin=-lim, vmax=lim)# norm=SymLogNorm(linthresh=1)) # delta wtd that was target
#     plt.colorbar(pcm, orientation='vertical', shrink=0.8)
#     plt.title(f"OG delta WTD")
#     plt.tight_layout()

#     vmax = np.nanmax(y_pred_denorm[i,:,:]*mask_na)
#     vmin = np.nanmin(y_pred_denorm[i,:,:]*mask_na)
#     lim = np.max([np.abs(vmax), np.abs(vmin)])
#     y_pred_map = y_pred_denorm[i,:,:]*mask_na
#     plt.subplot(2, 4, 6)
#     pcm = plt.imshow(y_pred_map[0], cmap='RdBu', vmin=-lim, vmax=lim)#, norm=SymLogNorm(linthresh=1))#norm=colors.CenteredNorm())# norm=SymLogNorm(linthresh=1)) #delta wtd that was predicted
#     plt.colorbar(pcm, orientation='vertical', shrink=0.8)
#     plt.title(f"predicted delta WTD")    
#     plt.tight_layout()

#     y_target_map = y_target[i,:,:]*mask_na
#     y_pred_map = y_pred_denorm[i,:,:]*mask_na
#     vmin = min([np.nanmin(y_target_map),np.nanmin(y_pred_map)])
#     vmax = max([np.nanmax(y_target_map),np.nanmax(y_pred_map)])
#     plt.subplot(2, 4, 7)
#     plt.scatter(y_target_map, y_pred_map,alpha=0.5, facecolors='none', edgecolors='r')
#     plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
#     plt.xlabel('OG delta WTD')
#     plt.xlim(vmin,vmax)
#     plt.ylim(vmin,vmax)
#     plt.ylabel('Simulated delta WTD')
#     plt.title(f"OG vs Sim delta WTD")
#     plt.tight_layout()

#     ax = plt.subplot(2, 4, 8)
#     map_tile_plot.plot(ax=ax)
#     ax.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
#     plt.tight_layout()
    
#     plt.savefig(r'%s\plot_timesplit_%s.png' %(log_dir_fig, i))

