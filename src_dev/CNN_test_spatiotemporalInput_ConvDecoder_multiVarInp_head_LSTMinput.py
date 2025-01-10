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
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
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
# general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'
# define the number of months in the dataset and the number of epochs
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
def_epochs = 10
lr_rate = 0.0001
batchSize = 1
kernel = 3
targetvar = 'head'
patience = 5
# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)


# small_lon_bounds = (7, 10) #CH bounds(5,10)#360x360
# small_lat_bounds = (47, 50)#CH bounds(45,50)#360x360
# large_lon_bounds = (5, 11) #CH bounds(5,10)
# large_lat_bounds = (45, 51)#CH bounds(45,50)
# half_lon_bounds = (5, 12.5) 
# half_lat_bounds = (45, 52.5)

# map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
# fig, axs = plt.subplots(1, 1, figsize=(10, 10))
# map_tile.Band1[0].plot(ax=axs)
# axs.add_patch(plt.Rectangle((lon_bounds[0], lat_bounds[0]), lon_bounds[1] - lon_bounds[0], lat_bounds[1] - lat_bounds[0], fill=None, color='red'))
# axs.add_patch(plt.Rectangle((small_lon_bounds[0], small_lat_bounds[0]), small_lon_bounds[1] - small_lon_bounds[0], small_lat_bounds[1] - small_lat_bounds[0], fill=None, color='red'))
# axs.add_patch(plt.Rectangle((large_lon_bounds[0], large_lat_bounds[0]), large_lon_bounds[1] - large_lon_bounds[0], large_lat_bounds[1] - large_lat_bounds[0], fill=None, color='blue'))
# axs.add_patch(plt.Rectangle((half_lon_bounds[0], half_lat_bounds[0]), half_lon_bounds[1] - half_lon_bounds[0], half_lat_bounds[1] - half_lat_bounds[0], fill=None, color='green'))
# plt.tight_layout()
    
'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs\%s_%s_%s_%s_%s_nextMonth_LSTMheadinput' %(targetvar, def_epochs, lr_rate ,batchSize, kernel)
log_dir_fig = r'..\training\logs\%s_%s_%s_%s_%s_nextMonth_LSTMheadinput\spatial_eval_plots' %(targetvar, def_epochs, lr_rate ,batchSize, kernel)
#create folder in case not there yet
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)


'''create mask (for land/ocean)'''
map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
map_cut = map_tile.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
plt.imshow(map_cut.Band1[0, :, :])
mask = map_cut.to_array().values
# mask where everything that is nan is 0 and everything else is 1
mask = np.nan_to_num(mask, copy=False, nan=0)
mask = np.where(mask==0, 0, 1)
mask = mask[0, :, :]
# plt.imshow(mask[0, :, :])
np.save(r'%s\mask.npy'%log_directory, mask)

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

params_sel = ['bed_conductance_used_para',
            'initial_head_uppermost_layer_para',
            'top_uppermost_layer',
            'horizontal_conductivity_uppermost_layer',
            'bottom_uppermost_layer',
            'vertical_conductivity_uppermost_layer',
            'drain_conductance',
            'primary_storage_coefficient_uppermost_layer',
            'surface_water_elevation',
            'net_RCH',
            'drain_elevation_uppermost_layer',
            'abstraction_uppermost_layer',
            'surface_water_bed_elevation_used']

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
    np.save(r'%s\%s.npy'%(log_directory, param), data_arr)
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

LSTMheadinput = xr.open_dataset(r'..\data\all_predictions_LSTM.nc', decode_times=False)
head_predlstm = LSTMheadinput.to_array()[0, :, :, :]
# plot wtd for one month, indicating range of values
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('WTD for one month')
# plt.imshow(wtd[0, :, :], cmap='viridis') #the array version of the input data is flipped
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.title('Mask')
# plt.imshow(mask[0,:,:])

''''calculate the head for each month - define target (y) and input (X) arrays for the CNN'''
target_head = top_upper - wtd #calculate the head for each month
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(top_upper[0, :, :], cmap='viridis')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(wtd[0, :, :], cmap='viridis')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(target_head[0, :, :], cmap='viridis')
plt.colorbar()

# target_head = np.diff(wtd, axis=0) #calculate the head for each month
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(wtd[0, :, :], cmap='viridis')
# plt.colorbar()
# plt.subplot(1, 3, 2)
# plt.imshow(wtd[1, :, :], cmap='viridis')
# plt.colorbar()
# plt.subplot(1, 3, 3)
# plt.imshow(target_head[0, :, :], cmap='viridis')
# plt.colorbar()

y = target_head[1:, np.newaxis, :, :] #shape[timesteps, channels, lat, lon]
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
              head_predlstm], axis=1)
            #   wtd
            #   ], axis=1)
X = X_All[:-1,:,:,:] #remove first month to match the delta wtd data
np.save(r'%s\X.npy'%log_directory, X)
np.save(r'%s\y.npy'%log_directory, y)

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
np.save(r'%s\X_norm_arr.npy'%log_directory, X_norm_arr)
np.save(r'%s\inp_var_mean.npy'%log_directory, inp_var_mean)
np.save(r'%s\inp_var_std.npy'%log_directory, inp_var_std)

out_var_mean = []
out_var_std = []
y_norm = []
for i in range(y.shape[1]):
    mean = y[:, i, :, :].mean()
    std = y[:, i, :, :].std()
    # check if every value in array is 0, if so, skip normalisation
    if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        y_temp = X[:, i, :, :]
    else:
        y_temp = (X[:, i, :, :] - mean) / std
    y_temp = (y[:, i, :, :] - mean) / std
    y_norm.append(y_temp)
    out_var_mean.append(mean)
    out_var_std.append(std)
y_norm_arr = np.array(y_norm)
y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
np.save(r'%s\y_norm_arr.npy'%log_directory, y_norm_arr)
np.save(r'%s\out_var_mean.npy'%log_directory, out_var_mean)
np.save(r'%s\out_var_std.npy'%log_directory, out_var_std)


'''split the patches into training, validation and test sets'''
# X_train, X_val_test = train_test_split(X_norm_arr, test_size=0.7, random_state=10)
# X_test, X_val = train_test_split(X_val_test, test_size=0.6, random_state=10)  # 20% test, 20% validation

# y_train, y_val_test = train_test_split(y_norm_arr, test_size=0.7, random_state=10)
# y_test, y_val = train_test_split(y_val_test, test_size=0.6, random_state=10)  # 20% test, 20% validation

# mask_train, mask_val_test = train_test_split(mask, test_size=0.7, random_state=10)
# mask_test, mask_val = train_test_split(mask_val_test, test_size=0.6, random_state=10)  # 20% test, 20% validation

# mask_train = mask_train[:, np.newaxis, :, :]
# mask_test = mask_test[:, np.newaxis, :, :]
# mask_val = mask_val[:, np.newaxis, :, :]

'''split the patches into training, validation and test sets but keep the time series in order'''
trainsize = 0.3
testsize= 0.3
valsize = 0.4
X_train = X_norm_arr[:int(X_norm_arr.shape[0]*trainsize), :, :, :]
X_test = X_norm_arr[int(X_norm_arr.shape[0]*trainsize):int(X_norm_arr.shape[0]*(trainsize+testsize)), :, :, :]
X_val = X_norm_arr[int(X_norm_arr.shape[0]*(trainsize+testsize)):, :, :, :]

y_train = y_norm_arr[:int(y_norm_arr.shape[0]*trainsize), :, :, :]
y_test = y_norm_arr[int(y_norm_arr.shape[0]*trainsize):int(y_norm_arr.shape[0]*(trainsize+testsize)), :, :, :]
y_val = y_norm_arr[int(y_norm_arr.shape[0]*(trainsize+testsize)):, :, :, :]

mask = mask[1:, :, :]
mask_train = mask[:int(y_norm_arr.shape[0]*trainsize), np.newaxis, :, :]
mask_test = mask[int(y_norm_arr.shape[0]*trainsize):int(y_norm_arr.shape[0]*(trainsize+testsize)), np.newaxis,:, :]
mask_val = mask[int(y_norm_arr.shape[0]*(trainsize+testsize)):, np.newaxis,:, :]

np.save(r'%s\mask_train.npy'%log_directory, mask_train)
np.save(r'%s\mask_val.npy'%log_directory, mask_val)
np.save(r'%s\mask_test.npy'%log_directory, mask_test)

np.save(r'%s\X_train.npy'%log_directory, X_train)
np.save(r'%s\X_val.npy'%log_directory, X_val)
np.save(r'%s\X_test.npy'%log_directory, X_test)

np.save(r'%s\y_train.npy'%log_directory, y_train)
np.save(r'%s\y_val.npy'%log_directory, y_val)
np.save(r'%s\y_test.npy'%log_directory, y_test)


'''transform the data into tensors'''
def transformArrayToTensor(array):
    return torch.from_numpy(array).float()
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

train_loader = DataLoader(CustomDataset(X_train, y_train, mask_train), batch_size=batchSize, shuffle=False)
validation_loader = DataLoader(CustomDataset(X_val, y_val, mask_val), batch_size=batchSize, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test, mask_test), batch_size=batchSize, shuffle=False)
print(next(iter(train_loader))[0].shape)

class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of two Conv2D layers,
    each followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
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

class UNet6(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet6, self).__init__()
        
        # Encoder: Downsampling path
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 1024)#new layer
        self.encoder6 = EncoderBlock(1024, 2048)#new layer

        # Bottleneck layer (middle part of the U-Net)
        self.bottleneck = ConvBlock(2048, 4096)

        # Decoder: Upsampling path
        self.decoder1 = DecoderBlock(4096, 2048)
        self.decoder2 = DecoderBlock(2048, 1024)
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder5 = DecoderBlock(256, 128)
        self.decoder6 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        x2, p2 = self.encoder2(p1) # Second block
        x3, p3 = self.encoder3(p2) # Third block
        x4, p4 = self.encoder4(p3) # Fourth block
        x5, p5 = self.encoder5(p4) # Fifth block
        x6, p6 = self.encoder6(p5) # Sixth block

        # Bottleneck (middle)
        bottleneck = self.bottleneck(p6)

        # Decoder
        d1 = self.decoder1(bottleneck, x6)  # Upsample from bottleneck and add skip connection from encoder4
        d2 = self.decoder2(d1, x5)          # Continue with decoder and corresponding skip connection from encoder
        d3 = self.decoder3(d2, x4)          # Continue with decoder and corresponding skip connection from encoder
        d4 = self.decoder4(d3, x3)          # Continue with decoder and corresponding skip connection from encoder
        d5 = self.decoder5(d4, x2)          # Continue with decoder and corresponding skip connection from encoder
        d6 = self.decoder6(d5, x1)          # Continue with decoder and corresponding skip connection from encoder
       
        # Final output layer
        output = self.final_conv(d6)  
        return output
    
class UNet4(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet4, self).__init__()
        
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
    
class UNet2(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet2, self).__init__()
        
        # Encoder: Downsampling path
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)

        # Bottleneck layer (middle part of the U-Net)
        self.bottleneck = ConvBlock(128, 256)

        # Decoder: Upsampling path
        self.decoder1 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        x2, p2 = self.encoder2(p1) # Second block

        # Bottleneck (middle)
        bottleneck = self.bottleneck(p2)

        # Decoder  # Continue with decoder and corresponding skip connection from encoder3
        d1 = self.decoder1(bottleneck, x2)          # Continue with decoder and corresponding skip connection from encoder2
        d2 = self.decoder2(d1, x1)          # Continue with decoder and corresponding skip connection from encoder1

        # Final output layer
        output = self.final_conv(d2)        # Reduce to the number of output channels (e.g., 1 for groundwater head)
        return output


# Instantiate the model, define the loss function and the optimizer
writer = SummaryWriter(log_dir=log_directory)
#make sure model uses GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet(input_channels=21, output_channels=1).to(device)
model = UNet2(input_channels=21, output_channels=1)


''' RS target wtd'''
def rmse_new(outputs, targets, mas):
    diff = (targets[mas] - outputs[mas])
    # if torch.isnan(diff).any():
    #     print("NaN in diff")
    squared_diff = diff ** 2
    # if torch.isnan(squared_diff).any():
    #     print("NaN in squared_diff")
    mse = torch.mean(squared_diff)
    # if torch.isnan(mse).any():
    #     print("NaN in mse")
    rmse_value = torch.sqrt(mse)
    return rmse_value

def RS_train_test_model_newrmse_head(model, train_loader, test_loader, lr_rate, num_epochs, save_path, patience, writer=None):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    best_test_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0
        len_train_loader = 0
        len_test_loader = 0
        print(next(iter(train_loader))[0].shape)
    
        for i, (inputs, targets, masks) in enumerate(train_loader):
            print(i, 'out of', len(train_loader))
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()
            # print('input shape', inputs.shape)
            # print('target shape', targets.shape)
            # print('mask shape', masks.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs nan:', torch.isnan(outputs).any())
            # print('outputs shape:', outputs.shape)	
           
            if masks.shape != outputs.shape:
                # masks = masks[np.newaxis,:,:,:]
                # print('mask dim', masks.dim(), masks.shape)
            # print('calc rmse')
            loss = rmse_new(outputs, targets, masks) 
            loss.backward()
            optimizer.step()
            # print('train loss item:', loss.item())
            # sys.stdout.flush()    
            if not torch.isnan(loss):
                running_train_loss += loss.item()# * accumulation_steps  # Multiply back for tracking
                len_train_loader += 1
               
        epoch_train_loss = running_train_loss / len_train_loader #if I skip nan values I have to adapt the calc

        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)

        # Testing Phase
        print('testing phase')
        model.eval()
        running_test_loss = 0.0
        # with torch.set_grad_enabled(True):
        with torch.no_grad():
            for i, (inputs, targets, masks) in enumerate(test_loader):
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()
       
                outputs = model(inputs)
                # print('outputs nan:', torch.isnan(outputs).any())
                # print('outputs shape:', outputs.shape)

                if masks.shape != outputs.shape:
                #    masks = masks[np.newaxis,:,:,:]
                    # print('mask dim', masks.dim(), masks.shape)
                loss = rmse_new(outputs, targets, masks)# / accumulation_steps
                # print('test loss item:', loss.item())
           
                if not torch.isnan(loss):
                    running_test_loss += loss.item()
                    len_test_loader += 1
                    # else:
                    #     print("NaN encountered in loss calculation. Skipping this instance.")
            epoch_test_loss = running_test_loss / len_test_loader #if I skip nan values I have to adapt the calc

            if writer:
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Testing Loss: {epoch_test_loss:.4f}")
    
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print("Best model saved!")
            else:
                epochs_without_improvement += 1
                print('no improvement in test loss for', epochs_without_improvement, 'epochs')
            if epochs_without_improvement >= patience:
                    print("Early stopping!")
                    break
    print('training and testing done')
    return
    
def RS_val_model_newrmse_head(model, validation_loader, writer=None):
    model.eval()
    running_val_loss = 0.0
    len_validation_loader = 0
    with torch.no_grad():
        for inputs, targets, masks in validation_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()

            outputs = model(inputs)
            if masks.shape != outputs.shape:
                # masks = masks[np.newaxis,:,:,:]
                print('mask dim', masks.dim(), masks.shape)
        
            loss = rmse_new(outputs, targets, masks)# / accumulation_steps
            # print('val loss item:', loss.item())
        
            if not torch.isnan(loss):
                running_val_loss += loss.item()
                len_validation_loader += 1
                # else:
                #     print("NaN encountered in loss calculation. Skipping this instance.")
        val_loss = running_val_loss / len_validation_loader
        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/val_epoch', val_loss)
        # Print test results
        print(f"Val Loss: {val_loss:.4f}")

    print('validation done')
    return 

RS_train_test_model_newrmse_head(model, train_loader, test_loader, lr_rate, def_epochs, log_directory, patience, writer=writer)
RS_val_model_newrmse_head(model, validation_loader, writer=None)

def plot_tensorboard_logs(log_dir):
    # List all event files in the log directory
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
    # print(event_files)
    # Initialize lists to store the data
    train_loss = []
    val_loss = []
    test_loss = []

    stepstr = []
    stepsva = []
    stepste = []

    # Iterate through all event files and extract data
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()

    # Extract scalars
    loss_train = event_acc.Scalars('Loss/train_epoch')
    loss_test = event_acc.Scalars('Loss/test_epoch')
    # loss_val = event_acc.Scalars('Loss/val_epoch')
    # print(loss_val)
    

    # Append to the lists
    for i in range(len(loss_train)):
        stepstr.append(loss_train[i].step)
        train_loss.append(loss_train[i].value)
    
    for i in range(len(loss_test)):
        stepste.append(loss_test[i].step)
        test_loss.append(loss_test[i].value)


    # for i in range(len(loss_val)):
    #     stepsva.append(loss_val[i].step)
    #     val_loss.append(loss_val[i].value)
            

    # Plot the training and test losses
    fig, ax1 = plt.subplots()
    ax1.plot(stepstr, train_loss, label='Train Loss', color='blue', linestyle='dashed')
    ax1.plot(stepste, test_loss, label='Test Loss', color='green', linestyle='solid')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('loss (RMSE)')
    # ax1.legend(loc='upper left')
    # ax2 = ax1.twinx()
    # ax1.scatter(stepste, val_loss, label='Val Loss', color='orange')
    # ax2.set_ylabel('Test Loss')
    plt.title('Training, testing and validation loss')
    ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(r'%s\training_loss.png' %(log_directory))

plot_tensorboard_logs(log_directory)


'''running the model on original data'''
model_reload = UNet2(input_channels=21, output_channels=1)
model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))

# test_loader /= np.load(r'%s/test_loader.npy' %log_directory)
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
y_pred_full = run_model_on_full_data(model_reload, validation_loader) #this is now the delta wtd


y_reload = np.load(r'%s\y.npy'%log_directory)
# target_train, target_val_test = train_test_split(y, test_size=0.7, random_state=10)
# target_test, target_val = train_test_split(target_val_test, test_size=0.6, random_state=10)  # 20% test, 20% validation
target_val = y_reload[int(y.shape[0]*(trainsize+testsize)):,:,:,:]
# out_var_mean = np.load(r'%s\out_var_mean.npy' %log_directory)
# out_var_std = np.load(r'%s\out_var_std.npy'%log_directory)
y_pred_denorm = y_pred_full*out_var_std[0] + out_var_mean[0] #denormalise the predicted delta wtd

mask_val_na = np.where(mask_val==0, np.nan, 1)

for i in range(y_pred_denorm.shape[0])[:10]:
    print(i, 'of', y_pred_denorm.shape[0])

    vmin = min([np.nanmin(target_val[i, 0, :, :]),np.nanmin(y_pred_denorm[i, 0, :, :])])
    vmax = max([np.nanmax(target_val[i, 0, :, :]),np.nanmax(y_pred_denorm[i, 0, :, :])])
    # lim = np.max([np.abs(vmax), np.abs(vmin)])
    # vminR = np.percentile(y_pred_denorm[i, 0, :, :], 5)
    # vmaxR = np.percentile(y_pred_denorm[i, 0, :, :], 95)
    # vminT = np.percentile(target_val[i, 0, :, :], 5)
    # vmaxT = np.percentile(target_val[i, 0, :, :], 95)
    # vmax = np.max([vmaxR, vmaxT])
    # vmin = np.min([vminR, vminT])

    lim = np.max([np.abs(vmax), np.abs(vmin)])

    target_val = target_val*mask_val_na
    y_pred_denorm = y_pred_denorm*mask_val_na

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(target_val[i, 0, :, :], cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    # plt.title('Actual delta (colorbar 5-95 percentile)')
    plt.title('Actual delta')

    plt.subplot(1, 4, 2)
    plt.imshow(y_pred_denorm[i, 0, :, :], cmap='RdBu',vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    # plt.title('Predicted delta (colorbar 5-95 percentile)')
    plt.title('Predicted delta')

    # vmin = min([np.nanmin(target_val[i, 0, :, :]),np.nanmin(y_pred_denorm[i, 0, :, :])])
    # vmax = max([np.nanmax(target_val[i, 0, :, :]),np.nanmax(y_pred_denorm[i, 0, :, :])])
    plt.subplot(1, 4, 3)
    plt.scatter((target_val[i,0, :, :]).flatten(), (y_pred_denorm[i, 0, :, :]).flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Predicted delta') 
    plt.xlabel('Actual delta')

    plt.subplot(1, 4, 4)
    diff = (target_val[i, 0, :, :]) - (y_pred_denorm[i, 0, :, :]) #difference between wtd and calculated wtd
    vmax = np.nanmax(np.percentile(diff,95))
    vmin = np.nanmin(np.percentile(diff,5))
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.imshow(diff, cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    # plt.title('Difference target-predicted (colorbar 5-95 percentile)')
    plt.title('Difference target-predicted')

    plt.savefig(r'%s\plot_timesplit_%s.png' %(log_dir_fig, i))



'''run full data through model'''   
X_norm_arr = np.load(r'%s\X_norm_arr.npy'%log_directory)
y_norm_arr = np.load(r'%s\y_norm_arr.npy'%log_directory)
mask = np.load(r'%s\mask.npy'%log_directory)

run_loader = DataLoader(CustomDataset(X_norm_arr, y_norm_arr, mask), batch_size=batchSize, shuffle=False)
model_reload = UNet2(input_channels=21, output_channels=1)
model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
y_pred_full = run_model_on_full_data(model_reload, run_loader) #this is now the delta wtd
out_var_mean = np.load(r'%s\out_var_mean.npy' %log_directory)
out_var_std = np.load(r'%s\out_var_std.npy'%log_directory)  
X = np.load(r'%s\X.npy'%log_directory)
heads_lstm = X[:, -1, :, :]
y_pred_full_denorm = y_pred_full*out_var_std[0] + out_var_mean[0] #denormalise the predicted delta wtd
target = np.load(r'%s\y.npy'%log_directory)
target_time_match = target[1:,:,:,:]

min_val = np.min([np.nanmin(target_time_match), np.nanmin(y_pred_full_denorm)])
max_val = np.max([np.nanmax(target_time_match), np.nanmax(y_pred_full_denorm)])
lim = np.max([np.abs(min_val), np.abs(max_val)])
plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
plt.imshow(target_time_match[0, 0, :, :], cmap='viridis', vmin=min_val, vmax=max_val)
plt.title('Actual heads')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 2)
plt.imshow(y_pred_full_denorm[0, 0, :, :], cmap='viridis', vmin=min_val, vmax=max_val)
plt.title('Predicted heads')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 3)
diff = target[0, 0, :, :] - y_pred_full_denorm[0, 0, :, :]
mindiff = np.nanmin(np.percentile(diff, 5))
maxdiff = np.nanmax(np.percentile(diff, 95))
limdiff = np.max([np.abs(mindiff), np.abs(maxdiff)])
plt.imshow(diff, cmap='RdBu', vmin=-limdiff, vmax=limdiff)
plt.colorbar(shrink=0.8)
plt.title('Difference (based on 5 and 95 percentile)')
plt.subplot(3, 3, 4)
plt.imshow(target_time_match[0, 0, :, :], cmap='viridis', vmin=min_val, vmax=max_val)
plt.title('Actual heads')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 5)
plt.imshow(heads_lstm[0, :, :], cmap='viridis', vmin=min_val, vmax=max_val)
plt.title('heads from LSTM')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 6)
difflstm = target[0, 0, :, :] - heads_lstm[0, :, :]
mindiffL = np.nanmin(np.percentile(difflstm, 5))
maxdiffL = np.nanmax(np.percentile(difflstm, 95))
limdiffL = np.max([np.abs(mindiffL), np.abs(maxdiffL)])
plt.imshow(difflstm, cmap='RdBu', vmin=-limdiffL, vmax=limdiffL)
plt.title('Difference heads - LSTM')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 7)
plt.imshow(y_pred_denorm[0, 0, :, :], cmap='viridis', vmin=min_val, vmax=max_val)
plt.title('Predicted heads')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 8)
plt.imshow(heads_lstm[0, :, :], cmap='viridis', vmin=min_val, vmax=max_val)
plt.title('heads from LSTM')
plt.colorbar(shrink=0.8)
plt.subplot(3, 3, 9)
difflstmCNN = y_pred_denorm[0, 0, :, :] - heads_lstm[0, :, :]
mindiffLCNN = np.nanmin(np.percentile(difflstmCNN, 5))
maxdiffLCNN = np.nanmax(np.percentile(difflstmCNN, 95))
limdiffLCNN = np.max([np.abs(mindiffLCNN), np.abs(maxdiffLCNN)])
plt.imshow(difflstmCNN, cmap='RdBu', vmin=-limdiffLCNN, vmax=limdiffLCNN)
plt.title('Difference CNN - LSTM')
plt.colorbar(shrink=0.8)
plt.tight_layout
plt.savefig(r'%s\plot_full_data_map_check.png' %log_dir_fig)

'''pick random cells for timeseries check'''
#select random cells combination from the 360x360 grid
xs = np.random.randint(0, 360, 8)
ys = np.random.randint(0, 360, 8)
plt.figure(figsize=(20, 8))
plt.subplot(3, 3, 1)
plt.imshow(y_pred_full_denorm[0, 0, :, :])
plt.scatter(xs, ys, color='r')
plt.colorbar(shrink=0.8)

for i, (x, y) in enumerate(zip(xs, ys)):
    print('cell', i)
    plt.subplot(3, 3, i+2)
    plt.plot(target_time_match[:, 0, x, y], label='Actual')
    plt.plot(y_pred_full_denorm[:, 0, x, y], label='Predicted')
    plt.plot(heads_lstm[:, x, y], label='LSTM')
    plt.legend()
plt.savefig(r'%s\plot_full_data_timeseries_check.png' %log_dir_fig)