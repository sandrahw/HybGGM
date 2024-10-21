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
general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'
# define the number of months in the dataset and the number of epochs
data_length = 72 # 72 number of months in current dataset (needed for dataprep function to extend the static parameters)    
data_short= 72
def_epochs = 10
lr_rate = 0.001
batchSize = 1
kernel = 3
trainSize = 0.3
testSize = 0.4
sequence_length = 6

# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)


'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs\%s_%s_%s_%s_%s_%s' %(def_epochs, lr_rate ,batchSize, kernel, sequence_length,data_short)
log_dir_fig = r'..\training\logs\%s_%s_%s_%s_%s_%s\spatial_eval_plots' %(def_epochs, lr_rate ,batchSize, kernel, sequence_length,data_short)
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
# plt.subplot(1, 2, 2)
# plt.title('Mask')
# plt.imshow(mask[0,:,:])

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


'''shorten data for experiment'''
X_norm_arr = X_norm_arr[:data_short, :, :, :]
y_norm_arr = y_norm_arr[:data_short, :, :, :]

'''introduce sequence shift for lsmt aspect'''
# Define the sequence length (e.g., 5 months)
# sequence_length = 2

# Prepare input (X) and target (y) sequences
X_norm_sequences = []
y_norm_sequences = []

# Loop over the data to create sequences
for i in range(len(X_norm_arr) - sequence_length):
    X_seq = X_norm_arr[i:i+sequence_length, :, :, :]  # Input sequence of length 5
    y_seq = y_norm_arr[i+sequence_length, :, :, :]    # Target is the next time step after the input sequence
    X_norm_sequences.append(X_seq) 
    y_norm_sequences.append(y_seq)  
# Convert lists to numpy arrays
X_sequences = np.array(X_norm_sequences)  # Shape: (num_sequences, sequence_length, 22, 360, 360)
y_sequences = np.array(y_norm_sequences)  # Shape: (num_sequences, 1, 360, 360)

'''temporal split the patches into training, validation and test sets'''
X_train = X_sequences[:int(trainSize*len(X_sequences)), :, :, :, :]
X_val = X_sequences[int(trainSize*len(X_sequences)):(int(trainSize*len(X_sequences))+(int((1-(trainSize+testSize))*len(X_sequences)))), :, :, :, :]
X_test = X_sequences[int((1-(testSize))*len(X_sequences)):, :, :, :, :]

y_train = y_sequences[:int(trainSize*len(y_sequences)), :, :, :]
y_val = y_sequences[int(trainSize*len(y_sequences)):(int(trainSize*len(y_sequences))+(int((1-(trainSize+testSize))*len(y_sequences)))), :, :, :]
y_test = y_sequences[int((1-(testSize))*len(y_sequences)):, :, :, :]


'''remove mask in every training, validation and test patch'''
def remove_mask_patch(r):
    r_train = r[:, :, :-1, :, :]
    r_mask = r[:, :, -1, :, :]
    r_mask = np.where(r_mask<=0, 0, 1)
    mask_bool = r_mask.astype(bool)
    mask_bool = mask_bool[:, :, np.newaxis, :, :]
    return r_train, mask_bool

X_train, mask_train = remove_mask_patch(X_train)
X_val, mask_val = remove_mask_patch(X_val)
X_test, mask_test = remove_mask_patch(X_test)
np.save(r'%s/X_train.npy' %log_directory, X_train)
np.save(r'%s/X_val.npy' %log_directory, X_val)
np.save(r'%s/X_test.npy' %log_directory, X_test)

np.save(r'%s/mask_train.npy' %log_directory, mask_train)
np.save(r'%s/mask_val.npy' %log_directory, mask_val)
np.save(r'%s/mask_test.npy' %log_directory, mask_test)

np.save(r'%s/y_train.npy' %log_directory, y_train)
np.save(r'%s/y_val.npy' %log_directory, y_val)
np.save(r'%s/y_test.npy' %log_directory, y_test)



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
train_loader = DataLoader(CustomDataset(X_train, y_train, mask_train), batch_size=batchSize, shuffle=False)
validation_loader = DataLoader(CustomDataset(X_val, y_val, mask_val), batch_size=batchSize, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test, mask_test), batch_size=batchSize, shuffle=False)



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

class UNetLSTM_Option1(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, lstm_layers, dropout):
        super(UNetLSTM_Option1, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # LSTM after decoder
        self.lstm = nn.LSTM(input_size=484, hidden_size=hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        # Decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        print('batch_size, seq_len, channels, height, width', batch_size, seq_len, channels, height, width)

        # Flatten input for CNN part
        x = x.view(batch_size * seq_len, channels, height, width)
        print('xshape', x.shape)
        # Encoder
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)
        print('x1', x1.shape)   
        print('x2', x2.shape)
        print('x3', x3.shape)
        print('x4', x4.shape)
        
         # Bottleneck
        bottleneck = self.bottleneck(p4)
        print('bottleneck', bottleneck.shape)
        batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck = bottleneck.size()
        print('batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck', batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck)

        # Reshape for LSTM
        bottleneck_flat = bottleneck.view(batch_size_bottleneck, channels_bottleneck, -1)  
        print('bottleneckflat', bottleneck_flat.shape)
        # LSTM
        lstm_out, _ = self.lstm(bottleneck_flat)  
        print('lstm_out', lstm_out.shape)
        # Reshape LSTM output back to spatial form
        lstm_out_reshaped = lstm_out.view(batch_size*batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck)
        print('lstm_out_reshaped', lstm_out_reshaped.shape)
        # Decoder
        d1 = self.decoder1(lstm_out_reshaped, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)

        # Final output
        out = self.final_conv(d4)
        # print('out', out.shape)
        out = out.view(batch_size, seq_len, output_channels, height, width)

        return out
class UNetLSTM_Option2(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, lstm_layers=1):
        super(UNetLSTM_Option2, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # LSTM after decoder
        self.lstm = nn.LSTM(input_size=484, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        # Decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        print('batch_size, seq_len, channels, height, width', batch_size, seq_len, channels, height, width)

        # Flatten input for CNN part
        x = x.view(batch_size * seq_len, channels, height, width)
        print('xshape', x.shape)
        # Encoder
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)
        print('x1', x1.shape)   
        print('x2', x2.shape)
        print('x3', x3.shape)
        print('x4', x4.shape)
        
         # Bottleneck
        bottleneck = self.bottleneck(p4)
        print('bottleneck', bottleneck.shape)
        batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck = bottleneck.size()
        print('batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck', batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck)

        # Reshape for LSTM
        bottleneck_flat = bottleneck.view(batch_size_bottleneck, channels_bottleneck, -1)  
        print('bottleneckflat', bottleneck_flat.shape)
        # LSTM
        lstm_out, _ = self.lstm(bottleneck_flat)  
        print('lstm_out', lstm_out.shape)
        # Reshape LSTM output back to spatial form
        lstm_out_reshaped = lstm_out.view(batch_size*batch_size_bottleneck, channels_bottleneck, height_bottleneck, width_bottleneck)
        print('lstm_out_reshaped', lstm_out_reshaped.shape)
        # Decoder
        d1 = self.decoder1(lstm_out_reshaped, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)

        # Final output
        out = self.final_conv(d4)
        # print('out', out.shape)
        out = out.view(batch_size, seq_len, output_channels, height, width)

        return out
    

class UNetLSTM_Option3(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, lstm_layers=1):
        super(UNetLSTM_Option3, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
    
        # Decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        # LSTM and FC (placeholder for dynamic input size initialization)
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = None
        self.fc = None

        # Final output layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        # print('batch_size, seq_len, channels, height, width', batch_size, seq_len, channels, height, width)

        # Flatten input for CNN part
        x = x.view(batch_size * seq_len, channels, height, width)
        # print('xshape', x.shape)
        # Encoder
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)
        # print('x1', x1.shape)   
        # print('x2', x2.shape)
        # print('x3', x3.shape)
        # print('x4', x4.shape)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        # print('bottleneck', bottleneck.shape)
        # Decoder
        d1 = self.decoder1(bottleneck, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)

        # print('d1', d1.shape)
        # print('d2', d2.shape)
        # print('d3', d3.shape)
        # print('d4', d4.shape)

        # Get the spatial dimensions of the decoder output dynamically
        d4_batch, d4_channels, d4_height, d4_width = d4.size()
        print('Decoder output shape:', d4_batch, d4_channels, d4_height, d4_width)

        # Initialize LSTM and FC only once, based on the spatial dimensions of the first forward pass
        if self.lstm is None:
            lstm_input_size = d4_channels * d4_height * d4_width
            self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, lstm_input_size)

        # Reshape decoder output for LSTM
        d4_flat = d4.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 64 * height * width]

        # LSTM (process temporal information)
        lstm_out, _ = self.lstm(d4_flat)  # Shape: [batch_size, seq_len, hidden_dim]
        print('LSTM output shape:', lstm_out.shape)

        # Apply fully connected layer to map LSTM output back to spatial dimensions
        lstm_out_mapped = self.fc(lstm_out)
        print('Mapped LSTM output shape:', lstm_out_mapped.shape)

        # Reshape to spatial dimensions (batch_size * seq_len, 64, height, width)
        lstm_out_reshaped = lstm_out_mapped.view(batch_size * seq_len, d4_channels, d4_height, d4_width)
        print('Reshaped LSTM output shape:', lstm_out_reshaped.shape)

        # Final output
        out = self.final_conv(lstm_out_reshaped)
        print('Final output shape before reshaping:', out.shape)

        # Reshape to match the sequence format (batch_size, seq_len, channels, height, width)
        out = out.view(batch_size, seq_len, -1, d4_height, d4_width)
        print('Final output shape:', out.shape)

        return out
    
writer = SummaryWriter(log_dir=log_directory)
# Instantiate the model
input_channels = 21  # Number of input variables (channels)
output_channels = 1  # Single output channel (groundwater head)
hidden_dim = 484  # Number of LSTM units
lstm_layers = 1  # Number of LSTM layers
dropout =0

model = UNetLSTM_Option1(input_channels, output_channels, hidden_dim, lstm_layers, dropout)
# model = UNetLSTM_Option3(input_channels, output_channels, hidden_dim, lstm_layers)
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

# RMSE function
def rmse_cnnlstm(outputs, targets, mas):
    return torch.sqrt(F.mse_loss(outputs[mas], targets[mas]))


def train_model_UnetLSTM(model, train_loader, validation_loader, optimizer, num_epochs, writer=None):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0

        for inputs, targets, masks in train_loader:
            inputs = inputs.float()
            # print('input shape orig', inputs.shape)
            targets = targets.float()
            # print('target shape orig', targets.shape)
            masks = masks.bool()
            # print('mask shape orig', masks.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print('output shape', outputs.shape)
            if targets.shape != outputs.shape:
                batch, seql, chan, h, w = outputs.shape
                # print('seql, chan, h, w', seql, chan, h, w)
                targets = targets.expand(batch, seql, chan, h, w)
            # print('target shape ext', targets.shape)	
            loss = rmse_cnnlstm(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        if writer:
            writer.add_scalar('Loss/train_epoch', running_train_loss / len(train_loader), epoch)
  
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets, masks in validation_loader:
                inputs = inputs.float()
                targets = targets.float()

                outputs = model(inputs)
                if targets.shape != outputs.shape:
                    batch, seql, chan, h, w = outputs.shape
                    # print('seql, chan, h, w', seql, chan, h, w)
                    targets = targets.expand(batch, seql, chan, h, w)
                # print('target shape ext', targets.shape)	
                loss = rmse_cnnlstm(outputs, targets, masks)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(validation_loader.dataset)

        if writer:
            writer.add_scalar('Loss/validation_epoch', running_val_loss / len(validation_loader), epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(log_directory, 'best_model.pth'))
            print("Best model saved!")
    return

train_model_UnetLSTM(model, train_loader, validation_loader, optimizer, num_epochs=def_epochs, writer=writer)



def test_model_UnetLSTM(model, test_loader, writer=None):
    model.eval()
    running_test_loss = 0.0
    all_outputs = []

    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs = inputs.float()
            targets = targets.float()
            masks = masks.bool()

            outputs = model(inputs)
            if targets.shape != outputs.shape:
                    batch, seql, chan, h, w = outputs.shape
                    # print('seql, chan, h, w', seql, chan, h, w)
                    targets = targets.expand(batch, seql, chan, h, w)
            # Compute the loss
            loss = rmse_cnnlstm(outputs, targets, masks)
            running_test_loss += loss.item() * inputs.size(0)

            # Store outputs for further analysis or visualization if needed
            all_outputs.append(outputs.cpu().numpy())

        # Compute the average loss, RMSE, and MAE for the entire test dataset
        test_loss = running_test_loss / len(test_loader.dataset)
        # Log test metrics if using a writer (e.g., TensorBoard)
        if writer:
            writer.add_scalar('Loss/test_epoch', test_loss)
        # Combine all output batches into a single array
        all_outputs = np.concatenate(all_outputs, axis=0)

        # Print test results
        print(f"Test Loss: {test_loss:.4f}")

    return all_outputs

test_outputs = test_model_UnetLSTM(model, test_loader,  writer=writer)

# Train the model

# Training function
# def train_model_UnetLSTM1(model, train_loader, validation_loader, optimizer, num_epochs, writer=None):
#     best_val_loss = float('inf')
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         model.train()
#         running_train_loss = 0.0

#         for inputs, targets, masks in train_loader:
#             inputs = inputs.float()
#             # print('input shape orig', inputs.shape)
#             targets = targets.float()
#             # print('target shape orig', targets.shape)
#             masks = masks.bool()
#             # print('mask shape orig', masks.shape)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             # print('output shape', outputs.shape)
#             if targets.shape != outputs.shape:
#                 seql, chan, h, w = outputs.shape
#                 # print('seql, chan, h, w', seql, chan, h, w)
#                 targets = targets.expand(seql, chan, h, w)
#             # print('target shape ext', targets.shape)	
#             loss = rmse_cnnlstm(outputs, targets, masks[0,:,:,:,:])
#             loss.backward()
#             optimizer.step()

#             running_train_loss += loss.item()
#         epoch_train_loss = running_train_loss / len(train_loader.dataset)
#         if writer:
#             writer.add_scalar('Loss/train_epoch', running_train_loss / len(train_loader), epoch)
  
        
#         # Validation Phase
#         model.eval()
#         running_val_loss = 0.0

#         with torch.no_grad():
#             for inputs, targets, masks in validation_loader:
#                 inputs = inputs.float()
#                 targets = targets.float()

#                 outputs = model(inputs)
#                 if targets.shape != outputs.shape:
#                     seql, chan, h, w = outputs.shape
#                     # print('seql, chan, h, w', seql, chan, h, w)
#                     targets = targets.expand(seql, chan, h, w)
#                 loss = rmse_cnnlstm(outputs, targets, masks[0,:,:,:,:])
#                 running_val_loss += loss.item() * inputs.size(0)
        
#         epoch_val_loss = running_val_loss / len(validation_loader.dataset)

#         if writer:
#             writer.add_scalar('Loss/validation_epoch', running_val_loss / len(validation_loader), epoch)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
 
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             torch.save(model.state_dict(), os.path.join(log_directory, 'best_model.pth'))
#             print("Best model saved!")
#     return

# def test_model_UnetLSTM1(model, test_loader, writer=None):
#     model.eval()
#     running_test_loss = 0.0
#     all_outputs = []

#     with torch.no_grad():
#         for inputs, targets, masks in test_loader:
#             inputs = inputs.float()
#             targets = targets.float()
#             masks = masks.bool()

#             outputs = model(inputs)
#             if targets.shape != outputs.shape:
#                 seql, chan, h, w = outputs.shape
#                 # print('seql, chan, h, w', seql, chan, h, w)
#                 targets = targets.expand(seql, chan, h, w)

#             # Compute the loss
#             loss = rmse_cnnlstm(outputs, targets, masks[0,:,:,:,:])
#             running_test_loss += loss.item() * inputs.size(0)

#             # Store outputs for further analysis or visualization if needed
#             all_outputs.append(outputs.cpu().numpy())

#         # Compute the average loss, RMSE, and MAE for the entire test dataset
#         test_loss = running_test_loss / len(test_loader.dataset)
#         # Log test metrics if using a writer (e.g., TensorBoard)
#         if writer:
#             writer.add_scalar('Loss/test_epoch', test_loss)
#         # Combine all output batches into a single array
#         all_outputs = np.concatenate(all_outputs, axis=0)

#         # Print test results
#         print(f"Test Loss: {test_loss:.4f}")

#     return all_outputs
# # Train the model
# train_model_UnetLSTM1(model, train_loader, validation_loader, optimizer, num_epochs=def_epochs, writer=writer)

# test_outputs = test_model_UnetLSTM1(model, test_loader,  writer=writer)

def plot_tensorboard_logs(log_dir):
    # List all event files in the log directory
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]

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
    loss_val = event_acc.Scalars('Loss/validation_epoch')
    loss_test = event_acc.Scalars('Loss/test_epoch')

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
    
   
    # Plot the training and test losses
    fig, ax1 = plt.subplots()
    ax1.plot(stepstr, train_loss, label='Train Loss', color='blue')
    ax1.plot(stepsva, val_loss, label='Validation Loss', color='green')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss (rmse)')
    ax1.legend(loc='upper right')
    ax1.scatter(stepste, test_loss, label='Test Loss', color='orange')
    plt.title('Training, validation and test Loss')
    plt.tight_layout()
    plt.savefig(r'%s\training_loss.png' %(log_dir))

plot_tensorboard_logs(log_directory)

'''running the model on original data'''
# input_channels = 21  # Number of input variables (channels)
# output_channels = 1  # Single output channel (groundwater head)
# hidden_dim = 484  # Number of LSTM units
# lstm_layers = 1  # Number of LSTM layers

# model_reload = UNetLSTM_Option1(input_channels, output_channels, hidden_dim, lstm_layers)
# model_reload = UNetLSTM_Option2(input_channels, output_channels, hidden_dim, lstm_layers)
# model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
#run pretrained model from above on the original data
def run_model_on_full_data(model, data_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inp, tar, mask in data_loader:
            inp = inp.float()
            # tar = tar.float()  
            outputs = model(inp)
            all_outputs.append(outputs[:, -1].cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs


# X_test = np.load(r'%s/X_train.npy'%log_directory)
# mask_test = np.load(r'%s/mask_test.npy' %log_directory)
# y_test =  np.load(r'%s/y_test.npy' %log_directory)

# class CustomDataset(Dataset):
#     def __init__(self, data, labels, masks, transform=None):
#         """
#         Args:
#             data (torch.Tensor or numpy array): Input data (e.g., images).
#             labels (torch.Tensor or numpy array): Corresponding labels for the input data.
#             masks (torch.Tensor or numpy array): Masks corresponding to each input data.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample (e.g., for data augmentation).
#         """
#         self.data = data
#         self.labels = labels
#         self.masks = masks
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Get the input data, label, and mask for the given index
#         input_data = self.data[idx]
#         label = self.labels[idx]
#         mask = self.masks[idx]

#         # Apply any transformations if specified
#         if self.transform:
#             input_data = self.transform(input_data)

#         return input_data, label, mask

# test_loader = DataLoader(CustomDataset(X_test, y_test, mask_test), batch_size=batchSize, shuffle=False)

# Running the model on the entire dataset
# y_pred_full = run_model_on_full_data(model_reload, test_loader) #this is now the delta wtd
y_pred_full = run_model_on_full_data(model, test_loader) #this is now the delta wtd
# Denormalize the wtd data
# y= np.load(r'%s/y.npy' %log_directory)
y_short = y[:data_short]
target_test = y_short[-int(len(y_short)*testSize):] #delta wtd
# out_var_mean = np.load(r'%s\out_var_mean.npy' %log_directory)
# out_var_std = np.load(r'%s\out_var_std.npy'%log_directory)
y_pred_denorm = y_pred_full*out_var_std[0] + out_var_mean[0] #denormalise the predicted delta wtd

mask_test_na = np.where(mask_test==0, np.nan, 1)
mask_test_na = mask_test_na[0,0,: ,:,:]

for i in range(y_pred_denorm.shape[0])[:]:
    print(i, range(y_pred_denorm.shape[0]))

    # vmax = max([y_pred_denorm[i, 0, :, :].max(),target_test[i, 0, :, :].max()])
    # vmin = min([y_pred_denorm[i, 0, :, :].min(),target_test[i, 0, :, :].min()])
    # lim = np.max([np.abs(vmax), np.abs(vmin)])
    vminR = np.percentile(y_pred_denorm[i, 0, :, :], 5)
    vmaxR = np.percentile(y_pred_denorm[i, 0, :, :], 95)
    vminT = np.percentile(target_test[i, 0, :, :], 5)
    vmaxT = np.percentile(target_test[i, 0, :, :], 95)
    vmax = np.max([vmaxR, vmaxT])
    vmin = np.min([vminR, vminT])

    lim = np.max([np.abs(vmax), np.abs(vmin)])

    target_map = target_test[i, 0, :, :]*mask_test_na
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(target_map[0,:,:], cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    plt.title('Actual delta (colorbar 5-95 percentile)')
    plt.tight_layout()

    pred_map = y_pred_denorm[i, 0, :, :]*mask_test_na
    plt.subplot(1, 4, 2)
    plt.imshow(pred_map[0,:,:], cmap='RdBu',vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    plt.title('Predicted delta (colorbar 5-95 percentile)')

    vmin = min([np.nanmin(target_test[i, 0, :, :]*mask_test_na),np.nanmin(y_pred_denorm[i, 0, :, :]*mask_test_na)])
    vmax = max([np.nanmax(target_test[i, 0, :, :]*mask_test_na),np.nanmax(y_pred_denorm[i, 0, :, :]*mask_test_na)])
    plt.subplot(1, 4, 3)
    plt.scatter((target_test[i,0, :, :]*mask_test_na).flatten(), (y_pred_denorm[i, 0, :, :]*mask_test_na).flatten(),alpha=0.5, facecolors='none', edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Predicted delta') 
    plt.xlabel('Actual delta')

    plt.subplot(1, 4, 4)
    diff = (target_test[i, 0, :, :]*mask_test_na) - (y_pred_denorm[i, 0, :, :]*mask_test_na) #difference between wtd and calculated wtd
    vmax = np.nanmax(np.percentile(diff,95))
    vmin = np.nanmin(np.percentile(diff,5))
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.imshow(diff[0,:,:], cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    plt.title('Difference target-predicted (colorbar 5-95 percentile)')

    plt.savefig(r'%s\plot_timesplit_%s.png' %(log_dir_fig, i))

