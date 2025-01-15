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
        lat = data_cut.lat.values
        lon = data_cut.lon.values
        time = data_cut.time.values
       

    if param in params_initial: #repeat the initial values for each month as they are static
        data_arr = np.repeat(data_cut.to_array().values, data_length, axis=0)
        data_arr = np.where(data_arr==np.nan, 0, data_arr)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)
        lat = data_cut.lat.values
        lon = data_cut.lon.values
        time = np.arange(data_length)
    np.save(r'%s\%s.npy'%(log_directory, param), data_arr)
    return data_arr, lat, lon, time
# Set seed before any training or data loading happens
set_seed(10)

'''define general path, data length, case area, number of epochs, learning rate and batch size'''
# define the number of months in the dataset and the number of epochs
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
def_epochs = 10
lr_rate = 0.0001
batchSize = 1
kernel = 3
targetvar = 'head'
patience = 5
trainsize = 0.3
testsize= 0.3
valsize = 1 - trainsize - testsize
# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)

    
'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs_dev2\%s_%s_%s_%s_%s_CNN' %(targetvar, def_epochs, lr_rate ,batchSize, kernel)
log_dir_fig = r'..\training\logs_dev2\%s_%s_%s_%s_%s_CNN\figures' %(targetvar, def_epochs, lr_rate ,batchSize, kernel)
temp_model_output = r'..\data\temp_model_output'
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

params_sel = ['abstraction_uppermost_layer',
              'bed_conductance_used',
              'bottom_uppermost_layer',
              'drain_conductance',
              'drain_elevation_uppermost_layer',
              'horizontal_conductivity_uppermost_layer',
              'initial_head_uppermost_layer',
              'net_RCH', 
              'primary_storage_coefficient_uppermost_layer',
              'surface_water_bed_elevation_used',
              'surface_water_elevation',
              'top_uppermost_layer',
              'vertical_conductivity_uppermost_layer',
              'wtd'
            ]   
# select the files that are needed for the input
params_sel = params_monthly + params_initial
selInFiles = [f for f in inFiles if f.split('\\')[-1].split('.')[0] in params_sel] 
'''prepare the data for input by cropping the data to the specified lat and lon bounds for test regions'''
datacut = []
for f in selInFiles:
    print(f.split('\\')[-1].split('.')[0])
    param = f.split('\\')[-1].split('.')[0]
    data, lat, lon, time = data_prep(f, lat_bounds, lon_bounds, data_length)
    if param == 'wtd':
        wtd = data
        continue
    if param == 'top_uppermost_layer':
        top_upper = data
        datacut.append(data)
    else:
        datacut.append(data)


''''calculate the head for each month - define target (y) and input (X) arrays for the CNN'''
target_head = top_upper - wtd #calculate the head for each month
plt.figure( figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(top_upper[0, :, :], cmap='viridis')
plt.colorbar(shrink=0.5)
plt.title('top_upper')
plt.subplot(1, 3, 2)
plt.imshow(wtd[0, :, :], cmap='viridis')
plt.title('wtd')
plt.colorbar(shrink=0.5)
plt.subplot(1, 3, 3)
plt.imshow(target_head[0, :, :], cmap='viridis')
plt.title('target_head')
plt.colorbar(shrink=0.5)

X_all = np.stack(datacut, axis=1)
X = X_all
y = target_head[:, np.newaxis, :, :] 
np.save(r'%s\X.npy'%log_directory, X)
np.save(r'%s\y.npy'%log_directory, y)

'''normalising the data for every array and save mean and std for denormalisation'''
inp_var_mean = [] # list to store normalisation information for denormalisation later
inp_var_std = []
X_norm = []
for i in range(X.shape[1])[:]:
    mean = X[:, i, :, :].mean()
    std = X[:, i, :, :].std()
    # check if every value in array is 0, if so, skip normalisation
    if X[:, i, :, :].max() == 0 and X[:, i, :, :].min() == 0:
        print('skipped normalisation for array %s' %i)
        X_temp = X[:, i, :, :]
    else:
        X_temp = (X[:, i, :, :] - mean) / std
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
        y_temp = y[:, i, :, :]
    else:
        y_temp = (y[:, i, :, :] - mean) / std
    y_temp = (y[:, i, :, :] - mean) / std
    y_norm.append(y_temp)
    out_var_mean.append(mean)
    out_var_std.append(std)
y_norm_arr = np.array(y_norm)
y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
np.save(r'%s\y_norm_arr.npy'%log_directory, y_norm_arr)
np.save(r'%s\out_var_mean.npy'%log_directory, out_var_mean)
np.save(r'%s\out_var_std.npy'%log_directory, out_var_std)

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
model = UNet2(input_channels=X.shape[1], output_channels=1)



''' RS target wtd'''
def rmse_cnn(outputs, targets, mas):
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

def CNN_train_test_model_head(model, train_loader, test_loader, lr_rate, num_epochs, save_path, patience, writer=None):
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
                print('mask dim', masks.dim(), masks.shape)
            # print('calc rmse')
            loss = rmse_cnn(outputs, targets, masks) 
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
                    print('mask dim', masks.dim(), masks.shape)
                loss = rmse_cnn(outputs, targets, masks)# / accumulation_steps
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

def CNN_val_model_head(model, validation_loader, writer=None):
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
        
            loss = rmse_cnn(outputs, targets, masks)# / accumulation_steps
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

CNN_train_test_model_head(model, train_loader, test_loader, lr_rate, def_epochs, log_directory, patience, writer=writer)
CNN_val_model_head(model, validation_loader, writer=None)

def CNN_run_model(model, data_loader):
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
y_pred_val = CNN_run_model(model, validation_loader)
y_pred_val_denorm = y_pred_val * out_var_std[0] + out_var_mean[0]
y_pred_val_nc = xr.DataArray(y_pred_val_denorm[:,0,:,:], dims=['time', 'lat', 'lon'], coords={'time': time[-len(y_val):], 'lat': lat, 'lon': lon})
y_pred_val_nc.to_netcdf(r'%s\val_pred_denorm.nc'%log_directory)

full_data_loader = DataLoader(CustomDataset(X_norm_arr, y_norm_arr, mask), batch_size=batchSize, shuffle=False)
y_pred_full = CNN_run_model(model, full_data_loader)
np.save(r'%s\full_pred.npy'%log_directory, y_pred_full)
y_pred_full_denorm = y_pred_full * out_var_std[0] + out_var_mean[0]
y_pred_full_denorm_reshape = y_pred_full_denorm.reshape(y_pred_full_denorm.shape[0], y_pred_full_denorm.shape[2], y_pred_full_denorm.shape[3])
y_pred_full_nc = xr.DataArray(y_pred_full_denorm_reshape, dims=['time', 'lat', 'lon'], coords={'time': time, 'lat': lat, 'lon': lon})
y_pred_full_nc.to_netcdf(r'%s\full_pred_denorm.nc'%log_directory)
y_pred_full_nc.to_netcdf(r'%s\full_pred_denorm_UNet.nc'%temp_model_output)

min_val = np.min([np.nanmin(y), np.nanmin(y_pred_full_denorm)])
max_val = np.max([np.nanmax(y), np.nanmax(y_pred_full_denorm)])
lim = np.max([np.abs(min_val), np.abs(max_val)])

diff = y[0, 0, :, :] - y_pred_full_denorm[0, 0, :, :]
mindiff = np.nanmin(np.percentile(diff, 1))
maxdiff = np.nanmax(np.percentile(diff, 99))
limdiff = np.max([np.abs(mindiff), np.abs(maxdiff)])
plt.figure(figsize=(15,10))
plt.subplot(1, 3, 1)
plt.imshow(y_pred_full_denorm[0, 0, :, :], cmap='viridis', label='predicted',vmin=min_val, vmax=max_val)    
plt.colorbar(shrink=0.5)
plt.title('predicted')
plt.subplot(1, 3, 2)
plt.imshow(y[0, 0, :, :], cmap='viridis', label='target',vmin=min_val, vmax=max_val)
plt.title('target')
plt.colorbar(shrink=0.5)
plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='RdBu', label='diff',vmin=-limdiff, vmax=limdiff)
plt.title('difference')
plt.colorbar(shrink=0.5)
plt.savefig(r'%s\prediction.png'%log_dir_fig)


plt.figure(figsize=(10,10))
plt.scatter(y[0, 0, :, :], y_pred_full_denorm[0, 0, :, :])
plt.xlabel('target')
plt.ylabel('predicted')
plt.savefig(r'%s\scatterplot.png'%log_dir_fig)