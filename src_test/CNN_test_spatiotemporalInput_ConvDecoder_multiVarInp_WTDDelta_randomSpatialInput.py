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
from scipy.stats import gaussian_kde

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

'''define sample path, case area, number of epochs, learning rate and batch size'''
cnn_sample_path =r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Data\GLOBGM\input\tiles_input\tile_048-163\transient\cnn_samples_180'
def_epochs = 10
lr_rate = 0.0001
batchSize = 64
kernel = 3

'''create log directory for tensorboard logs'''
log_directory = r'..\training\logs\%s_%s_%s_%s' %(def_epochs, lr_rate ,batchSize, kernel)
log_dir_fig = r'..\training\logs\%s_%s_%s_%s\spatial_eval_plots' %(def_epochs, lr_rate ,batchSize, kernel)
#create folder in case not there yet
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)


'''load the modflow files and prepare the data for input'''
params_modflow = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'surface_water_bed_elevation_used',
 'surface_water_elevation', 'net_RCH', 
 'bottom_lowermost_layer', 'bottom_uppermost_layer', 
 'drain_conductance', 
 'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
 'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
 'top_uppermost_layer',
 'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer',
 'globgm-wtd']


def cnn_sample_prep(sample, ccn_samples, log_dir, params_mf):
    print(f'sample: {sample}')
    sample_arrays = []
    for param in params_mf[:]:
        # print(f'param: {param}')
        samples_load = np.load(r'%s\%s_array_%s.npy' % (ccn_samples, sample, param))
        # print(samples_load.shape)
        #check if sample has nan or inf values and replace them with 0
        if param == 'globgm-wtd':
            # print('create delta wtd')
            t_0 = samples_load[:,:1,0, :, :]
            t_min1 = samples_load[:,1:,0,:, :]
            target = t_0 - t_min1 #this is the delta wtd
            #include only the previous timestep for wtd info in input data
            samples_load_sel = target
            # print(param, samples_load_sel.shape) 
            # print('create mask')
            # print(target)
            mask = np.nan_to_num(target, copy=False, nan=0)
            mask = np.where(mask==0, 0, 1)
            # print(mask)
            mask_bool = mask.astype(bool)
            # print(mask_bool)
        else:
            #include only the previous timestep
            samples_load_sel = samples_load[:,1:,:]
            # print(param, samples_load_sel.shape)

        if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
            print(f'nan or inf values in {sample} {param}')
            samples_load_sel = np.nan_to_num(samples_load_sel, copy=False, nan=0)
            samples_load_sel = np.where(samples_load_sel==np.nan, 0, samples_load_sel)
            samples_load_sel = np.where(samples_load_sel==np.inf, 0, samples_load_sel)
            if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                print(f'nan or inf values STILL in {sample} {param}')

        sample_arrays.append(samples_load_sel)
    # stack the arrays together
    sample_arrays = np.stack(sample_arrays, axis=1)

    # np.save(r'%s\input_%s.npy' %(log_dir, sample), sample_arrays)
    # np.save(r'%s\target_%s.npy' %(log_dir, sample), target)
    # np.save(r'%s\mask_%s.npy' %(log_dir, sample), mask_bool)
    return sample_arrays, target, mask_bool

input_train, target_train, mask_train = cnn_sample_prep('training', cnn_sample_path, log_directory, params_modflow)
input_val, target_val, mask_val = cnn_sample_prep('validation', cnn_sample_path, log_directory, params_modflow)
input_test, target_test, mask_test = cnn_sample_prep('testing', cnn_sample_path, log_directory, params_modflow)

# plot wtd for one month, indicating range of values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('WTD for one month')
plt.imshow(target_train[0, 0,:, :], cmap='viridis') #the array version of the input data is flipped
plt.colorbar()

'''normalising the data for every array and save mean and std for denormalisation'''
#TODO normalisation is not necessarily over all the tile data!
def normalize_samples_X(data, sample, log_dir):
    inp_var_mean = [] # list to store normalisation information for denormalisation later
    inp_var_std = []
    X_norm = []
    X = data
    # print('X nan', np.isnan(X).any())
    # print('X inf', np.isinf(X).any())
    # print('X', X)
    for i in range(X.shape[1])[:]:
        # gather the same layer from train, val and test set and combine to one array
        temp = np.concatenate((input_train[:, i, :, :, :], input_val[:, i,:, :, :], input_test[:, i, :, :, :]), axis=0)
        mean = temp.mean()
        std = temp.std()
        # print('mean', mean, 'std', std)
        # print('X max', temp.max())
        # print('X min', temp.min())
        # check if every value in array is 0, if so, skip normalisation
        if X[:, i, :, :, :].max() == 0 and X[:, i, :, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            X_temp = X[:, i,:, :, :]
        else:
            X_temp = (X[:, i, :, :, :] - mean) / std
        if np.isnan(X_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(X_temp, copy=False, nan=0)

      
        # print(mean, std, X_temp)
        X_norm.append(X_temp)
        inp_var_mean.append(mean)
        inp_var_std.append(std)

    #from list to array
    X_norm_arr = np.array(X_norm)
    X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3, 4)
    # np.save(r'%s\inp_%s_norm_arr.npy'%(log_dir, sample), X_norm_arr)
    # np.save(r'%s\inp_%s_var_mean.npy'%(log_dir, sample), inp_var_mean)
    # np.save(r'%s\inp_%s_var_std.npy'%(log_dir, sample), inp_var_std)

    return X_norm_arr, inp_var_mean, inp_var_std

input_train_norm, inp_var_train_mean, inp_var_train_std = normalize_samples_X(input_train, 'training', log_directory)
input_val_norm, inp_var_val_mean, inp_var_val_std = normalize_samples_X(input_val, 'validation', log_directory)
input_test_norm, inp_var_test_mean, inp_var_test_std = normalize_samples_X(input_test, 'testing', log_directory)

def normalize_samples_y(data, sample, log_dir):
    out_var_mean = []
    out_var_std = []
    y_norm = []
    y = data
    # print('X nan', np.isnan(y).any())
    # print('X inf', np.isinf(y).any())
    for i in range(y.shape[1]):
        # gather the same layer from train, val and test set and combine to one array
        temp = np.concatenate((target_train[:, i, :, :], target_val[:, i, :, :], target_test[:, i,  :, :]), axis=0)
        mean = temp.mean()
        std = temp.std()
        # print('mean', mean, 'std', std)
        # print('X max', temp.max())
        # print('X min', temp.min())
        # check if every value in array is 0, if so, skip normalisation
        if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            y_temp = y[:, i, :, :, :]
        else:
            y_temp = (y[:, i, :, :] - mean) / std

        if np.isnan(y_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(y_temp, copy=False, nan=0)


        y_norm.append(y_temp)
        out_var_mean.append(mean)
        out_var_std.append(std)
    y_norm_arr = np.array(y_norm)
    y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    # np.save(r'%s\tar_%s_norm_arr.npy'%(log_dir, sample), y_norm_arr)
    # np.save(r'%s\out_%s_var_mean.npy'%(log_dir, sample), out_var_mean)
    # np.save(r'%s\out_%s_var_std.npy'%(log_dir, sample), out_var_std)
    return y_norm_arr, out_var_mean, out_var_std

target_train_norm, out_var_train_mean, out_var_train_std = normalize_samples_y(target_train, 'training', log_directory)
target_val_norm, out_var_val_mean, out_var_val_std = normalize_samples_y(target_val, 'validation', log_directory)
target_test_norm, out_var_test_mean, out_var_test_std = normalize_samples_y(target_test, 'testing', log_directory)

# np.isnan(target_train_norm).any()
# np.isnan(target_val_norm).any()
# np.isnan(target_test_norm).any()

#
input_train_norm = input_train_norm[:,:,0,:,:]
input_val_norm = input_val_norm[:,:,0,:,:]
input_test_norm = input_test_norm[:,:,0,:,:]

#downsize example by only taking 10% of the data
def downsize_data(data, target, mask, perc):
    data_down = data[:int(data.shape[0]*perc)]
    target_down = target[:int(target.shape[0]*perc)]
    mask_down = mask[:int(mask.shape[0]*perc)]
    return data_down, target_down, mask_down

input_train_norm_down, target_train_norm_down, mask_train_down = downsize_data(input_train_norm, target_train_norm, mask_train, 0.2)
input_val_norm_down, target_val_norm_down, mask_val_down = downsize_data(input_val_norm, target_val_norm, mask_val, 0.2)
input_test_norm_down, target_test_norm_down, mask_test_down = downsize_data(input_test_norm, target_test_norm, mask_test, 0.2)


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
train_loader = DataLoader(CustomDataset(input_train_norm_down, target_train_norm_down, mask_train_down), batch_size=batchSize, shuffle=False)
validation_loader = DataLoader(CustomDataset(input_val_norm_down, target_val_norm_down, mask_val_down), batch_size=batchSize, shuffle=False)
test_loader = DataLoader(CustomDataset(input_test_norm_down, target_test_norm_down, mask_test_down), batch_size=batchSize, shuffle=False)

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
        # batch_size, seq_len, channels, height, width = x.size()
        # print('batch_size, seq_len, channels, height, width', batch_size, seq_len, channels, height, width)

        # Flatten input for CNN part
        # x = x.view(batch_size , seq_len* channels, height, width)
        # print('xshape', x.shape)
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
        # output = self.final_conv(d6)  
        return output
# Instantiate the model, define the loss function and the optimizer
writer = SummaryWriter(log_dir=log_directory)
#make sure model uses GPU if available
model = UNet(input_channels=21, output_channels=1)
# torch.save(model.state_dict(), os.path.join(log_directory, 'model_untrained.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_rate)
# RMSE function
def rmse(outputs, targets, mas):
    return torch.sqrt(F.mse_loss(outputs[mas], targets[mas]))


# Training function
def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, writer=None):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0

        for inputs, targets, masks in train_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()

            # print('inputs dim', inputs.dim(), inputs.shape)
            # print('targets dim', targets.dim(), targets.shape)
            # print('masks dim', masks.dim(), masks.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs dim', outputs.dim(), outputs.shape)
            # loss = criterion(outputs[masks], targets[masks])
            if targets.shape != outputs.shape:
                targets = targets.squeeze(1)
                # print('targets dim', targets.dim(), targets.shape)
            if masks.shape != outputs.shape:
                masks = masks[0,0,:,:,:]
                # print('mask dim', masks.dim(), masks.shape)
            loss = rmse(outputs, targets, masks)
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
                inputs = inputs.float()# inputs.to(device).float()
                targets = targets.float()#targets.to(device).float()
                masks = masks.bool()#masks.to(device).bool()

                outputs = model(inputs)
                if targets.shape != outputs.shape:
                     targets = targets.squeeze(1)
                      # print('targets dim', targets.dim(), targets.shape)
                if masks.shape != outputs.shape:
                     masks = masks[0,0,:,:,:]
                    # print('mask dim', masks.dim(), masks.shape)
    
                loss = rmse(outputs, targets, masks)
                running_val_loss += loss.item()
         
        epoch_val_loss = running_val_loss / len(validation_loader.dataset)

        if writer:
            writer.add_scalar('Loss/validation_epoch', running_val_loss / len(validation_loader), epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(log_directory, 'best_model.pth'))
            print("Best model saved!")

    return

train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=def_epochs, writer=writer)

def test_model(model, test_loader, criterion, writer=None):
    model.eval()
    running_test_loss = 0.0
    all_outputs = []

    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs = inputs.float()# inputs.to(device).float()
            targets = targets.float()#targets.to(device).float()
            masks = masks.bool()#masks.to(device).bool()

            outputs = model(inputs)
            if targets.shape != outputs.shape:
                targets = targets.squeeze(1)
                # print('targets dim', targets.dim(), targets.shape)
            if masks.shape != outputs.shape:
                masks = masks[0,0,:,:,:]
                # print('mask dim', masks.dim(), masks.shape)
            loss = rmse(outputs, targets, masks)
            running_test_loss += loss.item()

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

test_outputs = test_model(model, test_loader, criterion, writer=writer)

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
    loss_val = event_acc.Scalars('Loss/validation_epoch')
    print(loss_val)
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
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Training/Validation Loss')
    # ax1.legend(loc='upper left')
    # ax2 = ax1.twinx()
    ax1.scatter(stepste, test_loss, label='Test Loss', color='orange')
    # ax2.set_ylabel('Test Loss')
    plt.title('Training and Test Loss')
    ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(r'%s\training_loss.png' %(log_directory))

plot_tensorboard_logs(log_directory)

'''running the model on original data'''
model_reload = UNet(input_channels=21, output_channels=1)
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
y_pred_full = run_model_on_full_data(model, test_loader) #this is now the delta wtd


# y = np.load(r'%s\y.npy'%log_directory)
# target_train, target_val_test = train_test_split(y, test_size=0.7, random_state=10)
# target_val, target_test = train_test_split(target_val_test, test_size=0.6, random_state=10)  # 20% test, 20% validation

# out_var_mean = np.load(r'%s\out_var_test_mean.npy' %log_directory)
# out_var_std = np.load(r'%s\out_var_test_std.npy'%log_directory)
y_pred_denorm = y_pred_full*out_var_test_std[0] + out_var_test_mean[0] #denormalise the predicted delta wtd

mask_test_na = np.where(mask_test==0, np.nan, 1)
np.save(r'%s\y_pred_denorm.npy'%log_directory, y_pred_denorm)
np.save(r'%s\target_test.npy'%log_directory, target_test)
for i in range(y_pred_denorm.shape[0])[0:10]:
    print(i, range(y_pred_denorm.shape[0]))

    targetplot = target_test[i, 0, :, :]*mask_test_na[i,0,:,:]
    predplot = y_pred_denorm[i, 0, :, :]*mask_test_na[i,0,:,:]

    #check if all values in targetplot at nan values
    if np.isnan(targetplot).all():
        print('nan values in targetplot')
        plt.figure(figsize=(20, 8))
        plt.plot('only nan values in targetplot - no wtd to be simulated')
        plt.savefig(r'%s\plot_timesplit_%s.png' %(log_dir_fig, i))
        continue

    vminR = np.percentile(predplot, 5)
    vmaxR = np.percentile(predplot, 95)
    vminT = np.percentile(targetplot, 5)
    vmaxT = np.percentile(targetplot, 95)
    vmax = np.max([vmaxR, vmaxT])
    vmin = np.min([vminR, vminT])

    lim = np.max([np.abs(vmax), np.abs(vmin)])

    plt.figure(figsize=(40, 10))
    plt.subplot(1, 5, 1)
    plt.imshow(targetplot, cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    plt.title('Actual delta (colorbar 5-95 percentile)')
    plt.tight_layout()

    plt.subplot(1, 5, 2)
    plt.imshow(predplot, cmap='RdBu',vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    plt.title('Predicted delta (colorbar 5-95 percentile)')

    vmin = min([np.nanmin(targetplot),np.nanmin(predplot)])
    vmax = max([np.nanmax(targetplot),np.nanmax(predplot)])
    
    # Calculate the point density
    mask = ~np.isnan(targetplot)
    xTarget = targetplot[mask] #mask out nan's otherwise kde does not work
    xPred = predplot[mask]   
    xy = np.vstack([xTarget,xPred])
    z = gaussian_kde(xy)(xy)
    vmin = min([np.nanmin(targetplot),np.nanmin(predplot)])
    vmax = max([np.nanmax(targetplot),np.nanmax(predplot)])
    plt.subplot(1, 5, 3)
    plt.scatter(xTarget, xPred,c=z, cmap='plasma', facecolors='none', s=10)#, edgecolors='r')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    plt.ylabel('Predicted delta') 
    plt.xlabel('Actual delta')
    plt.colorbar(shrink=0.8)

    plt.subplot(1, 5, 4)
    diff = targetplot - predplot #difference between wtd and calculated wtd
    vmax = np.nanmax(np.percentile(diff,95))
    vmin = np.nanmin(np.percentile(diff,5))
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.imshow(diff, cmap='RdBu', vmin=-lim, vmax=lim)
    plt.colorbar(shrink=0.8)
    plt.title('Difference target-predicted (colorbar 5-95 percentile)')

    plt.subplot(1, 5, 5)
    # Example maps
    map1 = targetplot
    map2 = predplot
    relative_error = (map1 - map2) / map1
    vmax = np.nanmax(np.percentile(relative_error,95))
    vmin = np.nanmin(np.percentile(relative_error,5))
    lim = np.max([np.abs(vmax), np.abs(vmin)])
    plt.imshow(relative_error, cmap='RdBu', vmin=-lim, vmax=lim)
    plt.title('Relative error (colorbar 5-95 percentile)')
    plt.colorbar(shrink=0.8)
    plt.savefig(r'%s\plot_timesplit_%s.png' %(log_dir_fig, i))







