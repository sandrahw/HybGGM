'''load libraries'''
import os
import sys 
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import models
import train

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
'''define necessary information'''
if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    sys.stdout.flush()
else:
    print("CUDA is not available. Using CPU.")
    sys.stdout.flush()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

samplesize = 1.0
epochs = 50
learning_rate = 0.001
batch_size = 8
area = 180
model_type = 'UNet6'
patience = 10
sub_batch_size = 1
if batch_size<sub_batch_size:
    sub_batch_size = batch_size

'''create folders'''
print('define paths for cnn samples and testing data')
sys.stdout.flush()
testing_path = 'data/testing_random_sampling_fulltile_%s_inclwtd'%area #path to folders with training,testing, validation data
print('create log directory for tensorboard logs')
sys.stdout.flush()
log_directory = 'results/testing/random_sampling_fulltile_%s_inclwtd_limitedInpSel/%s_%s_%s_%s_%s' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
log_dir_fig = 'results/testing/random_sampling_fulltile_%s_inclwtd_limitedInpSel/%s_%s_%s_%s_%s/plots' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)
'''load limited input datasets'''
print('define limited input variables')
sys.stdout.flush()
limitedInpSel = ['abstraction_uppermost_layer',
                 'bed_conductance_used',
                 'drain_elevation_uppermost_layer',
                 'surface_water_bed_elevation_used',
                 'surface_water_elevation', 
                 'net_RCH',
                 'bottom_uppermost_layer',
                 'drain_conductance',
                 'horizontal_conductivity_uppermost_layer',
                 'primary_storage_coefficient_uppermost_layer',
                 'vertical_conductivity_uppermost_layer',
                 'globgm-wtd'] 

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
#find the indices of the selected input variables from the params_modflow list      
selected_indices = [params_modflow.index(i) for i in limitedInpSel]

print('define input data and load target and mask data')
sys.stdout.flush()
input_training = np.load(testing_path + '/inp_training_norm_arr.npy')
input_training_sel = input_training[:,selected_indices]
np.save(testing_path + '/input_training_norm_arr_limitedInpSel.npy', input_training_sel)
input_testing = np.load(testing_path + '/inp_testing_norm_arr.npy')
input_testing_sel = input_testing[:,selected_indices]
np.save(testing_path + '/input_testing_norm_arr_limitedInpSel.npy', input_testing_sel)
input_validation = np.load(testing_path + '/inp_validation_norm_arr.npy')
input_validation_sel = input_validation[:,selected_indices]
np.save(testing_path + '/input_validation_norm_arr_limitedInpSel.npy', input_validation_sel)

target_training = np.load(testing_path + '/tar_training_norm_arr.npy')
target_testing = np.load(testing_path + '/tar_testing_norm_arr.npy')
target_validation = np.load(testing_path + '/tar_validation_norm_arr.npy')

mask_training = np.load(testing_path + '/mask_training.npy')
mask_testing = np.load(testing_path + '/mask_testing.npy')
mask_validation = np.load(testing_path + '/mask_validation.npy')

'''create training, testing, validation loaders'''
print('create training, testing, validation loaders')
sys.stdout.flush()
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
train_loader = DataLoader(CustomDataset(input_training_sel, target_training, mask_training), batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(CustomDataset(input_validation_sel, target_validation, mask_validation), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CustomDataset(input_testing_sel, target_testing, mask_testing), batch_size=batch_size, shuffle=False)

#print trainloader shape
next(iter(train_loader))[0].shape

'''load models'''
print('load models')
sys.stdout.flush()
if model_type == 'UNet6':
    writer = models.SummaryWriter(log_dir=log_directory)
    model = models.UNet6(input_channels=12, output_channels=1)
    model.to(device)
    print('memory after loading model')
    # print(torch.cuda.memory_summary())

if model_type == 'UNet2':
    writer = models.SummaryWriter(log_dir=log_directory)
    model = model.sUNet2(input_channels=12, output_channels=1)
    model.to(device)
    print('memory after loading model')
    # print(torch.cuda.memory_summary())

'''train models'''
print('train models')
sys.stdout.flush()
train.RS_train_test_model(model, train_loader, test_loader, learning_rate, epochs, log_directory, patience, device, batch_size, sub_batch_size,writer=writer)
    
'''validate models'''
print('validate models')
sys.stdout.flush()
train.RS_val_model(model, validation_loader,  device,  batch_size, sub_batch_size, writer=writer)


'''run full predictions'''
print('run prediction based on validation set')
sys.stdout.flush() 
y_pred_raw = models.RS_run_model_on_full_data(model, validation_loader, device) #this is now the delta wtd

out_var_test_mean = np.load('%s/full_out_var_mean.npy' %testing_path)
out_var_test_std = np.load('%s/full_out_var_std.npy'%testing_path)
y_pred_denorm = y_pred_raw*out_var_test_std[0] + out_var_test_mean[0] 
np.save('%s/yfull_pred_denorm_new.npy' %log_directory, y_pred_denorm)
np.save('%s/yfull_pred_raw_new.npy' %log_directory, y_pred_raw)
print('validation prediction finished')	

sys.ValueError('stop here')

'''run full predictions'''
print('load pre normalisation values')
#TODO select again only limited input data
sub_samples = np.arange(0, 100, 1)
for ss in sub_samples[:]:
    sample_arrays = []
    for param in params_modflow[:]:
        print(param, ss)        
        samples_load = np.load('%s/full_input_%s_subsample_%s.npy' % (full_tile_subsample_path, param, ss))
        sample_arrays.append(samples_load)
    # stack the arrays together
    sample_arrays = np.stack(sample_arrays, axis=1)
    np.save('%s/full_input_subsample_%s.npy' %(full_tile_subsample_path, ss), sample_arrays)
#TODO check if model can be run like that for full prediction
print('load model')
sys.stdout.flush()
if model_type == 'UNet2':
    model_reload = models.UNet2(input_channels=21, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    model_reload = model_reload.to(device)  
if model_type == 'UNet6':   
    model_reload = models.UNet6(input_channels=21, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    model_reload = model_reload.to(device)
sub_samples = np.arange(0, 100, 1)
for ss in sub_samples[:]:
    print('prediction sample %s' %ss)
    sys.stdout.flush()
    input_norm = np.load('%s/full_input_subsample_%s.npy' %(full_tile_subsample_path, ss))
    target_norm = np.load('%s/full_target_deltawtd_subsample_%s.npy' %(full_tile_subsample_path, ss))
    mask = np.load('%s/full_mask.npy' %(testing_path))
    mask = mask[:len(target_norm),:,:]  
    np.save('%s/full_mask_subsample.npy' %(full_tile_subsample_path), mask)
    mask = np.load('%s/full_mask_subsample.npy' %(full_tile_subsample_path))
    full_run_loader = data.DataLoader(data.CustomDataset(input_norm, target_norm, mask), batch_size=batch_size, shuffle=False)
    print('run full prediction')
    sys.stdout.flush()
    y_pred_full = models.RS_run_model_on_full_data(model_reload, full_run_loader, device)
    out_var_test_mean = np.load('%s/full_out_var_mean.npy' %testing_path)
    out_var_test_std = np.load('%s/full_out_var_std.npy'%testing_path)
    y_pred_denorm_full = y_pred_full*out_var_test_std.item() + out_var_test_mean.item()
    fullpred_path = '%s/full_pred' %log_directory
    if not os.path.exists(fullpred_path):
        os.makedirs(fullpred_path)
    np.save('%s/y_pred_denorm_full_%s.npy' %(fullpred_path,ss), y_pred_denorm_full)
    np.save('%s/y_pred_raw_full_%s.npy' %(fullpred_path,ss), y_pred_full)
    print('prediction sample %s done' %ss)	
    sys.stdout.flush()