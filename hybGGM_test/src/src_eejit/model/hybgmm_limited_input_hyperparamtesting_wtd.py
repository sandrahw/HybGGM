'''load libraries'''
import os
import sys 
import random
import torch
import numpy as np
import models
import train
import data

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

args = sys.argv
# args = ['X' , 1.0, 50, 0.0001, 32, '180', 'UNet2'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type
samplesize = args[1]
samplesize = float(samplesize)
epochs = args[2]
epochs = int(epochs)
learning_rate = args[3]
learning_rate = float(learning_rate)
batch_size = args[4]
batch_size = int(batch_size)
area = args[5]
area = int(area)
model_type = args[6]
model_type = str(model_type)
patience = 10
# sub_batch_size = 1

'''create folders'''
print('define paths for cnn samples and testing data')
sys.stdout.flush()
cnn_sample_path ='/scratch/depfg/hausw001/data/globgm/tiles_input/tile_048-163/transient/cnn_samples_10_10_10/'
testing_path = 'data/testing_random_sampling_fulltile_%s_101010'%area #path to folders with training,testing, validation data
if not os.path.exists(testing_path):
    os.makedirs(testing_path)

print('create log directory for tensorboard logs')
sys.stdout.flush()
log_directory = 'results/testing/random_sampling_fulltile_%s_101010_limitedInpSel_wtd/%s_%s_%s_%s_%s' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
log_dir_fig = 'results/testing/random_sampling_fulltile_%s_101010_limitedInpSel_wtd/%s_%s_%s_%s_%s/plots' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)

print('check if testing prediction files are already there and if so stop the process')
sys.stdout.flush()
if os.path.exists(log_directory + '/yfull_pred_denorm_new.npy'):
    print('Hyperparameter tuning already done and prediction files are there')
    sys.stdout.flush()
    exit()
else:
    print('Hyperparameter tuning not done yet - start process')
    sys.stdout.flush()

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

print('check if input data is already prepared for data loaders and if not prepare it')
sys.stdout.flush()
if os.path.exists(testing_path+'/tar_validation_norm_arr.npy'):
    print('data testing folder already prepared - reloading data')
    sys.stdout.flush()
    train_loader, validation_loader, test_loader = data.RS_reload_data_wtd(testing_path, batch_size)
else:
    print('data testing folder not prepared - prepare data')   
    sys.stdout.flush()
    train_loader, validation_loader, test_loader = data.RS_cnn_data_prep_target_wtd(cnn_sample_path, limitedInpSel, testing_path, batch_size)

print('train loader check', next(iter(train_loader))[0].shape)
print('validation loader check', next(iter(validation_loader))[0].shape)
print('test loader check', next(iter(test_loader))[0].shape)


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
    model = models.UNet2(input_channels=12, output_channels=1)
    model.to(device)
    print('memory after loading model')
    # print(torch.cuda.memory_summary())

'''train models'''
print('train models')
sys.stdout.flush()
# train.RS_train_test_model_newrmse(model, train_loader, test_loader, learning_rate, epochs, log_directory, patience, device, batch_size, sub_batch_size, writer=writer)    
train.RS_train_test_model_newrmse_wtdtarget(model, train_loader, test_loader, learning_rate, epochs, log_directory, patience, device, writer=writer)


'''validate models'''
print('validate models')
sys.stdout.flush()
# train.RS_val_model_newrmse(model, validation_loader,  device, writer=writer)
train.RS_val_model_newrmse_wtdtarget(model, validation_loader,  device, writer=writer)



'''run full predictions'''
print('run prediction based on validation set')
sys.stdout.flush() 
if model_type == 'UNet2':
    print('reload model')
    model_reload = models.UNet2(input_channels=12, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    model_reload.to(device)

if model_type == 'UNet6':   
    print('reload model')
    model_reload = models.UNet6(input_channels=12, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    model_reload.to(device)

y_pred_raw = models.RS_run_model_on_full_data(model_reload, validation_loader, device) #this is now the delta wtd

# full_tile_subsample_path = 'data/testing_random_sampling_fulltile_180_inclwtd'
# # mean = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/full_inp_var_mean_globgm-wtd.npy')
# # std = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/data/testing_random_sampling_fulltile_180_inclwtd/full_inp_var_std_globgm-wtd.npy')
    
out_var_test_mean = np.load('%s/full_inp_var_mean_globgm-wtd.npy' %testing_path)
out_var_test_std = np.load('%s/full_inp_var_std_globgm-wtd.npy' %testing_path)
y_pred_denorm = y_pred_raw*out_var_test_std+ out_var_test_mean
np.save('%s/yfull_pred_denorm_new.npy' %log_directory, y_pred_denorm)
np.save('%s/yfull_pred_raw_new.npy' %log_directory, y_pred_raw)
print('validation prediction finished')	
sys.stdout.flush()

'''run full predictions'''
print('run full predictions')
sys.stdout.flush()
samplepath = '%s/subsamples' %testing_path
if not os.path.exists(samplepath):
    os.makedirs(samplepath)
sub_samples = np.arange(0, 100, 1)
for ss in sub_samples[:]:
    sample_arrays = []
    for param in limitedInpSel[:]:
        print(param, ss)        
        samples_load = np.load('%s/subsamples/full_input_%s_subsample_%s.npy' % (testing_path, param, ss))
        sample_arrays.append(samples_load)
    # stack the arrays together
    sample_arrays = np.stack(sample_arrays, axis=1)
    np.save('%s/full_input_subsample_%s.npy' %(samplepath, ss), sample_arrays)


print('load model')
sys.stdout.flush()
if model_type == 'UNet2':
    model_reload = models.UNet2(input_channels=12, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    model_reload = model_reload.to(device)  
if model_type == 'UNet6':   
    model_reload = models.UNet6(input_channels=12, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    model_reload = model_reload.to(device)
sub_samples = np.arange(0, 100, 1)
fullpred_path = '%s/full_pred' %log_directory
if not os.path.exists(fullpred_path):
    os.makedirs(fullpred_path)
for ss in sub_samples[:]:
    print('prediction sample %s' %ss)
    sys.stdout.flush()
    input_norm = np.load('%s/full_input_subsample_%s.npy' %(samplepath, ss))
    target_norm = np.load('%s/full_target_wtd_subsample_%s.npy' %(samplepath, ss))
    mask = np.load('%s/full_mask.npy' %(testing_path))
    mask = mask[:len(target_norm),:,:]  
    np.save('%s/full_mask_subsample.npy' %(samplepath), mask)
    mask = np.load('%s/full_mask_subsample.npy' %(samplepath))
    full_run_loader = data.DataLoader(data.CustomDataset(input_norm, target_norm, mask), batch_size=batch_size, shuffle=False)
    print('run full prediction')
    sys.stdout.flush()
    y_pred_full = models.RS_run_model_on_full_data(model_reload, full_run_loader, device)
    out_var_test_mean = np.load('%s/full_inp_var_mean_globgm-wtd.npy' %testing_path)
    out_var_test_std = np.load('%s/full_inp_var_std_globgm-wtd.npy'%testing_path)
    y_pred_denorm_full = y_pred_full*out_var_test_std.item() + out_var_test_mean.item()

    np.save('%s/y_pred_denorm_full_%s.npy' %(fullpred_path,ss), y_pred_denorm_full)
    np.save('%s/y_pred_raw_full_%s.npy' %(fullpred_path,ss), y_pred_full)
    print('prediction sample %s done' %ss)	
    sys.stdout.flush()