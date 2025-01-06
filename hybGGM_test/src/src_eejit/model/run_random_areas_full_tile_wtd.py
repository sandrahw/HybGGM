#CNN_test
import os
import sys 
import random
import torch
import numpy as np
import data
import models


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

if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

print('define sample path, tile, number of epochs, learning rate and batch size, etc')
sys.stdout.flush()
args = sys.argv
args = ['scriptpath' , 1.0, 10, 0.0001, 32, '180', 'UNet2'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type
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
# if batch_size<sub_batch_size:
#     sub_batch_size = batch_size

print('define paths for cnn samples and testing data')
sys.stdout.flush()
cnn_sample_path ='/scratch/depfg/hausw001/data/globgm/tiles_input/tile_048-163/transient/cnn_samples_10_10_10/'
testing_path = '../../data/testing_random_sampling_fulltile_%s_101010'%area #path to folders with training,testing, validation data
full_tile_subsample_path = '../../data/testing_random_sampling_fulltile_%s_101010/subsamples'%area
if not os.path.exists(testing_path):
    os.makedirs(testing_path)   
if not os.path.exists(full_tile_subsample_path):
    os.makedirs(full_tile_subsample_path)

print('create log directory for tensorboard logs')
sys.stdout.flush()
log_directory = '../../results/testing/random_sampling_fulltile_%s_101010_limitedInpSel_wtd/%s_%s_%s_%s_%s' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
log_dir_fig = '../../results/testing/random_sampling_fulltile_%s_101010_limitedInpSel_wtd/%s_%s_%s_%s_%s/plots' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)


print('check if testing prediction files are already there and if so stop the process')
sys.stdout.flush()
if os.path.exists(log_directory + '/yfull_pred_denorm_new.npy'):
    print('Hyperparameter tuning already done and prediction files are there')
    sys.stdout.flush()
else:
    print('Hyperparameter tuning not done yet - stop')
    sys.stdout.flush()
    exit()


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
def RS_normalize_full_samples_sep_var_X(data, log_dir, param):
    X = data
    mean = X.mean()
    std = X.std()
    if X.max() == 0 and X.min() == 0:
        print('skipped normalisation for array' )
        X_norm_arr = X
    else:
        X_norm_arr = (X - mean) / std
    if np.isnan(X_norm_arr).any():
        print('nan in array')
        print(mean, std)
        print('replace nan values with 0')
        X_norm_arr = np.nan_to_num(X_norm_arr, copy=False, nan=0)
    print(X_norm_arr.shape)
    # X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3, 4)
    np.save('%s/full_inp_norm_arr_%s.npy'%(log_dir, param), X_norm_arr)
    np.save('%s/full_inp_var_mean_%s.npy'%(log_dir, param), mean)
    np.save('%s/full_inp_var_std_%s.npy'%(log_dir, param), std)

    return X_norm_arr, mean, std

def RS_normalize_full_samples_sep_var_y(data, log_dir):
    y = data
    mean = y.mean()
    std = y.std()
    if y.max() == 0 and y.min() == 0:
        print('skipped normalisation for array')
        y_norm_arr = y
    else:
        y_norm_arr = (y - mean) / std

    if np.isnan(y_norm_arr).any():
        print('nan in array')
        print(mean, std)
        print('replace nan values with 0')
        y_norm_arr = np.nan_to_num(y_norm_arr, copy=False, nan=0)
    print(y_norm_arr.shape)
    # y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    np.save('%s/full_tardeltawtd_norm_arr.npy'%(log_dir), y_norm_arr)
    np.save('%s/full_out_var_mean.npy'%(log_dir), mean)
    np.save('%s/full_out_var_std.npy'%(log_dir), std)
    return y_norm_arr, mean, std

print('check if subsamples are already created otherwise create them')
sys.stdout.flush()
if os.path.exists('%s/full_input_globgm-wtd_subsample_99.npy' % (full_tile_subsample_path)):
    print('subsamples already created')
    sys.stdout.flush()
else:
    print('subsamples not created yet - start process')
    sys.stdout.flush()
    for param in limitedInpSel[:]:
            print(f'param: {param}, %s/full_array_%s.npy' % (cnn_sample_path, param))
            samples_load = np.load('%s/full_array_%s.npy' % (cnn_sample_path, param))
            #print size of the array
            # print("%d bytes" % (samples_load.size * samples_load.itemsize))
            # print(samples_load.shape)
            #check if sample has nan or inf values and replace them with 0
            if param == 'globgm-wtd':
                print('create delta wtd')
                t_0 = samples_load[:,:1,0, :, :] #wtd
                t_min1 = samples_load[:,1:,0,:, :] #wtd
                target = t_0 #this is the delta wtd
                np.save('%s/full_target_wtd.npy' %(testing_path), target)
                print('target shape', target.shape) 
                samples_load_sel = t_min1 #
                print(param, samples_load_sel.shape)
                mask = np.nan_to_num(target, copy=False, nan=0)
                mask = np.where(mask==0, 0, 1)
                # print(mask)
                mask_bool = mask.astype(bool)
                np.save('%s/full_mask.npy' %(testing_path), mask_bool)
                # print(mask_bool)
                print('normalise target data')
                y_norm_arr, mean, std = RS_normalize_full_samples_sep_var_y(target, testing_path)
                print('divide input data into 100 samples')	
                # print(y_norm_arr.shape)
                samplesize = y_norm_arr.shape[0]//100
                n_samples = np.arange(0, y_norm_arr.shape[0], samplesize)
                m_samples = np.arange(0, 100, 1)
                for n, m in zip(n_samples[:], m_samples[:]):
                    # print(n, m)
                    sample = y_norm_arr[n:n+samplesize]
                    # print(sample.shape)
                    np.save('%s/full_target_wtd_subsample_%s.npy' %(full_tile_subsample_path, m), sample)
            else:
                #include only the previous timestep
                samples_load_sel = samples_load[:,1:,:, :]
                # print(param, samples_load_sel.shape)

            if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                print(f'nan or inf values in {param}')
                samples_load_sel = np.nan_to_num(samples_load_sel, copy=False, nan=0)
                samples_load_sel = np.where(samples_load_sel==np.nan, 0, samples_load_sel)
                samples_load_sel = np.where(samples_load_sel==np.inf, 0, samples_load_sel)
                if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                    print(f'nan or inf values STILL in {param}')

            # np.save('%s/full_input_%s.npy' %(log_directory, param), samples_load_sel)
            print('normalise input data')
            X_norm_arr, mean, std = RS_normalize_full_samples_sep_var_X(samples_load_sel, testing_path, param)

            #divide X_norm_arr into 100 samples and save them
            print('divide input data into 100 samples')	
            # print(X_norm_arr.shape)
            samplesize = X_norm_arr.shape[0]//100
            n_samples = np.arange(0, X_norm_arr.shape[0], samplesize)
            m_samples = np.arange(0, 100, 1)
            for n, m in zip(n_samples[:], m_samples[:]):
                # print(n, m)
                sample = X_norm_arr[n:n+samplesize]
                # print(sample.shape)
                np.save('%s/full_input_%s_subsample_%s.npy' %(full_tile_subsample_path, param, m), sample)
            

# print('check if input and target data subsamples are already stacked together otherwise stack them together')
# sys.stdout.flush()
# if os.path.exists('%s/full_input_subsample_99.npy' % (full_tile_subsample_path)):
#     print('subsamples already stacked together')
#     sys.stdout.flush()
# else:   
#     sub_samples = np.arange(0, 100, 1)
#     for ss in sub_samples[:]:
#         sample_arrays = []
#         for param in limitedInpSel[:]:
#             print(param, ss)        
#             samples_load = np.load('%s/full_input_%s_subsample_%s.npy' % (full_tile_subsample_path, param, ss))
#             sample_arrays.append(samples_load)
#         sample_arrays = np.stack(sample_arrays, axis=1)
#         np.save('%s/full_input_subsample_%s.npy' %(full_tile_subsample_path, ss), sample_arrays)

# print('check if prediction submodel is already there and if so continue with full prediction')
# sys.stdout.flush()
# if os.path.exists(log_directory + 'full_pred/y_pred_denorm_full_99.npy'):
#     print('full prediction there')
#     sys.stdout.flush()
#     exit()
# else:
#     print('load model')
#     sys.stdout.flush()
#     if model_type == 'UNet2':
#         model_reload = models.UNet2(input_channels=12, output_channels=1)
#         model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
#         model_reload = model_reload.to(device)  
#     if model_type == 'UNet6':   
#         model_reload = models.UNet6(input_channels=12, output_channels=1)
#         model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
#         model_reload = model_reload.to(device)
#     sub_samples = np.arange(0, 100, 1)
#     for ss in sub_samples[:]:
#         print('prediction sample %s' %ss)
#         sys.stdout.flush()
#         input_norm = np.load('%s/full_input_subsample_%s.npy' %(full_tile_subsample_path, ss))
#         target_norm = np.load('%s/full_target_deltawtd_subsample_%s.npy' %(full_tile_subsample_path, ss))
#         mask = np.load('%s/full_mask.npy' %(testing_path))
#         mask = mask[:len(target_norm),:,:]  
#         np.save('%s/full_mask_subsample.npy' %(full_tile_subsample_path), mask)
#         mask = np.load('%s/full_mask_subsample.npy' %(full_tile_subsample_path))
#         full_run_loader = data.DataLoader(data.CustomDataset(input_norm, target_norm, mask), batch_size=batch_size, shuffle=False)
#         print('run full prediction')
#         sys.stdout.flush()
#         y_pred_full = models.RS_run_model_on_full_data(model_reload, full_run_loader, device)
#         out_var_test_mean = np.load('%s/full_out_var_mean.npy' %testing_path)
#         out_var_test_std = np.load('%s/full_out_var_std.npy'%testing_path)
#         y_pred_denorm_full = y_pred_full*out_var_test_std.item() + out_var_test_mean.item()
#         fullpred_path = '%s/full_pred' %log_directory
#         if not os.path.exists(fullpred_path):
#             os.makedirs(fullpred_path)
#         np.save('%s/y_pred_denorm_full_%s.npy' %(fullpred_path,ss), y_pred_denorm_full)
#         np.save('%s/y_pred_raw_full_%s.npy' %(fullpred_path,ss), y_pred_full)
#         print('prediction sample %s done' %ss)	
#         sys.stdout.flush()


