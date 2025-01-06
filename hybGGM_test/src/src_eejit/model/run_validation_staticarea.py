import random
import numpy as np
import torch
import data
import models
import sys
import os
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
print('current path', os.getcwd())

# os.chdir('/eejit/home/hausw001/HybGGM/hybGGM_test')
args = sys.argv
# args = ['X' ,0.4, 0.3, 50, 0.0001, 16, 'UNet6', 'small'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type
testSize = args[1]
testSize = float(testSize)
trainSize = args[2] #validation size is 1-testSize-trainSize #
trainSize = float(trainSize)
epochs = args[3]
epochs = int(epochs)
learning_rate = args[4]
learning_rate = float(learning_rate)
batch_size = args[5]
batch_size = int(batch_size)
model_type = args[6]
area = args[7]

'''create log directory for tensorboard logs'''
'''usual small area'''
if area == 'small':
    lon_bounds = (7, 10) #CH bounds(5,10)#360x360
    lat_bounds = (47, 50)#CH bounds(45,50)#360x360
    log_directory = 'results/testing/small_area_048_initialtesting/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/small_area_048_initialtesting/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)
    #check if model and y_pred_denorm_new.npy are already there and if so stop the process
    if not os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
        print('Folder and files dont exist, move to next hyperparam setup', log_directory)
        exit()
    else:
        print('Hyperparameter tuning already done and prediction files are there now running validation')


        
'''larger area (one quarter of a tile)'''
if area == 'large':
    lon_bounds = (5, 11) #720x720
    lat_bounds = (45, 51)#720x720
    log_directory = 'results/testing/larger_area_048/600/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/larger_area_048/600/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)
    if not os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
        print('Folder and files dont exist, move to next hyperparam setup', log_directory)
        exit()
    else:
        print('Hyperparameter tuning already done and prediction files are there now running validation')

'''half a tile'''
if area == 'half':
    lon_bounds = (5, 12.5) 
    lat_bounds = (45, 52.5)
    log_directory = 'results/testing/half_area_048/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/half_area_048/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)
    if not os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
        print('Folder and files dont exist, move to next hyperparam setup', log_directory)
        exit()
    else:
        print('Hyperparameter tuning already done and prediction files are there now running validation')

'''full tile'''
if area == 'full':
    lon_bounds = (0, 15) #1800x1800
    lat_bounds = (45, 60)#1800x1800
    log_directory = 'results/testing/full_area_048/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/full_area_048/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)
    if not os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
        print('Folder and files dont exist, move to next hyperparam setup', log_directory)
        exit()
    else:
        print('Hyperparameter tuning already done and prediction files are there now running validation')


testing_path = 'data/testing_lat_{}_{}_lon_{}_{}'.format(lat_bounds[0], lat_bounds[1], lon_bounds[0], lon_bounds[1])

'''load model and data'''
print('load model and data')
if model_type == 'UNet2':
    model_reload = models.UNet2(input_channels=21, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
    
if model_type == 'UNet6':   
    model_reload = models.UNet6(input_channels=21, output_channels=1)
    model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))

X_val = np.load(testing_path + '/X_val.npy')
y_val = np.load(testing_path + '/y_val.npy')
mask_val = np.load(testing_path + '/mask_val.npy')

validation_loader = data.DataLoader(data.CustomDataset(X_val, y_val, mask_val), batch_size=batch_size, shuffle=False)

'''run validation'''
print('run validation')
val_score = train.test_model(model_reload, validation_loader)
print('Validation Score: ', val_score)
np.save(os.path.join(log_directory, 'val_score.npy'), val_score)


