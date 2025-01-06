import random
import numpy as np
import torch
import data
import models
import sys
import os

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
# args = ['X' ,0.4, 0.3, 50, 0.0001, 16, 'UNet6', 'full'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type
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

'''specific small data case - define general path, data length, case area, number of epochs, learning rate and batch size'''
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
# define the lat and lon bounds for the test region

'''create log directory for tensorboard logs'''
'''usual small area'''
if area == 'small':
    lon_bounds = (7, 10) #CH bounds(5,10)#360x360
    lat_bounds = (47, 50)#CH bounds(45,50)#360x360
    log_directory = 'results/testing/small_area_048_initialtesting/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/small_area_048_initialtesting/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)
'''larger area (one quarter of a tile)'''
if area == 'large':
    # lon_bounds = (5, 11) #CH bounds(5,10)#720x720
    # lat_bounds = (45, 51)#CH bounds(45,50)#720x720
    lon_bounds = (5, 11) #CH bounds(5,10)
    lat_bounds = (45, 51)#CH bounds(45,50)
    log_directory = 'results/testing/larger_area_048/600/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/larger_area_048/600/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)

'''half a tile'''
if area == 'half':
    lon_bounds = (5, 12.5) 
    lat_bounds = (45, 52.5)
    log_directory = 'results/testing/half_area_048/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/half_area_048/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)

'''full tile'''
if area == 'full':
    lon_bounds = (0, 15) #1800x1800
    lat_bounds = (45, 60)#1800x1800
    log_directory = 'results/testing/full_area_048/%s_%s_%s_%s' %(model_type, epochs, learning_rate, batch_size)
    log_dir_fig = 'results/testing/full_area_048/%s_%s_%s_%s/plots' %(model_type, epochs, learning_rate, batch_size)

if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)


testing_path = 'data/testing_lat_{}_{}_lon_{}_{}'.format(lat_bounds[0], lat_bounds[1], lon_bounds[0], lon_bounds[1])
if not os.path.exists(testing_path):
    os.makedirs(testing_path)
else: 
    print('data folder already prepared')
#check if folder is empty or if temp files that are otherwise created during the process are in there
if len(os.listdir(testing_path)) < 9:
    print('data testing folder is empty')
    print('Data loading')
    mask_01 = data.mask_data_0_1('data/target/wtd.nc', lat_bounds, lon_bounds, testing_path)
    X, y, inFiles = data.input_data_load('data/input', 'data/target/wtd.nc', lat_bounds, lon_bounds, data_length, testing_path, mask_01)

    print('Normalization')
    X_norm, y_norm = data.normalize(X, y, testing_path)
else:
    print('data testing folder already prepared')


print('Hyperparameter tuning definition and start')
'check if testing prediction files are already there and if so stop the process'
if not os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
    print('corrected prediction files not there yet')
    if os.path.exists(log_directory + '/best_model.pth'):
        print('best model already there')
        print('rerun prediction')
        print('Model type: %s, Epochs: %s, Learning rate: %s, Batch size: %s' %(model_type, epochs, learning_rate, batch_size))
        models.hyperparam_tuning_prediction( batch_size, model_type, testing_path, log_directory)
        #print done and continue with next step
        print('prediction done')
    else:
        print('no best model found')
        print(log_directory + '/best_model.pth')
        X_norm = np.load(testing_path + '/X_norm_arr.npy')
        y_norm = np.load(testing_path + '/y_norm_arr.npy')

        print('testSize:', testSize, 'trainSize:', trainSize, 'epochs:', epochs, 'learning_rate:', learning_rate, 'batch_size:', batch_size, 'model_type:', model_type)
        models.hyperparam_tuning_training(X_norm, y_norm, testSize, trainSize, epochs, learning_rate, batch_size, model_type, testing_path, area, log_directory)
        models.hyperparam_tuning_prediction(batch_size, model_type, testing_path, log_directory)



# X_norm = np.load(testing_path + '/X_norm_arr.npy')
# y_norm = np.load(testing_path + '/y_norm_arr.npy')

# print('testSize:', testSize, 'trainSize:', trainSize, 'epochs:', epochs, 'learning_rate:', learning_rate, 'batch_size:', batch_size, 'model_type:', model_type)
# models.hyperparam_tuning_training(X_norm, y_norm, testSize, trainSize, epochs, learning_rate, batch_size, model_type, testing_path, area, log_directory)
# models.hyperparam_tuning_prediction(batch_size, model_type, testing_path, log_directory)

