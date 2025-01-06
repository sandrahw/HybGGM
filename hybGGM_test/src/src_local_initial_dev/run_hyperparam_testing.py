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

args = sys.argv
# args = [0.4, 0.3, 10, 0.0001, 1, 'ConvExample'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type

'''specific small data case - define general path, data length, case area, number of epochs, learning rate and batch size'''
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)
# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)

#TODO raw tile data on ejit on scratch and in .map file format
# create data testing directory based on cutout lat lon if not there yet 
print('current path', os.getcwd())


testing_path = '../../data/testing_lat_{}_{}_lon_{}_{}'.format(lat_bounds[0], lat_bounds[1], lon_bounds[0], lon_bounds[1])
if not os.path.exists(testing_path):
    os.makedirs(testing_path)
else: 
    print('data testing folder already prepared')
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
'''create log directory for tensorboard logs'''
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

X_norm = np.load(testing_path + '/X_norm_arr.npy')
y_norm = np.load(testing_path + '/y_norm_arr.npy')

print('testSize:', testSize, 'trainSize:', trainSize, 'epochs:', epochs, 'learning_rate:', learning_rate, 'batch_size:', batch_size, 'model_type:', model_type)
models.hyperparam_tuning_training(X_norm, y_norm, testSize, trainSize, epochs, learning_rate, batch_size, model_type, testing_path)
models.hyperparam_tuning_prediction(epochs, learning_rate, batch_size, model_type, testing_path)

