import random
import numpy as np
import torch
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

'''define general path, data length, case area, number of epochs, learning rate and batch size'''
data_length = 72 # number of months in current dataset (needed for dataprep function to extend the static parameters)

# define the lat and lon bounds for the test region
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)

print('Data loading')
mask_01 = data.mask_data_0_1(r'..\data\target\wtd.nc', lat_bounds, lon_bounds, r'..\data\testing')
X, y = data.input_data_load(r'..\data\input', r'..\data\target\wtd.nc', lat_bounds, lon_bounds, data_length, r'..\data\testing', mask_01)

print('Normalization')
X_norm, y_norm = data.normalize(X, y, r'..\data\testing')


print('Hyperparameter tuning definition and start')
'''create log directory for tensorboard logs'''
testSize = 0.4
trainSize = 0.3 #validation size is 1-testSize-trainSize
epochs = [10]
learning_rate = [0.001]
batch_size = [1]
model_type = ['UNet2', 'UNet4', 'UNet6']#, 'CNNLSTM']

models.hyperparam_tuning_training(X_norm, y_norm, testSize, trainSize, epochs, learning_rate, batch_size, model_type)

models.hyperparam_tuning_prediction(epochs, learning_rate, batch_size, model_type)
