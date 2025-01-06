#CNN_test
import os
import sys 
import random
import torch
import numpy as np
import data
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

# if torch.cuda.is_available():
#     print("CUDA is available! Using GPU.")
# else:
#     print("CUDA is not available. Using CPU.")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('device:', device)

args = sys.argv
# args = ['X' , 0.1, 5, 0.0001, 2, '180', 'UNet6'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type
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

'''define sample path, case area, number of epochs, learning rate and batch size'''
cnn_sample_path ='data/tile048_random_sampling_%s' %area
testing_path = 'data/testing_random_sampling_%s'%area

if not os.path.exists(testing_path):
    os.makedirs(testing_path)   

'''create log directory for tensorboard logs'''
log_directory = 'results/testing/random_sampling_%s/cpu_runs/%s_%s_%s_%s_%s' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
log_dir_fig = 'results/testing/random_sampling_%s/cpu_runs/%s_%s_%s_%s_%s/plots' %(area, model_type, epochs, learning_rate, batch_size, samplesize)

if not os.path.exists(log_directory):
    os.makedirs(log_directory) 
if not os.path.exists(log_dir_fig):
    os.makedirs(log_dir_fig)

'check if testing prediction files are already there and if so stop the process'
if os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
    print('Hyperparameter tuning already done and prediction files are there')
    exit()
else:
    print('Hyperparameter tuning not done yet - start process')

'''check if data is already prepared and if not prepare it'''
if len(os.listdir(testing_path)) < 26:
    print('prepare data testing folder')
    train_loader, validation_loader, test_loader = data.RS_dataprep(cnn_sample_path, testing_path, samplesize, batch_size)
else:
    print('data testing folder already prepared - reloading data')
    train_loader, validation_loader, test_loader = data.RS_reload_data(testing_path, samplesize, batch_size)


print('Hyperparameter tuning definition and start')
print('training model: %s, learning rate: %s, epochs: %s, batch size: %s' %(model_type, learning_rate, epochs, batch_size))
models.RS_cpu_hyperparam_tuning_training(train_loader, test_loader, validation_loader, model_type, learning_rate, epochs, log_directory, patience)
print('prediction model: %s, learning rate: %s, epochs: %s, batch size: %s' %(model_type, learning_rate, epochs, batch_size))
models.RS_cpu_hyperparam_tuning_prediction(validation_loader, model_type, log_directory, testing_path)






