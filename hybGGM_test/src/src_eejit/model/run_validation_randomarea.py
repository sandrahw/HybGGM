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
args = ['X' ,1, 10, 0.0001, 256, 180, 'UNet2'] #testSize, trainSize, epochs, learning_rate, batch_size, model_type
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
sub_batch_size = 1
if batch_size<sub_batch_size:
    sub_batch_size = batch_size
    


'''define sample path, case area, number of epochs, learning rate and batch size'''
# cnn_sample_path ='data/tile048_random_sampling_%s' %area
testing_path = '../../data/testing_random_sampling_%s'%area

'''create log directory for tensorboard logs'''
log_directory = '../../results/testing/random_sampling_%s/%s_%s_%s_%s_%s' %(area, model_type, epochs, learning_rate, batch_size, samplesize)
log_dir_fig = '../../results/testing/random_sampling_%s/%s_%s_%s_%s_%s/plots' %(area, model_type, epochs, learning_rate, batch_size, samplesize)

'check if testing prediction files are already there and if so stop the process'
if os.path.exists(log_directory + '/y_pred_denorm_new.npy'):
    print('Hyperparameter tuning already done and prediction files are there')
    exit()
else:
    print('Hyperparameter tuning not done yet - start process')

if os.path.exists(log_directory + '/best_model.pth'):
    print('best model available - continue with validation')
    print('load model and data')
    if model_type == 'UNet2':
        model_reload = models.UNet2(input_channels=21, output_channels=1)
        model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
        
    if model_type == 'UNet6':   
        model_reload = models.UNet6(input_channels=21, output_channels=1)
        model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
else:
    print('best model not available - stop process')
    exit()


print('load loaders for validation')
train_loader, validation_loader, test_loader = data.RS_reload_data(testing_path, samplesize, batch_size)

'''run validation'''
print('run validation')
val_score = train.RS_cpu_val_model(model_reload, validation_loader)
print('Validation Score: ', val_score)
np.save(os.path.join(log_directory, 'val_score.npy'), val_score)

print('run prediction')
train.RS_cpu_hyperparam_tuning_prediction(validation_loader, model_type, log_directory, testing_path)


