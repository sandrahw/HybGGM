import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import data

testSize = 0.4
trainSize = 0.3 #validation size is 1-testSize-trainSize
folders = glob.glob(r'..\results\testing\*')
num_folders = len(folders)  
colors = plt.cm.viridis(np.linspace(0, 1, num_folders)) 

'''overarching folder figure losses'''
fig, ax = plt.subplots()
for fol,col in zip(folders[:-1], colors[:-1]):
    # print(fol)
    # get folder name which is the hyperparameter information in last few characters
    fol_name = fol.split('\\')[-1]
    print(fol_name)
   
    event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]

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

    ax.plot(stepstr, train_loss, label=fol_name, color=col, linestyle='solid')
    ax.plot(stepsva, val_loss, color=col, linestyle='dashed')
    ax.scatter(stepste, test_loss, color=col, marker='x')

ax.plot(np.NaN, np.NaN, label='model train', color='k', linestyle='solid')
ax.plot(np.NaN, np.NaN, label='model val', color='k', linestyle='dashed')
ax.scatter(np.NaN, np.NaN, label='independent test', color='k', marker='x')

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))

plt.title('Training, Validation and Test Loss')
plt.xlabel('epochs')
plt.ylabel('loss (rmse)')

plt.tight_layout()
plt.savefig(r'..\results\training_loss.png')

'''spatial plots for different hyperparameter runs'''
mask = np.load(r'..\data\testing\mask_test.npy')
mask_test_na = np.where(mask==0, np.nan, 1)
target = np.load(r'..\data\testing\y.npy')
target_train, target_val_test = data.train_test_split(target, test_size=1-trainSize, random_state=10)
target_val, target_test = data.train_test_split(target_val_test, test_size=1-testSize, random_state=10) 
def calculate_cellwise_correlation(map1, map2):
            """
            Calculate the Pearson correlation coefficient for each cell across two maps.
            
            Parameters:
            map1, map2: Two 2D numpy arrays of the same shape
            
            Returns:
            A 2D array where each cell contains the Pearson correlation coefficient
            between the corresponding cells in map1 and map2.
            """
            if map1.shape != map2.shape:
                raise ValueError("Both maps must have the same shape.")
            
            # Calculate the correlation between corresponding cells across the two maps
            mean_map1 = np.mean(map1)
            mean_map2 = np.mean(map2)

            std_map1 = np.std(map1)
            std_map2 = np.std(map2)

            # Element-wise correlation
            correlation_map = (map1 - mean_map1) * (map2 - mean_map2) / (std_map1 * std_map2)
            
            return correlation_map

for fol in folders[:]:
    y_pred_denorm = np.load(r'%s\y_pred_denorm.npy' %fol)
    for i in range(y_pred_denorm.shape[0])[:5]:
        print(i, range(y_pred_denorm.shape[0]))

        vminR = np.percentile(y_pred_denorm[i, 0, :, :], 5)
        vmaxR = np.percentile(y_pred_denorm[i, 0, :, :], 95)
        vminT = np.percentile(target_test[i, 0, :, :], 5)
        vmaxT = np.percentile(target_test[i, 0, :, :], 95)
        vmax = np.max([vmaxR, vmaxT])
        vmin = np.min([vminR, vminT])

        lim = np.max([np.abs(vmax), np.abs(vmin)])

        plt.figure(figsize=(40, 10))
        plt.subplot(1, 5, 1)
        plt.imshow(target_test[i, 0, :, :]*mask_test_na[0,0,:,:], cmap='RdBu', vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Actual delta (colorbar 5-95 percentile)')
        plt.tight_layout()

        plt.subplot(1, 5, 2)
        plt.imshow(y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:], cmap='RdBu',vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Predicted delta (colorbar 5-95 percentile)')

        vmin = min([np.nanmin(target_test[i, 0, :, :]*mask_test_na[0,0,:,:]),np.nanmin(y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:])])
        vmax = max([np.nanmax(target_test[i, 0, :, :]*mask_test_na[0,0,:,:]),np.nanmax(y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:])])
        plt.subplot(1, 5, 3)
        plt.scatter((target_test[i,0, :, :]*mask_test_na[0,0,:,:]).flatten(), (y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:]).flatten(),alpha=0.2, facecolors='none', edgecolors='r')
        plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)
        plt.ylabel('Predicted delta') 
        plt.xlabel('Actual delta')

        plt.subplot(1, 5, 4)
        diff = (target_test[i, 0, :, :]*mask_test_na[0,0,:,:]) - (y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:]) #difference between wtd and calculated wtd
        vmax = np.nanmax(np.percentile(diff,95))
        vmin = np.nanmin(np.percentile(diff,5))
        lim = np.max([np.abs(vmax), np.abs(vmin)])
        plt.imshow(diff, cmap='RdBu', vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Difference target-predicted (colorbar 5-95 percentile)')

        plt.subplot(1, 5, 5)
        # Example maps
        map1 = target_test[i, 0, :, :]*mask_test_na[0,0,:,:]
        map2 = y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:]
        relative_error = (map1 - map2) / map1
        vmax = np.nanmax(np.percentile(relative_error,95))
        vmin = np.nanmin(np.percentile(relative_error,5))
        lim = np.max([np.abs(vmax), np.abs(vmin)])
        plt.imshow(relative_error, cmap='RdBu', vmin=-lim, vmax=lim)
        plt.title('Relative error (colorbar 5-95 percentile)')
        plt.colorbar(shrink=0.8)
 
        plt.savefig(r'%s\plots\plot_timesplit_%s.png' %(fol, i))







