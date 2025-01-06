import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import data


testSize = 0.4
trainSize = 0.3 #validation size is 1-testSize-trainSize

# folder1 = glob.glob('../../results/testing/small_area_048/360/UNet6*')
folders = glob.glob('../../results/testing/larger_area_048/*/UNet6*')
folders = folder1 + folder2
plotting_folder = '../../results/testing/larger_area_048/plots'

''' plots with rmse vs learning rate showing the last epoch rmse for each selected model'''
#select folders with UNet2 only
folders_unet2= [f for f in folders if 'UNet2' in f]
# sort folders_unet2 based on epochs
folders_unet2 = sorted(folders_unet2, key=lambda x: int(x.split('_')[6]))
colors2 = plt.cm.autumn(np.linspace(0, 1, len(folders_unet2)))

#select folders for UNet6 only
folders_unet6 = [f for f in folders if 'UNet6' in f]
# sort folders_unet6 based on epochs
folders_unet6 = sorted(folders_unet6, key=lambda x: int(x.split('_')[5]))
colors6 = plt.cm.winter(np.linspace(0, 1, len(folders_unet6)))

legend_elements = []
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15, 10), sharey=True)

# for fol, col in zip(folders_unet2[:], colors2[:]):
#     # get folder name which is the hyperparameter information in last few characters
#     fol_name = fol.split('/')[-1]
#     model = fol_name.split('_')[0]
#     epochs = fol_name.split('_')[1]
#     learnr = fol_name.split('_')[2]
#     batch = fol_name.split('_')[3]

#     event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]

# # Initialize lists to store the data
#     train_loss = []
#     test_loss = []
#     stepstr = []
#     stepste = []
#     # Iterate through all event files and extract data
#     event_acc = EventAccumulator(event_files[0])
#     event_acc.Reload()
#     # Extract scalars
#     loss_train = event_acc.Scalars('Loss/train_epoch')
#     loss_test = event_acc.Scalars('Loss/validation_epoch') #testing loss is saved as validation loss because of previous naming mistake
#     # Append to the lists
#     for i in range(len(loss_train)):
#         stepstr.append(loss_train[i].step)
#         train_loss.append(loss_train[i].value)
              
#     for i in range(len(loss_test)):
#         stepste.append(loss_test[i].step)
#         test_loss.append(loss_test[i].value)

#     try:
#         val_loss = np.load('%s/val_score.npy' %fol)
#     except:
#         print('no validation loss available for %s' %fol_name)
#         continue
    

#     ax1.plot(learnr, train_loss[-1], 'o',  color=col)
#     ax4.plot(learnr, test_loss[-1], 'D', color=col)  
#     ax7.plot(learnr, val_loss, 'x', color=col)    
#     ax7.set_xlabel('learning rates') 
#     ax1.set_ylabel('rmse model training')
#     ax4.set_ylabel('rmse model testing')
#     ax7.set_ylabel('rmse independent validation')
  
#     ax2.plot(epochs, train_loss[-1], 'o', color=col)
#     ax5.plot(epochs, test_loss[-1], 'D', color=col)
#     ax8.plot(epochs, val_loss, 'x', color=col)    
#     ax8.set_xlabel('epochs') 

#     ax3.plot(batch, train_loss[-1], 'o', color=col)
#     ax6.plot(batch, test_loss[-1], 'D', color=col)
#     ax9.plot(batch, val_loss, 'x', color=col)     
#     ax9.set_xlabel('batch size')


#     #set background color for each subplot
#     ax1.set_facecolor('lightgrey')
#     ax2.set_facecolor('lightgrey')
#     ax3.set_facecolor('lightgrey')
#     ax4.set_facecolor('lightgrey')
#     ax5.set_facecolor('lightgrey')
#     ax6.set_facecolor('lightgrey')
#     ax7.set_facecolor('lightgrey')
#     ax8.set_facecolor('lightgrey')
#     ax9.set_facecolor('lightgrey')

#     # Collect each unique label and color for the legend
#     legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s'%(model, epochs, batch, learnr)))

for fol, col in zip(folders_unet6[:], colors6[:]):
    # get folder name which is the hyperparameter information in last few characters
    fol_name = fol.split('/')[-1]
    model = fol_name.split('_')[0]
    epochs = fol_name.split('_')[1]
    learnr = fol_name.split('_')[2]
    batch = fol_name.split('_')[3]

    event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]
    # Initialize lists to store the data
    train_loss = []
    test_loss = []
    stepstr = []
    stepste = []
    # Iterate through all event files and extract data
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()
    # Extract scalars
    loss_train = event_acc.Scalars('Loss/train_epoch')
    loss_test = event_acc.Scalars('Loss/validation_epoch') #testing loss is saved as validation loss because of previous naming mistake
    # Append to the lists
    for i in range(len(loss_train)):
        stepstr.append(loss_train[i].step)
        train_loss.append(loss_train[i].value)
              
    for i in range(len(loss_test)):
        stepste.append(loss_test[i].step)
        test_loss.append(loss_test[i].value)

    try:
        val_loss = np.load('%s/val_score.npy' %fol)
    except:
        print('no validation loss available for %s' %fol_name)
        continue
    

    ax1.plot(learnr, train_loss[-1], 'o',  color=col)
    ax4.plot(learnr, test_loss[-1], 'D', color=col)  
    ax7.plot(learnr, val_loss, 'x', color=col)    
    ax7.set_xlabel('learning rates') 
    ax1.set_ylabel('rmse model training')
    ax4.set_ylabel('rmse model testing')
    ax7.set_ylabel('rmse independent validation')
  
    ax2.plot(epochs, train_loss[-1], 'o', color=col)
    ax5.plot(epochs, test_loss[-1], 'D', color=col)
    ax8.plot(epochs, val_loss, 'x', color=col)    
    ax8.set_xlabel('epochs') 

    ax3.plot(batch, train_loss[-1], 'o', color=col)
    ax6.plot(batch, test_loss[-1], 'D', color=col)
    ax9.plot(batch, val_loss, 'x', color=col)     
    ax9.set_xlabel('batch size')

    #set background color for each subplot
    ax1.set_facecolor('lightgrey')
    ax2.set_facecolor('lightgrey')
    ax3.set_facecolor('lightgrey')
    ax4.set_facecolor('lightgrey')
    ax5.set_facecolor('lightgrey')
    ax6.set_facecolor('lightgrey')
    ax7.set_facecolor('lightgrey')
    ax8.set_facecolor('lightgrey')
    ax9.set_facecolor('lightgrey')

    ax1.set_ylim(0, 1)
    # Collect each unique label and color for the legend
    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s'%(model, epochs, batch, learnr)))
# Add a single legend for all subplots outside the figure
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=5)
plt.tight_layout()
fig.savefig('%s/overview_hyperparam_testing_train_val_test_new.png' %plotting_folder, bbox_inches='tight')




'''spatial plots for different hyperparameter runs'''
# mask = np.load('../../data/testing_lat_47_50_lon_7_10/mask_test.npy') #360 old wrong
# mask = np.load('../../data/testing_lat_47_50_lon_7_10/mask_val.npy') #360
# target = np.load('../../data/testing_lat_47_50_lon_7_10/y.npy')#360

# mask = np.load('../../data/testing_lat_45_50_lon_5_10/mask_test.npy') #600
# mask = np.load('../../data/testing_lat_45_50_lon_5_10/mask_val.npy') #600
# target = np.load('../../data/testing_lat_45_50_lon_5_10/y.npy')#600

# mask = np.load('../../data/testing_lat_45_51_lon_5_11/mask_test.npy') #720
mask = np.load('../../data/testing_lat_45_51_lon_5_11/mask_val.npy') #720
target = np.load('../../data/testing_lat_45_51_lon_5_11/y.npy')#720

mask = np.load('../../data/testing_lat_45_52.5_lon_5_12.5/mask_val.npy') #900
target = np.load('../../data/testing_lat_45_52.5_lon_5_12.5/y.npy')#900

mask_test_na = np.where(mask==0, np.nan, 1)

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

# folders =  [f for f in folders if 'UNet6' in f]
for fol in folders[:]:
    # print(fol)
    try:
        y_pred_denorm = np.load('%s/y_pred_denorm_new.npy' %fol)
    except:
        print('no prediction file available for %s' %fol)
        continue
    for i in range(y_pred_denorm.shape[0])[:3]:
        # print(fol, i, y_pred_denorm.shape[0])
        print(fol, y_pred_denorm.shape, target_val.shape, mask_test_na.shape)
        if y_pred_denorm.shape[2] != target_val.shape[2]:
            print('skipping')
            continue

        vminR = np.percentile(y_pred_denorm[i, 0, :, :], 5)
        vmaxR = np.percentile(y_pred_denorm[i, 0, :, :], 95)
        vminT = np.percentile(target_val[i, 0, :, :], 5)
        vmaxT = np.percentile(target_val[i, 0, :, :], 95)
        vmax = np.max([vmaxR, vmaxT])
        vmin = np.min([vminR, vminT])

        lim = np.max([np.abs(vmax), np.abs(vmin)])

        plt.figure(figsize=(40, 10))
        plt.subplot(1, 5, 1)
        plt.imshow(target_val[i, 0, :, :]*mask_test_na[0,0,:,:], cmap='RdBu', vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Actual delta (colorbar 5-95 percentile)')
        plt.tight_layout()

        plt.subplot(1, 5, 2)
        plt.imshow(y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:], cmap='RdBu',vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Predicted delta (colorbar 5-95 percentile)')

        vmin = min([np.nanmin(target_val[i, 0, :, :]*mask_test_na[0,0,:,:]),np.nanmin(y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:])])
        vmax = max([np.nanmax(target_val[i, 0, :, :]*mask_test_na[0,0,:,:]),np.nanmax(y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:])])
        plt.subplot(1, 5, 3)
        plt.scatter((target_val[i,0, :, :]*mask_test_na[0,0,:,:]).flatten(), (y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:]).flatten(),alpha=0.2, facecolors='none', edgecolors='r')
        plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)
        plt.ylabel('Predicted delta') 
        plt.xlabel('Actual delta')

        plt.subplot(1, 5, 4)
        diff = (target_val[i, 0, :, :]*mask_test_na[0,0,:,:]) - (y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:]) #difference between wtd and calculated wtd
        vmax = np.nanmax(np.percentile(diff,95))
        vmin = np.nanmin(np.percentile(diff,5))
        lim = np.max([np.abs(vmax), np.abs(vmin)])
        plt.imshow(diff, cmap='RdBu', vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Difference target-predicted (colorbar 5-95 percentile)')

        plt.subplot(1, 5, 5)
        # Example maps
        map1 = target_val[i, 0, :, :]*mask_test_na[0,0,:,:]
        map2 = y_pred_denorm[i, 0, :, :]*mask_test_na[0,0,:,:]
        relative_error = (map1 - map2) / map1
        vmax = np.nanmax(np.percentile(relative_error,95))
        vmin = np.nanmin(np.percentile(relative_error,5))
        lim = np.max([np.abs(vmax), np.abs(vmin)])
        plt.imshow(relative_error, cmap='RdBu', vmin=-lim, vmax=lim)
        plt.title('Relative error (colorbar 5-95 percentile)')
        plt.colorbar(shrink=0.8)

        plt.savefig('%s/plots/plot_timesplit_%s_new.png' %(fol, i))
        plt.close()







