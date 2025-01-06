import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import data

testSize = 0.4
trainSize = 0.3 #validation size is 1-testSize-trainSize
folders = glob.glob('../results/testing/*')
num_folders = len(folders)  
colors = plt.cm.viridis(np.linspace(0, 1, num_folders)) 

fig, ax = plt.subplots(figsize=(10, 6))
for fol,col in zip(folders[:], colors[:]):
    # get folder name which is the hyperparameter information in last few characters
    fol_name = fol.split('/')[-1]
    # print(fol_name)
    
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
    # print(fol_name,  val_loss)

    ax.plot(stepstr, train_loss, label=fol_name, color=col, linestyle='solid')
    ax.plot(stepsva, val_loss, color=col, linestyle='dashed')
    ax.scatter(stepste, test_loss, color=col, marker='x')

ax.plot(np.NaN, np.NaN, label='model train', color='k', linestyle='solid')
ax.plot(np.NaN, np.NaN, label='model val', color='k', linestyle='dashed')
ax.scatter(np.NaN, np.NaN, label='independent test', color='k', marker='x')

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
# legend below figure with 3 columns
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
ax.set_ylim([0, 2])
plt.title('Training, Validation and Test Loss')
plt.xlabel('epochs')
plt.ylabel('loss (rmse)')

plt.tight_layout()
plt.savefig('../results/training_loss.png')
plt.close()

''' create folder selection based on models, epochs, learning reates and batch sizes'''
selective_arg = ['UNet2', 'UNet6', '_10_', '_20_', '_50_', '_100_', '_200_', '0.0001', ' 0.0005', '0.005', '0.001', '1_1', '_4', '_10']
for sela in selective_arg[:]:
    folders_sel = [f for f in folders if sela in f]
    #drop ConvExample in case in folder_sel
    folders_sel = [f for f in folders_sel if 'ConvExample' not in f]   
    
    # colors = plt.cm.viridis(np.linspace(0, 1, len(folders_sel))) 
    # folders_sel_len = len(folders_sel)
    # print(folders_sel)
    folders_sel_ext = [f for f in folders_sel if not '0.001' in f]
    colors = plt.cm.viridis(np.linspace(0, 1, len(folders_sel_ext))) 
    folders_sel_len = len(folders_sel_ext)
    '''overarching folder figure losses'''
    fig, ax = plt.subplots(figsize=(10, 6))
    # for fol,col in zip(folders_sel[:], colors[:]):
    for fol,col in zip(folders_sel_ext[:], colors[:]):
        # get folder name which is the hyperparameter information in last few characters
        fol_name = fol.split('/')[-1]
        # print(fol_name)
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
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
    # legend below figure with 3 columns
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.title('Training, Validation and Test Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss (rmse)')

    plt.tight_layout()
    # plt.savefig('../results/training_loss_%s.png' %sela)
    plt.savefig('../results/training_loss_%s_no0_001lr.png' %sela)
    plt.close()


''' plots with rmse vs learning rate showing the last epoch rmse for each selected model'''
# select folders with UNet2 only
folders_unet2= [f for f in folders if 'UNet2' in f]
colors2 = plt.cm.autumn(np.linspace(0, 1, len(folders_unet2)))

#select folders for UNet6 only
folders_unet6 = [f for f in folders if 'UNet6' in f]
colors6 = plt.cm.winter(np.linspace(0, 1, len(folders_unet6)))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
for fol, col in zip(folders_unet2[:], colors2[:]):
    # get folder name which is the hyperparameter information in last few characters
    fol_name = fol.split('/')[-1]
    model = fol_name.split('_')[0]
    epochs = fol_name.split('_')[1]
    learnr = fol_name.split('_')[2]
    batch = fol_name.split('_')[3]

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

    # ax1.plot(learnr, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax1.plot(learnr, test_loss[-1], 'x', color=col)    
    # ax1.legend(loc='upper right')
    ax1.set_xlabel('learning rates') 
    ax1.set_ylabel('rmse')
  
    # ax2.plot(epochs, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax2.plot(epochs, test_loss[-1], 'x', label='%s_%s_%s'%(model, epochs, batch), color=col)    
    # ax2.legend(loc='upper right')
    ax2.set_xlabel('epochs') 

    # ax3.plot(batch, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax3.plot(batch, test_loss[-1], 'x', color=col)     
    # ax3.legend(loc='upper right')
    ax3.set_xlabel('batch size')
  
    # ax4.plot(model, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax4.plot(model, test_loss[-1], 'x', color=col)     
    # ax4.legend(loc='upper right')
    ax4.set_xlabel('model')

    #create legend which is outside of figure spanning over the length of the total figure
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    # ax1.set_ylim([0, 1])

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

    # ax1.plot(learnr, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax1.plot(learnr, test_loss[-1], 'x', color=col)    
    # ax1.legend(loc='upper right')
    ax1.set_xlabel('learning rates') 
    ax1.set_ylabel('rmse')
  
    # ax2.plot(epochs, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax2.plot(epochs, test_loss[-1], 'x', label='%s_%s_%s'%(model, epochs, batch),color=col)    
    # ax2.legend(loc='upper right')
    ax2.set_xlabel('epochs') 

    # ax3.plot(batch, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax3.plot(batch, test_loss[-1], 'x', color=col)     
    # ax3.legend(loc='upper right')
    ax3.set_xlabel('batch size')
  
    # ax4.plot(model, train_loss[-1], 'o', label='%s_%s_%s'%(model, epochs, batch), color=col)
    ax4.plot(model, test_loss[-1], 'x', color=col)     
    # ax4.legend(loc='upper right')
    ax4.set_xlabel('model')

    #create legend which is outside of figure spanning over the length of the total figure
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    ax1.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('../results/overview_hyperparam_testing_testscores.png')


'''spatial plots for different hyperparameter runs'''
mask = np.load('../data/testing_lat_47_50_lon_7_10/mask_test.npy')
mask_test_na = np.where(mask==0, np.nan, 1)
target = np.load('../data/testing_lat_47_50_lon_7_10/y.npy')
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

# select folders which include batch size 10, epochs 50 
folders_sel = [f for f in folders if '_10' in f]
folders_sel = [f for f in folders_sel if '_50_' in f]   

for fol in folders_sel[:]:
    print(fol)
    y_pred_denorm = np.load('%s/y_pred_denorm.npy' %fol)
    for i in range(y_pred_denorm.shape[0])[:5]:
        print(fol, i, y_pred_denorm.shape[0])

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
 
        plt.savefig('%s/plots/plot_timesplit_%s.png' %(fol, i))







