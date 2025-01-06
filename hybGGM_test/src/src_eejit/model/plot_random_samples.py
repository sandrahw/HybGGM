import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import data


folders_180 = glob.glob('../../results/testing/random_sampling_180/UNet2*')
# folders_180cpu = glob.glob('../../results/testing/random_sampling_180/cpu_runs/UNet6*')
folders_360 = glob.glob('../../results/testing/random_sampling_360/UNet2*')
folders_600 = glob.glob('../../results/testing/random_sampling_600/UNet6*')

folders = folders_180# + folders_360 #+ folders_600
plotting_folder = '../../results/testing/random_sampling_180/plots'

''' plots with rmse vs learning rate showing the last epoch rmse for each selected model'''
# folders_180= [f for f in folders if '180' in f]
# only select folders with _2_, _4_, _8_ and _16_ _32_ _64_ _128  and _256_ in the name
# folders_180bs = [f for f in folders_180 if '_2_' in f or '_4_' in f or '_8_' in f or '_16_' in f or '_32_' in f or '_64_' in f or '_128_' in f or '_256_' in f]
# folders_180final = [f for f in folders_180bs if '0.0001' in f and '_50_' in f and '_1.0' in f]
#drop dubplicates
# folders_180 = list(set(folders_180final))
# only folders that have a y_pred_denorm.npy file
folders_180 = [f for f in folders_180 if os.path.exists('%s/y_pred_denorm_new.npy' %f)]
# sort the folders based on the batch size
folders_180 = sorted(folders_180, key=lambda x: int(x.split('_')[-2]))
colors180 = plt.cm.autumn(np.linspace(0, 1, len(folders_180)))


# folders_unet360 = [f for f in folders if '360' in f]
folders_360 = [f for f in folders_360 if os.path.exists('%s/y_pred_denorm_new.npy' %f)]
# sort the folders based on the batch size
folders_360 = sorted(folders_360, key=lambda x: int(x.split('_')[-2]))
colors360 = plt.cm.winter(np.linspace(0, 1, len(folders_360)))

folders_unet600 = [f for f in folders if '600' in f]
colors600 = plt.cm.binary(np.linspace(0, 1, len(folders_600)))

legend_elements = []
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(15, 10), sharey=True)

for fol, col in zip(folders_180[:], colors180[:]):
    # get folder name which is the hyperparameter information in last few characters
    fol_name = fol.split('/')[-1]
    print(fol_name)
    model = fol_name.split('_')[0]
    epochs = fol_name.split('_')[1]
    learnr = fol_name.split('_')[2]
    batch = fol_name.split('_')[3]
    samplesize = fol_name.split('_')[4]
    cpu = fol.split('/')[5]
    if cpu == 'cpu_runs':
        cpu = 'cpu'
    else:   
        cpu = 'gpu'

    event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]
    if event_files == []:
        print('no event files available for %s' %fol)
        continue
# Initialize lists to store the data
    train_loss = []
    test_loss = []
    val_loss = []
    stepstr = []
    stepste = []
    stepsval = []
    # Iterate through all event files and extract data
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()
    # Extract scalars
    try:
        loss_train = event_acc.Scalars('Loss/train_epoch')
        loss_test = event_acc.Scalars('Loss/test_epoch') 
        loss_val = event_acc.Scalars('Loss/val_epoch')
    except:
        print('no loss data available for %s' %fol)
        continue
    # Append to the lists
    for i in range(len(loss_train)):
        stepstr.append(loss_train[i].step)
        train_loss.append(loss_train[i].value)
              
    for i in range(len(loss_test)):
        stepste.append(loss_test[i].step)
        test_loss.append(loss_test[i].value)
    
    for i in range(len(loss_val)):
        stepsval.append(loss_val[i].step)
        val_loss.append(loss_val[i].value)


    ax1.plot(learnr, train_loss[-1], 'o',  color=col)
    ax5.plot(learnr, test_loss[-1], 'D', color=col)  
    ax9.plot(learnr, val_loss, 'x', color=col)    
    ax9.set_xlabel('learning rates') 
    ax1.set_ylabel('rmse model training')
    ax5.set_ylabel('rmse model testing')
    ax9.set_ylabel('rmse independent validation')
  
    ax2.plot(stepstr[-1]+1, train_loss[-1], 'o', color=col)
    ax6.plot(stepste[-1]+1, test_loss[-1], 'D', color=col)
    ax10.plot(stepste[-1]+1, val_loss, 'x', color=col)    
    ax10.set_xlabel('epochs') 
    epochs = stepstr[-1]+1

    ax3.plot(batch, train_loss[-1], 'o', color=col)
    ax7.plot(batch, test_loss[-1], 'D', color=col)
    ax11.plot(batch, val_loss, 'x', color=col)     
    ax11.set_xlabel('batch size')

    ax4.plot(samplesize, train_loss[-1], 'o', color=col)
    ax8.plot(samplesize, test_loss[-1], 'D', color=col)
    ax12.plot(samplesize, val_loss, 'x', color=col)
    ax12.set_xlabel('samplesize')

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
    ax10.set_facecolor('lightgrey')
    ax11.set_facecolor('lightgrey')
    ax12.set_facecolor('lightgrey')

    ax1.set_ylim(0, 0.5)


    # Collect each unique label and color for the legend
    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s_%s_%s'%(model, epochs, batch, learnr, samplesize, cpu)))

# for fol, col in zip(folders_360[:], colors360[:]):
#      # get folder name which is the hyperparameter information in last few characters
#     fol_name = fol.split('/')[-1]
#     model = fol_name.split('_')[0]
#     epochs = fol_name.split('_')[1]
#     learnr = fol_name.split('_')[2]
#     batch = fol_name.split('_')[3]
#     samplesize = fol_name.split('_')[4]

#     event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]

# # Initialize lists to store the data
#     train_loss = []
#     test_loss = []
#     val_loss = []
#     stepstr = []
#     stepste = []
#     stepsval = []
#     # Iterate through all event files and extract data
#     event_acc = EventAccumulator(event_files[0])
#     event_acc.Reload()
#     # Extract scalars
#     try:
#         loss_train = event_acc.Scalars('Loss/train_epoch')
#         loss_test = event_acc.Scalars('Loss/test_epoch') 
#         loss_val = event_acc.Scalars('Loss/val_epoch')
#     except:
#         print('no loss data available for %s' %fol)
#         continue
#     # Append to the lists
#     for i in range(len(loss_train)):
#         stepstr.append(loss_train[i].step)
#         train_loss.append(loss_train[i].value)
              
#     for i in range(len(loss_test)):
#         stepste.append(loss_test[i].step)
#         test_loss.append(loss_test[i].value)
    
#     for i in range(len(loss_val)):
#         stepsval.append(loss_val[i].step)
#         val_loss.append(loss_val[i].value)


#     ax1.plot(learnr, train_loss[-1], 'o',  color=col)
#     ax5.plot(learnr, test_loss[-1], 'D', color=col)  
#     ax9.plot(learnr, val_loss, 'x', color=col)    
#     ax9.set_xlabel('learning rates') 
#     ax1.set_ylabel('rmse model training')
#     ax5.set_ylabel('rmse model testing')
#     ax9.set_ylabel('rmse independent validation')

#     ax2.plot(stepstr[-1]+1, train_loss[-1], 'o', color=col)
#     ax6.plot(stepste[-1]+1, test_loss[-1], 'D', color=col)
#     ax10.plot(stepste[-1]+1, val_loss, 'x', color=col)    
#     ax10.set_xlabel('epochs') 
#     epochs = stepstr[-1]+1

#     ax3.plot(batch, train_loss[-1], 'o', color=col)
#     ax7.plot(batch, test_loss[-1], 'D', color=col)
#     ax11.plot(batch, val_loss, 'x', color=col)     
#     ax11.set_xlabel('batch size')

#     ax4.plot(samplesize, train_loss[-1], 'o', color=col)
#     ax8.plot(samplesize, test_loss[-1], 'D', color=col)
#     ax12.plot(samplesize, val_loss, 'x', color=col)
#     ax12.set_xlabel('samplesize')

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
#     ax10.set_facecolor('lightgrey')
#     ax11.set_facecolor('lightgrey')
#     ax12.set_facecolor('lightgrey')

#     # Collect each unique label and color for the legend
#     legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s_%s'%(model, epochs, batch, learnr, samplesize)))

# for fol, col in zip(folders_600[:], colors600[:]):
#       # get folder name which is the hyperparameter information in last few characters
#     fol_name = fol.split('/')[-1]
#     model = fol_name.split('_')[0]
#     epochs = fol_name.split('_')[1]
#     learnr = fol_name.split('_')[2]
#     batch = fol_name.split('_')[3]
#     samplesize = fol_name.split('_')[4]

#     event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]

# # Initialize lists to store the data
#     train_loss = []
#     test_loss = []
#     val_loss = []
#     stepstr = []
#     stepste = []
#     stepsval = []
#     # Iterate through all event files and extract data
#     event_acc = EventAccumulator(event_files[0])
#     event_acc.Reload()
#     # Extract scalars
#     try:
#         loss_train = event_acc.Scalars('Loss/train_epoch')
#         loss_test = event_acc.Scalars('Loss/test_epoch') 
#         loss_val = event_acc.Scalars('Loss/val_epoch')
#     except:
#         print('no loss data available for %s' %fol)
#         continue
#     # Append to the lists
#     for i in range(len(loss_train)):
#         stepstr.append(loss_train[i].step)
#         train_loss.append(loss_train[i].value)
              
#     for i in range(len(loss_test)):
#         stepste.append(loss_test[i].step)
#         test_loss.append(loss_test[i].value)
    
#     for i in range(len(loss_val)):
#         stepsval.append(loss_val[i].step)
#         val_loss.append(loss_val[i].value)


#     ax1.plot(learnr, train_loss[-1], 'o',  color=col)
#     ax5.plot(learnr, test_loss[-1], 'D', color=col)  
#     ax9.plot(learnr, val_loss, 'x', color=col)    
#     ax9.set_xlabel('learning rates') 
#     ax1.set_ylabel('rmse model training')
#     ax5.set_ylabel('rmse model testing')
#     ax9.set_ylabel('rmse independent validation')
  
#     ax2.plot(epochs, train_loss[-1], 'o', color=col)
#     ax6.plot(epochs, test_loss[-1], 'D', color=col)
#     ax10.plot(epochs, val_loss, 'x', color=col)    
#     ax10.set_xlabel('epochs') 

#     ax3.plot(batch, train_loss[-1], 'o', color=col)
#     ax7.plot(batch, test_loss[-1], 'D', color=col)
#     ax11.plot(batch, val_loss, 'x', color=col)     
#     ax11.set_xlabel('batch size')

#     ax4.plot(samplesize, train_loss[-1], 'o', color=col)
#     ax8.plot(samplesize, test_loss[-1], 'D', color=col)
#     ax12.plot(samplesize, val_loss, 'x', color=col)
#     ax12.set_xlabel('samplesize')

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
#     ax10.set_facecolor('lightgrey')
#     ax11.set_facecolor('lightgrey')
#     ax12.set_facecolor('lightgrey')

#     ax1.set_ylim(0, 1)

#     # Collect each unique label and color for the legend
#     legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s_%s'%(model, epochs, batch, learnr, samplesize)))

# Add a single legend for all subplots outside the figure
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=5)
plt.tight_layout()
fig.savefig('%s/overview_hyperparam_testing_train_val_test_new_180_randomsampling_earlystopping_U6.png' %plotting_folder, bbox_inches='tight')

'''scatter plots for different hyperparameter runs - color coded by batch size'''
folders_180 = glob.glob('../../results/testing/random_sampling_180/UNet2*')
plotting_folder = '../../results/testing/random_sampling_180/plots'
folders_180= [f for f in folders_180 if '180' in f]
# only folders that have a y_pred_denorm.npy file
folders_180 = [f for f in folders_180 if os.path.exists('%s/y_pred_denorm_new.npy' %f)]
# sort the folders based on the batch size
folders_180 = sorted(folders_180, key=lambda x: int(x.split('_')[-2]))

# select folders per batch size
folders_180_2 = [f for f in folders_180 if '_2_' in f]
folders_180_4 = [f for f in folders_180 if '_4_' in f]
folders_180_8 = [f for f in folders_180 if '_8_' in f]
folders_180_16 = [f for f in folders_180 if '_16_' in f]
folders_180_32 = [f for f in folders_180 if '_32_' in f]
folders_180_64 = [f for f in folders_180 if '_64_' in f]
folders_180_128 = [f for f in folders_180 if '_128_' in f]
folders_180_256 = [f for f in folders_180 if '_256_' in f]

#for every selected folder a different colormap
colors180_2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(folders_180_2)))
colors180_4 = plt.cm.Greens(np.linspace(0.4, 0.9, len(folders_180_4)))
colors180_8 = plt.cm.Blues(np.linspace(0.4, 0.9, len(folders_180_8)))
colors180_16 = plt.cm.Purples(np.linspace(0.4, 0.9, len(folders_180_16)))
colors180_32 = plt.cm.Greys(np.linspace(0.4, 0.9, len(folders_180_32)))
colors180_64 = plt.cm.spring(np.linspace(0, 0.5, len(folders_180_64)))
colors180_128 = plt.cm.Oranges(np.linspace(0, 0.5, len(folders_180_128)))
colors180_256 = plt.cm.winter(np.linspace(0, 0.5, len(folders_180_256)))


foldersplot = [folders_180_2, folders_180_4, folders_180_8, folders_180_16, folders_180_32, folders_180_64, folders_180_128, folders_180_256]
colorsplot = [colors180_2, colors180_4, colors180_8, colors180_16, colors180_32, colors180_64, colors180_128, colors180_256]

legend_elements = []
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(15, 10), sharey=True)
for folders_sel, colors_sel in zip(foldersplot, colorsplot):
    for fol, col in zip(folders_sel[:], colors_sel[:]):
        # get folder name which is the hyperparameter information in last few characters
        fol_name = fol.split('/')[-1]
        print(fol_name)
        model = fol_name.split('_')[0]
        epochs = fol_name.split('_')[1]
        learnr = fol_name.split('_')[2]
        batch = fol_name.split('_')[3]
        samplesize = fol_name.split('_')[4]
        cpu = fol.split('/')[5]
        if cpu == 'cpu_runs':
            cpu = 'cpu'
        else:   
            cpu = 'gpu'

        event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]
        if event_files == []:
            print('no event files available for %s' %fol)
            continue
    # Initialize lists to store the data
        train_loss = []
        test_loss = []
        val_loss = []
        stepstr = []
        stepste = []
        stepsval = []
        # Iterate through all event files and extract data
        event_acc = EventAccumulator(event_files[0])
        event_acc.Reload()
        # Extract scalars
        try:
            loss_train = event_acc.Scalars('Loss/train_epoch')
            loss_test = event_acc.Scalars('Loss/test_epoch') 
            loss_val = event_acc.Scalars('Loss/val_epoch')
        except:
            print('no loss data available for %s' %fol)
            continue
        # Append to the lists
        for i in range(len(loss_train)):
            stepstr.append(loss_train[i].step)
            train_loss.append(loss_train[i].value)
                
        for i in range(len(loss_test)):
            stepste.append(loss_test[i].step)
            test_loss.append(loss_test[i].value)
        
        for i in range(len(loss_val)):
            stepsval.append(loss_val[i].step)
            val_loss.append(loss_val[i].value)


        ax1.plot(learnr, train_loss[-1], 'o',  color=col)
        ax5.plot(learnr, test_loss[-1], 'D', color=col)  
        ax9.plot(learnr, val_loss, 'x', color=col)    
        ax9.set_xlabel('learning rates') 
        ax1.set_ylabel('rmse model training')
        ax5.set_ylabel('rmse model testing')
        ax9.set_ylabel('rmse independent validation')
    
        ax2.plot(stepstr[-1]+1, train_loss[-1], 'o', color=col)
        ax6.plot(stepste[-1]+1, test_loss[-1], 'D', color=col)
        ax10.plot(stepste[-1]+1, val_loss, 'x', color=col)    
        ax10.set_xlabel('epochs') 
        epochs = stepstr[-1]+1

        ax3.plot(batch, train_loss[-1], 'o', color=col)
        ax7.plot(batch, test_loss[-1], 'D', color=col)
        ax11.plot(batch, val_loss, 'x', color=col)     
        ax11.set_xlabel('batch size')

        ax4.plot(samplesize, train_loss[-1], 'o', color=col)
        ax8.plot(samplesize, test_loss[-1], 'D', color=col)
        ax12.plot(samplesize, val_loss, 'x', color=col)
        ax12.set_xlabel('samplesize')

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
        ax10.set_facecolor('lightgrey')
        ax11.set_facecolor('lightgrey')
        ax12.set_facecolor('lightgrey')

        ax1.set_ylim(0, 1)
        # Collect each unique label and color for the legend
        legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s_%s_%s'%(model, epochs, batch, learnr, samplesize, cpu)))
# Add a single legend for all subplots outside the figure
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=6)
plt.tight_layout()
fig.savefig('%s/overview_hyperparam_testing_train_val_test_new_180_randomsampling_earlystopping_U2_ext.png' %plotting_folder, bbox_inches='tight')




'''spatial plots for different hyperparameter runs'''
# mask180 = np.load('../../data/testing_random_sampling_180/mask_validation.npy') #180
# target180 = np.load('../../data/testing_random_sampling_180/target_validation.npy')#180

# mask360 = np.load('../../data/testing_random_sampling_360/mask_validation.npy') #360
# target360 = np.load('../../data/testing_random_sampling_360/target_validation.npy')#360

# mask600 = np.load('../../data/testing_random_sampling_600/mask_validation.npy') #600
# target600 = np.load('../../data/testing_random_sampling_600/target_validation.npy')#600


# mask_test_na_180 = np.where(mask180==0, np.nan, 1)
# mask_test_na_360 = np.where(mask360==0, np.nan, 1)
# mask_test_na_600 = np.where(mask600==0, np.nan, 1)
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import data

maskRS = np.load('../../data/testing_random_sampling_fulltile_180_101010_deltawtd/mask_validation.npy') #180
target180 = np.load('../../data/testing_random_sampling_fulltile_180_101010_deltawtd/target_validation.npy')#180
mask_test_na_180 = np.where(maskRS==0, np.nan, 1)

#check if non normalized data is giving different result
# maskRS = np.load('../../data/testing_random_sampling_fulltile_180_inclwtd/mask_validation.npy') #180
# target180_nonn = np.load('../../data/testing_random_sampling_fulltile_180_inclwtd/tar_validation_norm_arr.npy')#180
# mask_test_na_180 = np.where(maskRS==0, np.nan, 1)
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

folders = glob.glob('../../results/testing/random_sampling_fulltile_180_101010_limitedInpSel_deltawtd/UNet2_50*')
for fol in folders[:]:
    print(fol)
    sampleS = float(fol.split('_')[-1])
    if '180' in fol:
        target_val = target180[:int(target180.shape[0]*sampleS)]  
        mask_test_na = mask_test_na_180[:int(mask_test_na_180.shape[0]*sampleS)]  
    # if '360' in fol:
    #     target_val = target360[:int(target360.shape[0]*sampleS)]  
    #     mask_test_na = mask_test_na_360[:int(mask_test_na_360.shape[0]*sampleS)]
    # if '600' in fol:
    #     target_val = target600[:int(target600.shape[0]*sampleS)]  
    #     mask_test_na = mask_test_na_600[:int(mask_test_na_600.shape[0]*sampleS)]

    try:
        y_pred_denorm = np.load('%s/yfull_pred_denorm_new.npy' %fol)
        # y_pred_denorm = np.load('%s/yfull_pred_raw_new.npy' %fol) #check if nonnormalised data is giving different result
    except:
        print('no prediction file available for %s' %fol)
        continue
    for i in range(y_pred_denorm.shape[0])[:10]:
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
        plt.imshow(target_val[i, 0, :, :]*mask_test_na[i,0,:,:], cmap='RdBu', vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Actual delta (colorbar 5-95 percentile)')
        plt.tight_layout()

        # plt.imshow(target180_nonn[i, 0, :, :]*mask_test_na[i,0,:,:], cmap='RdBu')#, vmin=-lim, vmax=lim)
        # plt.colorbar(shrink=0.8)

        plt.subplot(1, 5, 2)
        plt.imshow(y_pred_denorm[i, 0, :, :]*mask_test_na[i,0,:,:], cmap='RdBu',vmin=-lim, vmax=lim)
        plt.colorbar(shrink=0.8)
        plt.title('Predicted delta (colorbar 5-95 percentile)')

        vmin = min([np.nanmin(target_val[i, 0, :, :]*mask_test_na[i,0,:,:]),np.nanmin(y_pred_denorm[i, 0, :, :]*mask_test_na[i,0,:,:])])
        vmax = max([np.nanmax(target_val[i, 0, :, :]*mask_test_na[i,0,:,:]),np.nanmax(y_pred_denorm[i, 0, :, :]*mask_test_na[i,0,:,:])])
        plt.subplot(1, 5, 3)
        plt.scatter((target_val[i,0, :, :]*mask_test_na[i,0,:,:]).flatten(), (y_pred_denorm[i, 0, :, :]*mask_test_na[i,0,:,:]).flatten(),alpha=0.2, facecolors='none', edgecolors='r')
        plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2)
        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)
        plt.ylabel('Predicted delta') 
        plt.xlabel('Actual delta')

        plt.subplot(1, 5, 4)
        diff = (target_val[i, 0, :, :]*mask_test_na[i,0,:,:]) - (y_pred_denorm[i, 0, :, :]*mask_test_na[i,0,:,:]) #difference between wtd and calculated wtd
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



'''scatter plots for differen hyperparameter runs - by batch size per validation samples'''
folders_180 = glob.glob('../../results/testing/random_sampling_360/UNet2*')
plotting_folder = '../../results/testing/random_sampling_360/plots'
# only folders that have a y_pred_denorm.npy file
folders_180 = [f for f in folders_180 if os.path.exists('%s/y_pred_denorm_new.npy' %f)]
# sort the folders based on the batch size
folders_180 = sorted(folders_180, key=lambda x: int(x.split('_')[-2]))

# select folders per batch size
folders_180_2 = [f for f in folders_180 if '_2_' in f]
folders_180_4 = [f for f in folders_180 if '_4_' in f]
folders_180_8 = [f for f in folders_180 if '_8_' in f]
folders_180_16 = [f for f in folders_180 if '_16_' in f]
folders_180_32 = [f for f in folders_180 if '_32_' in f]
folders_180_64 = [f for f in folders_180 if '_64_' in f]

colors180_2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(folders_180_2)))
colors180_4 = plt.cm.Greens(np.linspace(0.4, 0.9, len(folders_180_4)))
colors180_8 = plt.cm.Blues(np.linspace(0.4, 0.9, len(folders_180_8)))
colors180_16 = plt.cm.Purples(np.linspace(0.4, 0.9, len(folders_180_16)))
colors180_32 = plt.cm.Greys(np.linspace(0.4, 0.9, len(folders_180_32)))
colors180_64 = plt.cm.spring(np.linspace(0, 0.5, len(folders_180_64)))

folderlist = [folders_180_2, folders_180_4, folders_180_8, folders_180_16, folders_180_32, folders_180_64]
colorlist = [colors180_2, colors180_4, colors180_8, colors180_16, colors180_32, colors180_64]

mask180 = np.load('../../data/testing_random_sampling_360/mask_validation.npy') #180
mask_test_na = np.where(mask180==0, np.nan, 1)
target_val = np.load('../../data/testing_random_sampling_360/target_validation.npy')#180

areas = len(target_val)
#iterate over all areas (0 - areas)
areas_tot  = np.arange(0, areas, 1)

for t in areas_tot[1:5]:
    print('area %s' %t)
    legend_elements = []
    # Create a figure with constrained layout for spacing adjustments
    fig = plt.figure(constrained_layout=True, figsize=(20, 6))
    # Define the GridSpec with 2 rows and 4 columns
    # The left subplot will span both rows, and the right subplots will be in a 2x3 grid
    gs = gridspec.GridSpec(2, 4, figure=fig)
    # Large subplot on the left (spanning both rows)
    ax1 = fig.add_subplot(gs[:, 0])  # Use [:, 0] to span both rows in the first column
    ax1.set_title("wtd target")
    # colorbar of ax1 plot
   
    # Smaller subplots in a 2x3 grid on the right
    # Top row, right side
    ax2 = fig.add_subplot(gs[0, 1])  # Top left of the 2x3 grid
    ax3 = fig.add_subplot(gs[0, 2])  # Top middle of the 2x3 grid
    ax4 = fig.add_subplot(gs[0, 3])  # Top right of the 2x3 grid

    # Bottom row, right side
    ax5 = fig.add_subplot(gs[1, 1])  # Bottom left of the 2x3 grid
    ax6 = fig.add_subplot(gs[1, 2])  # Bottom middle of the 2x3 grid
    ax7 = fig.add_subplot(gs[1, 3])  # Bottom right of the 2x3 grid

    # Set titles for each subplot
    ax2.set_title("BS 2")
    ax3.set_title("BS 4")
    ax4.set_title("BS 8")
    ax5.set_title("BS 16")
    ax6.set_title("BS 32")
    ax7.set_title("BS 64")

    #set background color for each subplot
    ax2.set_facecolor('lightgrey')
    ax3.set_facecolor('lightgrey')
    ax4.set_facecolor('lightgrey')
    ax5.set_facecolor('lightgrey')
    ax6.set_facecolor('lightgrey')
    ax7.set_facecolor('lightgrey')

    for i, (flist, collist) in enumerate(zip(folderlist[:], colorlist[:])):
        j = i+1
        print(j)
        targetvals = []
        predictedvals = []
        for fol, col in zip(flist[:], collist[:]):
            # print(fol)
            fol_name = fol.split('/')[-1]
            batch = fol_name.split('_')[3]

            try:
                y_pred_denorm = np.load('%s/y_pred_denorm_new.npy' %fol)
            except:
                print('no prediction file available for %s' %fol)
                continue
            if y_pred_denorm.shape[2] != target_val.shape[2]:
                print('skipping')
                continue
            target = target_val[t, 0, :, :]*mask_test_na[t,0,:,:]
            pred = y_pred_denorm[t, 0, :, :]*mask_test_na[t,0,:,:]

            vmin = min([np.nanmin(target),np.nanmin(pred)])
            vmax = max([np.nanmax(target),np.nanmax(pred)])
            print(vmin, vmax)
            if np.isnan(vmin) and np.isnan(vmax):
                continue

            if j == 1:
                cax1 = ax1.imshow(target, cmap='viridis') # wtd target plot       
                ax2.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                ax2.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                ax2.set_xlim(vmin,vmax)
                ax2.set_ylim(vmin,vmax)
                ax2.set_ylabel('Predicted delta wtd' ) 
                ax2.set_xlabel('Actual delta wtd')
                legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    
            if j == 2:
                ax3.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                ax3.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                ax3.set_xlim(vmin,vmax)
                ax3.set_ylim(vmin,vmax)
                ax3.set_ylabel('Predicted delta wtd' ) 
                ax3.set_xlabel('Actual delta wtd')
                legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
            if j == 3:
                ax4.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                ax4.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                ax4.set_xlim(vmin,vmax)
                ax4.set_ylim(vmin,vmax)
                ax4.set_ylabel('Predicted delta wtd' ) 
                ax4.set_xlabel('Actual delta wtd')
                legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
            if j == 4:
                ax5.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                ax5.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                ax5.set_xlim(vmin,vmax)
                ax5.set_ylim(vmin,vmax)
                ax5.set_ylabel('Predicted delta wtd' ) 
                ax5.set_xlabel('Actual delta wtd')
                legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
            if j == 5:
                ax6.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                ax6.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                ax6.set_xlim(vmin,vmax)
                ax6.set_ylim(vmin,vmax)
                ax6.set_ylabel('Predicted delta wtd' ) 
                ax6.set_xlabel('Actual delta wtd')
                legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
            if j == 6:
                ax7.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                ax7.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                ax7.set_xlim(vmin,vmax)
                ax7.set_ylim(vmin,vmax)
                ax7.set_ylabel('Predicted delta wtd' ) 
                ax7.set_xlabel('Actual delta wtd')
                legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    cbar = fig.colorbar(cax1, ax=ax1, location="right", fraction=0.05, pad=0.05)
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=6)
    # plt.tight_layout()
    plt.savefig('%s/areas_scatter/scatter_predvstarget_randomsamples_180_%s.png' %(plotting_folder, t), bbox_inches='tight')
    plt.close()



'''scatter plots for differen hyperparameter runs - by batch size and over all validation samples'''
folders_180 = glob.glob('../../results/testing/random_sampling_180/UNet2*')
plotting_folder = '../../results/testing/random_sampling_180/plots'
# only folders that have a y_pred_denorm.npy file
folders_180 = [f for f in folders_180 if os.path.exists('%s/y_pred_denorm_new.npy' %f)]
# sort the folders based on the batch size
folders_180 = sorted(folders_180, key=lambda x: int(x.split('_')[-2]))

folders_180 = [f for f in folders_180 if '0.0001' in f]

# select folders per batch size
folders_180_2 = [f for f in folders_180 if '_2_' in f]
folders_180_4 = [f for f in folders_180 if '_4_' in f]
folders_180_8 = [f for f in folders_180 if '_8_' in f]
folders_180_16 = [f for f in folders_180 if '_16_' in f]
folders_180_32 = [f for f in folders_180 if '_32_' in f]
folders_180_64 = [f for f in folders_180 if '_64_' in f]

colors180_2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(folders_180_2)))
colors180_4 = plt.cm.Greens(np.linspace(0.4, 0.9, len(folders_180_4)))
colors180_8 = plt.cm.Blues(np.linspace(0.4, 0.9, len(folders_180_8)))
colors180_16 = plt.cm.Purples(np.linspace(0.4, 0.9, len(folders_180_16)))
colors180_32 = plt.cm.Greys(np.linspace(0.4, 0.9, len(folders_180_32)))
colors180_64 = plt.cm.spring(np.linspace(0, 0.5, len(folders_180_64)))

folderlist = [folders_180_2, folders_180_4, folders_180_8, folders_180_16, folders_180_32, folders_180_64]
colorlist = [colors180_2, colors180_4, colors180_8, colors180_16, colors180_32, colors180_64]

mask180 = np.load('../../data/testing_random_sampling_180/mask_validation.npy') #180
mask_test_na = np.where(mask180==0, np.nan, 1)
target_val = np.load('../../data/testing_random_sampling_180/target_validation.npy')#180

# legend_elements = []
# Create a figure with constrained layout for spacing adjustments
# fig = plt.figure(constrained_layout=True, figsize=(20, 6))
# # Define the GridSpec with 2 rows and 4 columns
# # The left subplot will span both rows, and the right subplots will be in a 2x3 grid
# gs = gridspec.GridSpec(2, 4, figure=fig)

# # Top row, right side
# ax2 = fig.add_subplot(gs[0, 1])  # Top left of the 2x3 grid
# ax3 = fig.add_subplot(gs[0, 2])  # Top middle of the 2x3 grid
# ax4 = fig.add_subplot(gs[0, 3])  # Top right of the 2x3 grid

# # Bottom row, right side
# ax5 = fig.add_subplot(gs[1, 1])  # Bottom left of the 2x3 grid
# ax6 = fig.add_subplot(gs[1, 2])  # Bottom middle of the 2x3 grid
# ax7 = fig.add_subplot(gs[1, 3])  # Bottom right of the 2x3 grid

# # Set titles for each subplot
# ax2.set_title("BS 2")
# ax3.set_title("BS 4")
# ax4.set_title("BS 8")
# ax5.set_title("BS 16")
# ax6.set_title("BS 32")
# ax7.set_title("BS 64")

# #set background color for each subplot
# ax2.set_facecolor('lightgrey')
# ax3.set_facecolor('lightgrey')
# ax4.set_facecolor('lightgrey')
# ax5.set_facecolor('lightgrey')
# ax6.set_facecolor('lightgrey')
# ax7.set_facecolor('lightgrey')


for i, (flist, collist) in enumerate(zip(folderlist[:], colorlist[:])):
    j = i+1
    print(j)
    legend_elements = []
    # Create a figure with constrained layout for spacing adjustments
    fig = plt.figure(constrained_layout=True, figsize=(20, 6))
    # Define the GridSpec with 2 rows and 4 columns
    # The left subplot will span both rows, and the right subplots will be in a 2x3 grid
    gs = gridspec.GridSpec(2, 4, figure=fig)
    # Smaller subplots in a 2x3 grid on the right
    # Top row, right side
    ax2 = fig.add_subplot(gs[0, 1])  # Top left of the 2x3 grid
    ax3 = fig.add_subplot(gs[0, 2])  # Top middle of the 2x3 grid
    ax4 = fig.add_subplot(gs[0, 3])  # Top right of the 2x3 grid

    # Bottom row, right side
    ax5 = fig.add_subplot(gs[1, 1])  # Bottom left of the 2x3 grid
    ax6 = fig.add_subplot(gs[1, 2])  # Bottom middle of the 2x3 grid
    ax7 = fig.add_subplot(gs[1, 3])  # Bottom right of the 2x3 grid

    # Set titles for each subplot
    ax2.set_title("BS 2")
    ax3.set_title("BS 4")
    ax4.set_title("BS 8")
    ax5.set_title("BS 16")
    ax6.set_title("BS 32")
    ax7.set_title("BS 64")

    #set background color for each subplot
    ax2.set_facecolor('lightgrey')
    ax3.set_facecolor('lightgrey')
    ax4.set_facecolor('lightgrey')
    ax5.set_facecolor('lightgrey')
    ax6.set_facecolor('lightgrey')
    ax7.set_facecolor('lightgrey')

    predall = []
    targetall = []
    for fol, col in zip(flist[:], collist[:]):
        print(fol)
        fol_name = fol.split('/')[-1]
        # batch = fol_name.split('_')[3]

        y_pred_denorm = np.load('%s/y_pred_denorm_new.npy' %fol)

        if y_pred_denorm.shape[2] != target_val.shape[2]:
            print('skipping')
            continue
        target = target_val[:, 0, :, :]*mask_test_na[:,0,:,:]
        pred = y_pred_denorm[:, 0, :, :]*mask_test_na[:,0,:,:]

        predall.append(pred)
        targetall.append(target)
    
    predplot = np.concatenate(predall, axis=0)
    targetplot = np.concatenate(targetall, axis=0)

    vmin = min([np.nanmin(targetplot),np.nanmin(predplot)])
    vmax = max([np.nanmax(targetplot),np.nanmax(predplot)])
    # print(vmin, vmax)
    if np.isnan(vmin) and np.isnan(vmax):
        continue

    if j == 1:
        # cax1 = ax1.imshow(target, cmap='viridis') # wtd target plot       
        ax2.scatter(targetplot, predplot, alpha=0.2, facecolors='none', edgecolors=col)
        ax2.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
        ax2.set_xlim(vmin,vmax)
        ax2.set_ylim(vmin,vmax)
        ax2.set_ylabel('Predicted delta wtd' ) 
        ax2.set_xlabel('Actual delta wtd')
        legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))

    if j == 2:
        ax3.scatter(targetplot, predplot, alpha=0.2, facecolors='none', edgecolors=col)
        ax3.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
        ax3.set_xlim(vmin,vmax)
        ax3.set_ylim(vmin,vmax)
        ax3.set_ylabel('Predicted delta wtd' ) 
        ax3.set_xlabel('Actual delta wtd')
        # legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    if j == 3:
        ax4.scatter(targetplot, predplot, alpha=0.2, facecolors='none', edgecolors=col)
        ax4.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
        ax4.set_xlim(vmin,vmax)
        ax4.set_ylim(vmin,vmax)
        ax4.set_ylabel('Predicted delta wtd' ) 
        ax4.set_xlabel('Actual delta wtd')
        # legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    if j == 4:
        ax5.scatter(targetplot, predplot, alpha=0.2, facecolors='none', edgecolors=col)
        ax5.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
        ax5.set_xlim(vmin,vmax)
        ax5.set_ylim(vmin,vmax)
        ax5.set_ylabel('Predicted delta wtd' ) 
        ax5.set_xlabel('Actual delta wtd')
        # legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    if j == 5:
        ax6.scatter(targetplot, predplot, alpha=0.2, facecolors='none', edgecolors=col)
        ax6.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
        ax6.set_xlim(vmin,vmax)
        ax6.set_ylim(vmin,vmax)
        ax6.set_ylabel('Predicted delta wtd' ) 
        ax6.set_xlabel('Actual delta wtd')
        # legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    if j == 6:
        ax7.scatter(targetplot, predplot, alpha=0.2, facecolors='none', edgecolors=col)
        ax7.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
        ax7.set_xlim(vmin,vmax)
        ax7.set_ylim(vmin,vmax)
        ax7.set_ylabel('Predicted delta wtd' ) 
        ax7.set_xlabel('Actual delta wtd')
        # legend_elements.append(Line2D([0], [0], marker='x', color=col, label=fol_name))
    # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=6)
    plt.savefig('%s/areas_scatter/scatter_predvstarget_randomsamples_180_total_%s.png' %(plotting_folder,j), bbox_inches='tight')
    plt.close()
# cbar = fig.colorbar(cax1, ax=ax1, location="right", fraction=0.05, pad=0.05)
# fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=6)
# plt.tight_layout()
# plt.savefig('%s/areas_scatter/scatter_predvstarget_randomsamples_180_total_%s.png' %(plotting_folder), bbox_inches='tight')
# plt.close()



'''scatter plots for different learning rates - by batch size per validation samples'''
folders_180 = glob.glob('../../results/testing/random_sampling_180/UNet2*')
plotting_folder = '../../results/testing/random_sampling_180/plots'
# only folders that have a y_pred_denorm.npy file
folders_180 = [f for f in folders_180 if os.path.exists('%s/y_pred_denorm_new.npy' %f)]
# sort the folders based on the batch size
folders_180 = sorted(folders_180, key=lambda x: int(x.split('_')[-2]))

mask180 = np.load('../../data/testing_random_sampling_180/mask_validation.npy') #180
mask_test_na = np.where(mask180==0, np.nan, 1)
target_val = np.load('../../data/testing_random_sampling_180/target_validation.npy')#180

areas = len(target_val)
#iterate over all areas (0 - areas)
areas_tot  = np.arange(0, areas, 1)

learning_rates = [0.0001, 0.0005, 0.001, 0.005]
epochs = [10, 50, 100]

for lr in epochs[:]:
    folders_180_lr = [f for f in folders_180 if '_%s_' %lr in f]
    # select folders per batch size
    folders_180_2 = [f for f in folders_180_lr if '_2_' in f]
    folders_180_4 = [f for f in folders_180_lr if '_4_' in f]
    folders_180_8 = [f for f in folders_180_lr if '_8_' in f]
    folders_180_16 = [f for f in folders_180_lr if '_16_' in f]
    folders_180_32 = [f for f in folders_180_lr if '_32_' in f]
    folders_180_64 = [f for f in folders_180_lr if '_64_' in f]

    colors180_2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(folders_180_2)))
    colors180_4 = plt.cm.Greens(np.linspace(0.4, 0.9, len(folders_180_4)))
    colors180_8 = plt.cm.Blues(np.linspace(0.4, 0.9, len(folders_180_8)))
    colors180_16 = plt.cm.Purples(np.linspace(0.4, 0.9, len(folders_180_16)))
    colors180_32 = plt.cm.Greys(np.linspace(0.4, 0.9, len(folders_180_32)))
    colors180_64 = plt.cm.spring(np.linspace(0, 0.5, len(folders_180_64)))

    folderlist = [folders_180_2, folders_180_4, folders_180_8, folders_180_16, folders_180_32, folders_180_64]
    colorlist = [colors180_2, colors180_4, colors180_8, colors180_16, colors180_32, colors180_64]

    for t in areas_tot[:10]:
        print('area %s' %t)
        legend_elements = []
        # Create a figure with constrained layout for spacing adjustments
        fig = plt.figure(constrained_layout=True, figsize=(20, 6))
        # Define the GridSpec with 2 rows and 4 columns
        # The left subplot will span both rows, and the right subplots will be in a 2x3 grid
        gs = gridspec.GridSpec(2, 4, figure=fig)
        # Large subplot on the left (spanning both rows)
        ax1 = fig.add_subplot(gs[:, 0])  # Use [:, 0] to span both rows in the first column
        ax1.set_title("wtd target")
       
        # Smaller subplots in a 2x3 grid on the right
        # Top row, right side
        ax2 = fig.add_subplot(gs[0, 1])  # Top left of the 2x3 grid
        ax3 = fig.add_subplot(gs[0, 2])  # Top middle of the 2x3 grid
        ax4 = fig.add_subplot(gs[0, 3])  # Top right of the 2x3 grid

        # Bottom row, right side
        ax5 = fig.add_subplot(gs[1, 1])  # Bottom left of the 2x3 grid
        ax6 = fig.add_subplot(gs[1, 2])  # Bottom middle of the 2x3 grid
        ax7 = fig.add_subplot(gs[1, 3])  # Bottom right of the 2x3 grid

        # Set titles for each subplot
        ax2.set_title("BS 2")
        ax3.set_title("BS 4")
        ax4.set_title("BS 8")
        ax5.set_title("BS 16")
        ax6.set_title("BS 32")
        ax7.set_title("BS 64")

        #set background color for each subplot
        ax2.set_facecolor('lightgrey')
        ax3.set_facecolor('lightgrey')
        ax4.set_facecolor('lightgrey')
        ax5.set_facecolor('lightgrey')
        ax6.set_facecolor('lightgrey')
        ax7.set_facecolor('lightgrey')

        for i, (flist, collist) in enumerate(zip(folderlist[:], colorlist[:])):
            j = i+1
            print(j)
            targetvals = []
            predictedvals = []
            for fol, col in zip(flist[:], collist[:]):
                # print(fol)
                fol_name = fol.split('/')[-1]
                batch = fol_name.split('_')[3]
                try:
                    y_pred_denorm = np.load('%s/y_pred_denorm_new.npy' %fol)
                    event_files = [os.path.join(fol, f) for f in os.listdir(fol) if 'events.out.tfevents' in f]
                    train_loss = []
                    stepstr = []
                    event_acc = EventAccumulator(event_files[0])
                    event_acc.Reload()
                    loss_train = event_acc.Scalars('Loss/train_epoch')
                    for i in range(len(loss_train)):
                        stepstr.append(loss_train[i].step)
                        train_loss.append(loss_train[i].value)
                except:
                    print('no prediction file available for %s' %fol)
                    continue
                if y_pred_denorm.shape[2] != target_val.shape[2]:
                    print('skipping')
                    continue
                target = target_val[t, 0, :, :]*mask_test_na[t,0,:,:]
                pred = y_pred_denorm[t, 0, :, :]*mask_test_na[t,0,:,:]

                vmin = min([np.nanmin(target),np.nanmin(pred)])
                vmax = max([np.nanmax(target),np.nanmax(pred)])
                # print(vmin, vmax)
                if np.isnan(vmin) and np.isnan(vmax):
                    continue

                if j == 1:
                    cax1 = ax1.imshow(target, cmap='viridis') # wtd target plot       
                    ax2.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                    ax2.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                    ax2.set_xlim(vmin,vmax)
                    ax2.set_ylim(vmin,vmax)
                    ax2.set_ylabel('Predicted delta wtd' ) 
                    ax2.set_xlabel('Actual delta wtd')
                    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_actE_%s' %(fol_name, stepstr[-1]+1)))
        
                if j == 2:
                    ax3.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                    ax3.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                    ax3.set_xlim(vmin,vmax)
                    ax3.set_ylim(vmin,vmax)
                    ax3.set_ylabel('Predicted delta wtd' ) 
                    ax3.set_xlabel('Actual delta wtd')
                    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_actE_%s' %(fol_name, stepstr[-1]+1)))
                if j == 3:
                    ax4.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                    ax4.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                    ax4.set_xlim(vmin,vmax)
                    ax4.set_ylim(vmin,vmax)
                    ax4.set_ylabel('Predicted delta wtd' ) 
                    ax4.set_xlabel('Actual delta wtd')
                    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_actE_%s' %(fol_name, stepstr[-1]+1)))
                if j == 4:
                    ax5.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                    ax5.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                    ax5.set_xlim(vmin,vmax)
                    ax5.set_ylim(vmin,vmax)
                    ax5.set_ylabel('Predicted delta wtd' ) 
                    ax5.set_xlabel('Actual delta wtd')
                    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_actE_%s' %(fol_name, stepstr[-1]+1)))
                if j == 5:
                    ax6.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                    ax6.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                    ax6.set_xlim(vmin,vmax)
                    ax6.set_ylim(vmin,vmax)
                    ax6.set_ylabel('Predicted delta wtd' ) 
                    ax6.set_xlabel('Actual delta wtd')
                    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_actE_%s' %(fol_name, stepstr[-1]+1)))
                if j == 6:
                    ax7.scatter(target.flatten(), pred.flatten(), alpha=0.2, facecolors='none', edgecolors=col)
                    ax7.plot([vmin, vmax], [vmin, vmax], 'k', lw=2)
                    ax7.set_xlim(vmin,vmax)
                    ax7.set_ylim(vmin,vmax)
                    ax7.set_ylabel('Predicted delta wtd' ) 
                    ax7.set_xlabel('Actual delta wtd')
                    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_actE_%s' %(fol_name, stepstr[-1])))
        cbar = fig.colorbar(cax1, ax=ax1, location="right", fraction=0.05, pad=0.05)
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=6)
        # plt.tight_layout()
        plt.savefig('%s/areas_scatter/scatter_predvstarget_randomsamples_180_%s_ep_%s.png' %(plotting_folder, t, lr), bbox_inches='tight')
        plt.close()