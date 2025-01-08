'''hyperparameter tuning visualization'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.lines import Line2D

folders = glob.glob(r'../training/logs_CNN_randomspatial/360cells/1*')
plotting_folder = r'../training/logs_CNN_randomspatial/360velss/plots'
if not os.path.exists(plotting_folder):
    os.makedirs(plotting_folder)
num_folders = len(folders)  
colors = plt.cm.viridis(np.linspace(0, 1, num_folders)) 

legend_elements = []
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(15, 10))
for fol, col in zip(folders[:], colors[:]):
    # get folder name which is the hyperparameter information in last few characters
    fol_name = fol.split('\\')[-1]
    print(fol_name)
    epochs = fol_name.split('_')[0]
    learnr = fol_name.split('_')[1]
    batch = fol_name.split('_')[2]
    kernel = fol_name.split('_')[3]
    # encoders = fol_name.split('_')[4]

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
    try:
        loss_test = event_acc.Scalars('Loss/test_epoch')
    except:
        print('no test loss available for %s' %fol_name)
        continue
    #loss_test = event_acc.Scalars('Loss/test_epoch')

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

    ax1.plot(learnr, train_loss[-1], 'o',  color=col)
    ax6.plot(learnr, val_loss[-1], 'D', color=col)  
    ax11.plot(learnr, test_loss[-1], 'x', color=col)    
    ax11.set_xlabel('learning rates') 
    ax1.set_ylabel('rmse training')
    ax6.set_ylabel('rmse validation')
    ax11.set_ylabel('rmse independent testing')
  
    ax2.plot(epochs, train_loss[-1], 'o', color=col)
    ax7.plot(epochs, val_loss[-1], 'D', color=col)
    ax12.plot(epochs, test_loss[-1], 'x', color=col)    
    ax12.set_xlabel('epochs') 

    ax3.plot(batch, train_loss[-1], 'o', color=col)
    ax8.plot(batch, val_loss[-1], 'D', color=col)
    ax13.plot(batch, test_loss[-1], 'x', color=col)     
    ax13.set_xlabel('batch size')

    ax4.plot(kernel, train_loss[-1], 'o', color=col)
    ax9.plot(kernel, val_loss[-1], 'D', color=col)
    ax14.plot(kernel, test_loss[-1], 'x', color=col)
    ax14.set_xlabel('kernel size')

    # ax5.plot(encoders, train_loss[-1], 'o', color=col)
    # ax10.plot(encoders, val_loss[-1], 'D', color=col)
    # ax15.plot(encoders, test_loss[-1], 'x', color=col)
    # ax15.set_xlabel('encoders')

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
    ax13.set_facecolor('lightgrey')
    ax14.set_facecolor('lightgrey')
    ax15.set_facecolor('lightgrey')

    # Collect each unique label and color for the legend
    legend_elements.append(Line2D([0], [0], marker='x', color=col, label='%s_%s_%s_%s'%(epochs, batch, learnr, kernel)))

# Add a single legend for all subplots outside the figure
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.002), ncol=5)
ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax3.set_ylim([0, 1])
ax4.set_ylim([0, 1])
ax5.set_ylim([0, 1])
ax6.set_ylim([0, 1])
ax7.set_ylim([0, 1])
ax8.set_ylim([0, 1])
ax9.set_ylim([0, 1])
ax10.set_ylim([0, 1])
ax11.set_ylim([0, 1])
ax12.set_ylim([0, 1])
ax13.set_ylim([0, 1])
ax14.set_ylim([0, 1])
ax15.set_ylim([0, 1])

plt.tight_layout()
fig.savefig('%s/overview_hyperparam_testing_train_val_test_allU6.png' %plotting_folder, bbox_inches='tight')


