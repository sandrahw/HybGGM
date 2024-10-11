'''hyperparameter tuning visualization'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


folders = glob.glob(r'..\training\logs\*')
num_folders = len(folders)  
colors = plt.cm.viridis(np.linspace(0, 1, num_folders)) 


fig, ax = plt.subplots()
for fol,col in zip(folders[:], colors[:]):
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
plt.savefig(r'..\training\logs\training_loss.png')

