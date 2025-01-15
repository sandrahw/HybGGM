import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob

training_folder = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Scripts\HybGGM\training\logs_dev2'
folders = glob.glob(r'%s\head*' % training_folder)

UNet = xr.open_dataset(r'%s\full_pred_denorm.nc' % [f for f in folders if '_CNN' in f][0]).to_array().values
LSTM = xr.open_dataset(r'%s\full_pred_denorm.nc' % [f for f in folders if '_LSTM' in f][1]).to_array().values
UNetLSTM = xr.open_dataset(r'%s\full_pred_denorm.nc' % [f for f in folders if '_CNNLSTMinp' in f][0]).to_array().values
LSTMUNet = xr.open_dataset(r'%s\full_pred_denorm.nc' % [f for f in folders if '_LSTMCNNinp' in f][0]).to_array().values

target = np.load(r'%s\y.npy' %[f for f in folders if '_LSTM_CNNinp' in f][0])
target = target[:, 0, :, :]
UNet = UNet[0,:, :, :]
LSTM = LSTM[0,:, :, :]
UNetLSTM = UNetLSTM[0,:, :, :]
LSTMUNet = LSTMUNet[0,:, :, :]

diff_UNet = target - UNet
diff_LSTM = target - LSTM
diff_UNetLSTM = target - UNetLSTM
diff_LSTMUNet = target - LSTMUNet


#combine all differences arrays together
diff = np.concatenate((diff_UNet, diff_LSTM, diff_UNetLSTM, diff_LSTMUNet), axis=0)
tot = np.concatenate((target, UNet, LSTM, UNetLSTM, LSTMUNet), axis=0)

#only consider diff values that are within the 1st and 99th percentile
diffpercthresh = np.clip(diff, np.percentile(diff, 1), np.percentile(diff, 99))

min_val = np.min(tot)
max_val = np.max(tot)

min_diff = np.min(diffpercthresh)
max_diff = np.max(diffpercthresh)
limdiff = max(abs(min_diff), abs(max_diff))

for i in np.arange(0, len(target))[-1:]:
    print(i)
    plt.figure(figsize=(30,10))
    plt.subplot(1, 5, 1)
    plt.imshow(target[i,:,:], cmap='viridis',vmin=min_val, vmax=max_val)
    plt.colorbar(shrink=0.5)
    plt.title('target')
    plt.subplot(1, 5, 2)
    plt.imshow(UNet[i,:,:], cmap='viridis',vmin=min_val, vmax=max_val)
    plt.colorbar(shrink=0.5)
    plt.title('UNet')
    plt.subplot(1, 5, 3)
    plt.imshow(LSTM[i,:,:], cmap='viridis',vmin=min_val, vmax=max_val)
    plt.colorbar(shrink=0.5)
    plt.title('LSTM')
    plt.subplot(1, 5, 4)
    plt.imshow(UNetLSTM[i,:,:], cmap='viridis', vmin=min_val, vmax=max_val)
    plt.colorbar(shrink=0.5)
    plt.title('UNetLSTM')
    plt.subplot(1, 5, 5)
    plt.imshow(LSTMUNet[i,:,:], cmap='viridis',vmin=min_val, vmax=max_val)
    plt.colorbar(shrink=0.5)
    plt.title('LSTMUNet')
    plt.tight_layout
    plt.savefig(r'%s\comparison_models_to_target_ts_%s.png' % (training_folder, i))

    plt.figure(figsize=(30,10))
    plt.subplot(1, 4, 1)
    plt.imshow(diff_UNet[i,:,:], cmap='RdBu',vmin=-limdiff, vmax=limdiff)
    plt.colorbar(shrink=0.5)
    plt.title('diff UNet')
    plt.subplot(1, 4, 2)
    plt.imshow(diff_LSTM[i,:,:], cmap='RdBu',vmin=-limdiff, vmax=limdiff)
    plt.colorbar(shrink=0.5)
    plt.title('diff LSTM')
    plt.subplot(1, 4, 3)
    plt.imshow(diff_UNetLSTM[i,:,:], cmap='RdBu',vmin=-limdiff, vmax=limdiff)
    plt.colorbar(shrink=0.5)
    plt.title('diff UNetLSTM')
    plt.subplot(1, 4, 4)
    plt.imshow(diff_LSTMUNet[i,:,:], cmap='RdBu',vmin=-limdiff, vmax=limdiff)
    plt.colorbar(shrink=0.5)
    plt.title('diff LSTMUNet')
    plt.tight_layout
    plt.savefig(r'%s\comparison_diff_models_to_target_ts_%s.png' % (training_folder, i))

    plt.figure(figsize=(10,10))
    plt.subplot(1, 1, 1)
    plt.scatter(target[i,:,:].flatten(), UNet[i,:,:].flatten(), s=1, c='b', label='UNet', alpha=0.5)
    plt.scatter(target[i,:,:].flatten(), LSTM[i,:,:].flatten(), s=1, c='r', label='LSTM', alpha=0.5)
    plt.scatter(target[i,:,:].flatten(), UNetLSTM[i,:,:].flatten(), s=1, c='g', label='UNetLSTM', alpha=0.5)
    plt.scatter(target[i,:,:].flatten(), LSTMUNet[i,:,:].flatten(), s=1, c='y', label='LSTMUNet', alpha=0.5)  
    plt.legend()
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('target')
    plt.ylabel('model')
    plt.savefig(r'%s\comparison_models_to_target_scatter_ts_%s.png' % (training_folder, i))


plt.figure(figsize=(10,10))
plt.scatter(target[:,:,:].flatten(), UNet[:,:,:].flatten(), s=1, c='b', label='UNet', alpha=0.5)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('target')
plt.ylabel('model')
plt.tight_layout
plt.savefig(r'%s\comparison_models_to_target_scatter_UNet.png' % (training_folder))

plt.figure(figsize=(10,10))
plt.scatter(target[:,:,:].flatten(), LSTM[:,:,:].flatten(), s=1, c='r', label='LSTM', alpha=0.5)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('target')
plt.ylabel('model')
plt.tight_layout
plt.savefig(r'%s\comparison_models_to_target_scatter_LSTM.png' % (training_folder))

plt.figure(figsize=(10,10))
plt.scatter(target[:,:,:].flatten(), UNetLSTM[:,:,:].flatten(), s=1, c='g', label='UNetLSTM', alpha=0.5)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('target')
plt.ylabel('model')
plt.tight_layout
plt.savefig(r'%s\comparison_models_to_target_scatter_UNetLSTM.png' % (training_folder))

plt.figure(figsize=(10,10))
plt.scatter(target[:,:,:].flatten(), LSTMUNet[:,:,:].flatten(), s=1, c='y', label='LSTMUNet', alpha=0.5)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('target')
plt.ylabel('model')
plt.tight_layout    
plt.savefig(r'%s\comparison_models_to_target_scatter_LSTMUNet.png' % (training_folder))


# plt.subplot(1, 5, 5)
plt.figure(figsize=(10,10))
plt.scatter(target[:,:,:].flatten(), UNet[:,:,:].flatten(), s=1, c='b', label='UNet', alpha=0.5)
plt.scatter(target[:,:,:].flatten(), LSTM[:,:,:].flatten(), s=1, c='r', label='LSTM', alpha=0.5)
plt.scatter(target[:,:,:].flatten(), UNetLSTM[:,:,:].flatten(), s=1, c='g', label='UNetLSTM', alpha=0.5)
plt.scatter(target[:,:,:].flatten(), LSTMUNet[:,:,:].flatten(), s=1, c='y', label='LSTMUNet', alpha=0.5)
plt.legend()
plt.savefig(r'%s\comparison_models_to_target_scatter.png' % (training_folder))

difflstms = LSTM - LSTMUNet

minlstm = np.min(np.percentile(difflstms, 1))
maxlstm = np.max(np.percentile(difflstms, 99))
limlstms = max(abs(minlstm), abs(maxlstm))
plt.figure(figsize=(10,10))
plt.imshow(difflstms[0,:,:], cmap='RdBu',vmin=-limlstms, vmax=limlstms)
plt.colorbar()
plt.title('diff LSTM - LSTMUNet')
plt.savefig(r'%s\comparison_diff_LSTM_LSTMUNet.png' % (training_folder))

minunets = UNet - UNetLSTM
minunet = np.min(np.percentile(minunets, 1))
maxunet = np.max(np.percentile(minunets, 99))
limunets = max(abs(minunet), abs(maxunet))
plt.figure(figsize=(10,10))
plt.imshow(minunets[0,:,:], cmap='RdBu',vmin=-limunets, vmax=limunets)
plt.colorbar()
plt.title('diff UNet - UNetLSTM')
plt.savefig(r'%s\comparison_diff_UNet_UNetLSTM.png' % (training_folder))


mixedmodel = UNetLSTM - LSTMUNet
minmixed = np.min(np.percentile(mixedmodel, 1))
maxmixed = np.max(np.percentile(mixedmodel, 99))
limmixed = max(abs(minmixed), abs(maxmixed))
plt.figure(figsize=(10,10))
plt.imshow(mixedmodel[0,:,:], cmap='RdBu',vmin=-limmixed, vmax=limmixed)
plt.colorbar()
plt.title('diff UNetLSTM - LSTMUNet')
plt.savefig(r'%s\comparison_diff_UNetLSTM_LSTMUNet.png' % (training_folder))


plt.figure(figsize=(10,10))
plt.hist(UNet.flatten(), bins=100, label='UNet', histtype='step', color='r')
plt.hist(LSTM.flatten(), bins=100, label='LSTM', histtype='step', color='blue')
plt.hist(UNetLSTM.flatten(), bins=100, label='UNetLSTM', histtype='step', color='orange')
plt.hist(LSTMUNet.flatten(), bins=100, label='LSTMUNet', histtype='step', color='green')
#plot target hist empty just the outline in black
plt.hist(target.flatten(), bins=100, label='target', histtype='step', color='k')
plt.legend()
plt.savefig(r'%s\comparison_models_to_target_hist.png' % (training_folder))



def compute_temporal_correlation(target, predictions):
    """
    Compute temporal correlation for each spatial location.
    """
    time_steps, n_lat, n_lon = target.shape
    correlation_map = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            target_series = target[:, i, j]
            predictions_series = predictions[:, i, j]
            if np.std(target_series) > 0 and np.std(predictions_series) > 0:
                correlation_map[i, j] = np.corrcoef(target_series, predictions_series)[0, 1]
            else:
                correlation_map[i, j] = np.nan  # Handle cases with no variation
    return correlation_map

corr_UNet = compute_temporal_correlation(target, UNet)
corr_LSTM = compute_temporal_correlation(target, LSTM)
corr_LSTMUNet = compute_temporal_correlation(target, LSTMUNet)
corr_UNetLSTM = compute_temporal_correlation(target, UNetLSTM)  

plt.figure(figsize=(20,10))
plt.subplot(1, 4, 1)
plt.imshow(corr_UNet, cmap='RdBu',vmin=-1, vmax=1) 
plt.colorbar(shrink=0.5)    
plt.title('corr UNet')
plt.subplot(1, 4, 2)
plt.imshow(corr_LSTM, cmap='RdBu',vmin=-1, vmax=1)
plt.colorbar(shrink=0.5)
plt.title('corr LSTM')
plt.subplot(1, 4, 3)
plt.imshow(corr_UNetLSTM, cmap='RdBu',vmin=-1, vmax=1)
plt.colorbar(shrink=0.5)
plt.title('corr UNetLSTM')
plt.subplot(1, 4, 4)
plt.imshow(corr_LSTMUNet, cmap='RdBu',vmin=-1, vmax=1)
plt.colorbar(shrink=0.5)
plt.title('corr LSTMUNet')
plt.tight_layout
plt.savefig(r'%s\comparison_corr_models_to_target.png' % (training_folder))

plt.figure(figsize=(10,10))
plt.hist(corr_UNet.flatten(), bins=100, label='UNet', histtype='step', color='r')
plt.hist(corr_LSTM.flatten(), bins=100, label='LSTM', histtype='step', color='blue')
plt.hist(corr_UNetLSTM.flatten(), bins=100, label='UNetLSTM', histtype='step', color='orange')
plt.hist(corr_LSTMUNet.flatten(), bins=100, label='LSTMUNet', histtype='step', color='green')
plt.legend()
plt.savefig(r'%s\comparison_corr_models_to_target_hist.png' % (training_folder))


#pick a few random locations and plot time series from both target and models in different subplots while highlighting the location in one supblot with the map and dots for the location
import random
import matplotlib.pyplot as plt
import numpy as np

n = 5
n_lat = target.shape[1]
n_lon = target.shape[2]
rand_lat = random.sample(range(n_lat), n)
rand_lon = random.sample(range(n_lon), n)

plt.figure(figsize=(30, 5))
plt.subplot(1, 6, 1)
plt.imshow(target[0, :, :], cmap='grey')
plt.scatter(rand_lon, rand_lat, c='r', s=50)
# plt.figure(figsize=(13, 10))
for i in range(n):
    plt.subplot(1, 6, 2+i)
    plt.plot(target[:, rand_lat[i], rand_lon[i]], label='target', color='black')
    plt.plot(UNet[:, rand_lat[i], rand_lon[i]], label='UNet', color='red')
    plt.plot(LSTM[:, rand_lat[i], rand_lon[i]], label='LSTM', color='blue')
    plt.plot(UNetLSTM[:, rand_lat[i], rand_lon[i]], label='UNetLSTM', color='orange')
    plt.plot(LSTMUNet[:, rand_lat[i], rand_lon[i]], label='LSTMUNet', color='green')
plt.legend()



