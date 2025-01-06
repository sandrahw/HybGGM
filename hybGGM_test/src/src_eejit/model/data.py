import glob
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
def mask_data_0_1(path, lat_b, lon_b, save_path):
    '''create mask (for land/ocean)'''
    map_tile = xr.open_dataset(path)
    map_cut = map_tile.sel(lat=slice(*lat_b), lon=slice(*lon_b))
    mask = map_cut.to_array().values
    # mask where everything that is nan is 0 and everything else is 1
    mask = np.nan_to_num(mask, copy=False, nan=0)
    mask = np.where(mask==0, 0, 1)
    mask = mask[0, :, :]
    np.save('%s/mask.npy'%save_path, mask)
    return mask

def data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial):
    '''function to prepare the data for input by:
     - cropping the data to the specified lat and lon bounds for test regions, 
     - transforming the data to numpy arrays,
     - and additionally dealing with nan and inf values by setting them to 0 
     #TODO find a better way to deal with nan and inf values
    '''
    # print('file', f)
    param = f.split('/')[-1].split('.')[0]
    # print('param', param)
    data = xr.open_dataset(f)
    data_cut = data.sel(lat=slice(*lat_b), lon=slice(*lon_b))
    if param in params_monthly:
        data_arr = data_cut.to_array().values
        data_arr = data_arr[0, :, :, :] 
        data_arr = np.nan_to_num(data_arr, copy=False, nan=0)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)

    if param in params_initial: #repeat the initial values for each month as they are static
        data_arr = np.repeat(data_cut.to_array().values, data_l, axis=0)
        data_arr = np.where(data_arr==np.nan, 0, data_arr)
        data_arr = np.where(data_arr==np.inf, 0, data_arr)
    return data_arr

# load the different modflow files (#TODO: find a way to load all files at once)
def load_cut_data(inFiles, targetFile, lat_b, lon_b, data_l, params_monthly, params_initial):
    for f in inFiles:
        # print(f)
        if 'abstraction_lowermost_layer' in f:
            abs_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'abstraction_uppermost_layer' in f:  
            abs_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'bed_conductance_used' in f:
            bed_cond = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'bottom_lowermost_layer' in f:
            bottom_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'bottom_uppermost_layer' in f:
            bottom_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'drain_conductance' in f:
            drain_cond = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'drain_elevation_lowermost_layer' in f:
            drain_elev_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)   
        if 'drain_elevation_uppermost_layer' in f:
            drain_elev_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'horizontal_conductivity_lowermost_layer' in f:
            hor_cond_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'horizontal_conductivity_uppermost_layer' in f:
            hor_cond_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'initial_head_lowermost_layer' in f: 
            init_head_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'initial_head_uppermost_layer' in f: 
            init_head_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'net_RCH' in f:
            recharge = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'primary_storage_coefficient_lowermost_layer' in f:
            prim_stor_coeff_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'primary_storage_coefficient_uppermost_layer' in f:
            prim_stor_coeff_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'surface_water_bed_elevation_used' in f:
            surf_wat_bed_elev = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'surface_water_elevation' in f:
            surf_wat_elev = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'top_uppermost_layer' in f:
            top_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'vertical_conductivity_lowermost_layer' in f:
            vert_cond_lower = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
        if 'vertical_conductivity_uppermost_layer' in f:
            vert_cond_upper = data_prep(f, lat_b, lon_b, data_l, params_monthly, params_initial)
    wtd = data_prep(targetFile, lat_b, lon_b, data_l, params_monthly, params_initial)
    return abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper

def input_data_load(inputPath, targetPath, lat_b, lon_b, data_l, save_path, mask):
    '''load the modflow files and prepare the data for input'''
    inFiles = glob.glob('%s/*.nc' %inputPath) #load all input files in the folder
    # print(inFiles)
    # modflow files that are saved monthly
    params_monthly = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
                      'bed_conductance_used', 
                      'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
                      'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
                      'surface_water_bed_elevation_used',
                      'surface_water_elevation', 'net_RCH', 'wtd']
    # other modflow files that seem to be static parameters
    params_initial = ['bottom_lowermost_layer', 'bottom_uppermost_layer', 
                      'drain_conductance', 
                      'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
                      'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
                      'top_uppermost_layer',
                      'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer']

    abs_lower, abs_upper, bed_cond, bottom_lower, bottom_upper, drain_cond, drain_elev_lower, drain_elev_upper, hor_cond_lower, hor_cond_upper, init_head_lower, init_head_upper, recharge, prim_stor_coeff_lower, prim_stor_coeff_upper, surf_wat_bed_elev, surf_wat_elev, wtd, top_upper, vert_cond_lower, vert_cond_upper = load_cut_data(inFiles, targetPath, lat_b, lon_b, data_l, params_monthly, params_initial)

    ''''calculate the delta wtd for each month - define target (y) and input (X) arrays for the CNN'''
    delta_wtd = np.diff(wtd, axis=0) #wtd is always for end of the month so here for example delta wtd between jan and feb means the delta for feb
    # define target (y) and input (X) arrays for the CNN
    y = delta_wtd[:, np.newaxis, :, :] #shape[timesteps, channels, lat, lon]
    X_All = np.stack([abs_lower, abs_upper, 
                bed_cond, 
                bottom_lower, bottom_upper, 
                drain_cond, drain_elev_lower, drain_elev_upper, 
                hor_cond_lower, hor_cond_upper, 
                init_head_lower, init_head_upper, 
                recharge, 
                prim_stor_coeff_lower, prim_stor_coeff_upper, 
                surf_wat_bed_elev, surf_wat_elev, 
                top_upper, 
                vert_cond_lower, vert_cond_upper, #vert_cond_lower has inf values (for the test case of CH -> in prep fct fill with 0 )
                wtd, 
                mask
                ], axis=1)
    X = X_All[1:,:,:,:] #remove first month to match the delta wtd data
    np.save('%s/X.npy'%save_path, X)
    np.save('%s/y.npy'%save_path, y)
    return X, y, inFiles

def normalize(X, y, save_path):
    '''normalising the data for every array and save mean and std for denormalisation'''
    inp_var_mean = [] # list to store normalisation information for denormalisation later
    inp_var_std = []
    X_norm = []
    for i in range(X.shape[1]):
        mean = X[:, i, :, :].mean()
        std = X[:, i, :, :].std()
        # check if every value in array is 0, if so, skip normalisation
        if X[:, i, :, :].max() == 0 and X[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            X_temp = X[:, i, :, :]
        else:
            X_temp = (X[:, i, :, :] - mean) / std
        # print(mean, std, X_temp)
        X_norm.append(X_temp)
        inp_var_mean.append(mean)
        inp_var_std.append(std)
    X_norm_arr = np.array(X_norm)
    X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3)
    np.save('%s/X_norm_arr.npy'%save_path, X_norm_arr)
    np.save('%s/inp_var_mean.npy'%save_path, inp_var_mean)
    np.save('%s/inp_var_std.npy'%save_path, inp_var_std)

    out_var_mean = []
    out_var_std = []
    y_norm = []
    for i in range(y.shape[1]):
        mean = y[:, i, :, :].mean()
        std = y[:, i, :, :].std()
        # check if every value in array is 0, if so, skip normalisation
        if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            y_temp = X[:, i, :, :]
        else:
            y_temp = (X[:, i, :, :] - mean) / std
        y_temp = (y[:, i, :, :] - mean) / std
        y_norm.append(y_temp)
        out_var_mean.append(mean)
        out_var_std.append(std)
    y_norm_arr = np.array(y_norm)
    y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    np.save('%s/y_norm_arr.npy'%save_path, y_norm_arr)
    np.save('%s/out_var_mean.npy'%save_path, out_var_mean)
    np.save('%s/out_var_std.npy'%save_path, out_var_std)
    return X_norm_arr, y_norm_arr

'''remove mask in every training, validation and test patch'''
def remove_mask_patch(r):
    r_train = r[:, :-1, :, :]
    r_mask = r[:, -1, :, :]
    r_mask = np.where(r_mask<=0, 0, 1)
    mask_bool = r_mask.astype(bool)
    mask_bool = mask_bool[:, np.newaxis, :, :]
    return r_train, mask_bool

'''transform the data into tensors'''
def transformArrayToTensor(array):
    return torch.from_numpy(array).float()

class CustomDataset(Dataset):
    def __init__(self, data, labels, masks, transform=None):
        """
        Args:
            data (torch.Tensor or numpy array): Input data (e.g., images).
            labels (torch.Tensor or numpy array): Corresponding labels for the input data.
            masks (torch.Tensor or numpy array): Masks corresponding to each input data.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., for data augmentation).
        """
        self.data = data
        self.labels = labels
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input data, label, and mask for the given index
        input_data = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        # Apply any transformations if specified
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, label, mask

def train_val_test_split(testsize, trainsize, X_norm_arr, y_norm_arr, batchSize, save_path):
    '''split the patches into training, validation and test sets'''
    X_train, X_val_test = train_test_split(X_norm_arr, test_size=1-trainsize, random_state=10)
    X_val, X_test = train_test_split(X_val_test, test_size=1-testsize, random_state=10)  

    y_train, y_val_test = train_test_split(y_norm_arr, test_size=1-trainsize, random_state=10)
    y_val, y_test = train_test_split(y_val_test, test_size=1-testsize, random_state=10) 

    X_train, mask_train = remove_mask_patch(X_train)
    X_val, mask_val = remove_mask_patch(X_val)
    X_test, mask_test = remove_mask_patch(X_test)
    np.save('%s/X_train.npy' %save_path, X_train)
    np.save('%s/X_val.npy' %save_path, X_val)
    np.save('%s/X_test.npy' %save_path, X_test)

    np.save('%s/mask_train.npy' %save_path, mask_train)
    np.save('%s/mask_val.npy' %save_path, mask_val)
    np.save('%s/mask_test.npy' %save_path, mask_test)

    np.save('%s/y_train.npy' %save_path, y_train)
    np.save('%s/y_val.npy' %save_path, y_val)
    np.save('%s/y_test.npy' %save_path, y_test)

    train_loader = DataLoader(CustomDataset(X_train, y_train, mask_train), batch_size=batchSize, shuffle=False)
    validation_loader = DataLoader(CustomDataset(X_val, y_val, mask_val), batch_size=batchSize, shuffle=False)
    test_loader = DataLoader(CustomDataset(X_test, y_test, mask_test), batch_size=batchSize, shuffle=False)

    return train_loader, validation_loader, test_loader


''''random samples prep'''
def RS_cnn_sample_prep(sample, cnn_sample_path, log_dir, params_mf):
    print(f'sample: {sample}')
    sample_arrays = []
    for param in params_mf[:]:
        # print(f'param: {param}')
        samples_load = np.load('%s/%s_array_%s.npy' % (cnn_sample_path, sample, param))
        # print(samples_load.shape)
        #check if sample has nan or inf values and replace them with 0
        if param == 'globgm-wtd':
            # print('create delta wtd')
            t_0 = samples_load[:,:1,0, :, :]
            t_min1 = samples_load[:,1:,0,:, :]
            target = t_0 - t_min1 #this is the delta wtd
            #include only the previous timestep for wtd info in input data
            # samples_load_sel = target #original testing had delta wtd also as input
            samples_load_sel = samples_load[:,1:,0, :, :]  #extra test run with wtd as input as well 
            # print(param, samples_load_sel.shape)
           
            # print(param, samples_load_sel.shape) 
            # print('create mask')
            # print(target)
            mask = np.nan_to_num(target, copy=False, nan=0)
            mask = np.where(mask==0, 0, 1)
            # print(mask)
            mask_bool = mask.astype(bool)
            # print(mask_bool)
        else:
            #include only the previous timestep
            samples_load_sel = samples_load[:,1:,:]
            # print(param, samples_load_sel.shape)

        if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
            print(f'nan or inf values in {sample} {param}')
            samples_load_sel = np.nan_to_num(samples_load_sel, copy=False, nan=0)
            samples_load_sel = np.where(samples_load_sel==np.nan, 0, samples_load_sel)
            samples_load_sel = np.where(samples_load_sel==np.inf, 0, samples_load_sel)
            if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                print(f'nan or inf values STILL in {sample} {param}')

        sample_arrays.append(samples_load_sel)
    # stack the arrays together
    
    sample_arrays = np.stack(sample_arrays, axis=1)
    print('sample arrays shape', sample_arrays.shape)
    sys.stdout.flush()

    np.save('%s/input_%s.npy' %(log_dir, sample), sample_arrays)
    np.save('%s/target_%s.npy' %(log_dir, sample), target)
    np.save('%s/mask_%s.npy' %(log_dir, sample), mask_bool)
    return sample_arrays, target, mask_bool


def RS_normalize_samples_X(data, inptrain, inpval, inptest, sample, log_dir):
    inp_var_mean = [] # list to store normalisation information for denormalisation later
    inp_var_std = []
    X_norm = []
    X = data
    # print('X nan', np.isnan(X).any())
    # print('X inf', np.isinf(X).any())
    # print('X', X)
    for i in range(X.shape[1])[:]:
        print('X shape', X.shape)
        # sys.stdout.flush()
        # gather the same layer from train, val and test set and combine to one array
        temp = np.concatenate((inptrain[:, i, :, :, :], inpval[:, i,:, :, :], inptest[:, i, :, :, :]), axis=0)
        mean = temp.mean()
        std = temp.std()
        # print('mean', mean, 'std', std)
        # print('X max', temp.max())
        # print('X min', temp.min())
        # check if every value in array is 0, if so, skip normalisation
        if X[:, i, :, :, :].max() == 0 and X[:, i, :, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            X_temp = X[:, i,:, :, :]
        else:
            X_temp = (X[:, i, :, :, :] - mean) / std
        if np.isnan(X_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(X_temp, copy=False, nan=0)

      
        # print(mean, std, X_temp)
        X_norm.append(X_temp)
        inp_var_mean.append(mean)
        inp_var_std.append(std)

    #from list to array
    X_norm_arr = np.array(X_norm)
    print('X norm shape before shaping', X_norm_arr.shape)
    sys.stdout.flush()
    X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3, 4)
    print('X norm shape after shapine', X_norm_arr.shape)
    sys.stdout.flush()
    np.save('%s/inp_%s_norm_arr.npy'%(log_dir, sample), X_norm_arr)
    np.save('%s/inp_%s_var_mean.npy'%(log_dir, sample), inp_var_mean)
    np.save('%s/inp_%s_var_std.npy'%(log_dir, sample), inp_var_std)

    return X_norm_arr, inp_var_mean, inp_var_std

def RS_normalize_samples_y(data, tartrain, tarval, tartest, sample, log_dir):
    out_var_mean = []
    out_var_std = []
    y_norm = []
    y = data
    # print('X nan', np.isnan(y).any())
    # print('X inf', np.isinf(y).any())
    for i in range(y.shape[1]):
        print('y shape', y.shape)
        # gather the same layer from train, val and test set and combine to one array
        temp = np.concatenate((tartrain[:, i, :, :], tarval[:, i, :, :], tartest[:, i,  :, :]), axis=0)
        mean = temp.mean()
        std = temp.std()
        # print('mean', mean, 'std', std)
        # print('X max', temp.max())
        # print('X min', temp.min())
        # check if every value in array is 0, if so, skip normalisation
        if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            y_temp = y[:, i, :, :, :]
        else:
            y_temp = (y[:, i, :, :] - mean) / std

        if np.isnan(y_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(y_temp, copy=False, nan=0)


        y_norm.append(y_temp)
        out_var_mean.append(mean)
        out_var_std.append(std)
    y_norm_arr = np.array(y_norm)
    print('y norm shape before shaping', y_norm_arr.shape)
    sys.stdout.flush()
    y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    print('y norm shape after shapine', y_norm_arr.shape)
    sys.stdout.flush()
    np.save('%s/tar_%s_norm_arr.npy'%(log_dir, sample), y_norm_arr)
    np.save('%s/out_%s_var_mean.npy'%(log_dir, sample), out_var_mean)
    np.save('%s/out_%s_var_std.npy'%(log_dir, sample), out_var_std)
    return y_norm_arr, out_var_mean, out_var_std

def RS_downsize_data(data, target, mask, perc):
    data_down = data[:int(data.shape[0]*perc)]
    target_down = target[:int(target.shape[0]*perc)]
    mask_down = mask[:int(mask.shape[0]*perc)]
    return data_down, target_down, mask_down

def RS_dataprep(cnn_sample_path, testing_path, samplesize, batch_size):
    '''load the modflow files and prepare the data for input'''
    params_modflow = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
                       'bed_conductance_used', 
                        'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
                        'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
                        'surface_water_bed_elevation_used',
                        'surface_water_elevation', 'net_RCH', 
                        'bottom_lowermost_layer', 'bottom_uppermost_layer', 
                        'drain_conductance', 
                        'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
                        'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
                        'top_uppermost_layer',
                        'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer',
                        'globgm-wtd']

    input_train, target_train, mask_train = RS_cnn_sample_prep('training', cnn_sample_path, testing_path, params_modflow)
    input_val, target_val, mask_val = RS_cnn_sample_prep('validation', cnn_sample_path, testing_path, params_modflow)
    input_test, target_test, mask_test = RS_cnn_sample_prep('testing', cnn_sample_path, testing_path, params_modflow)

    input_train_norm, inp_var_train_mean, inp_var_train_std = RS_normalize_samples_X(input_train, input_train, input_val, input_test, 'training', testing_path)
    input_val_norm, inp_var_val_mean, inp_var_val_std = RS_normalize_samples_X(input_val,  input_train, input_val, input_test, 'validation', testing_path)
    input_test_norm, inp_var_test_mean, inp_var_test_std = RS_normalize_samples_X(input_test,  input_train, input_val, input_test, 'testing', testing_path)

    target_train_norm, out_var_train_mean, out_var_train_std = RS_normalize_samples_y(target_train, target_train, target_val, target_test,'training', testing_path)
    target_val_norm, out_var_val_mean, out_var_val_std = RS_normalize_samples_y(target_val, target_train, target_val, target_test, 'validation', testing_path)
    target_test_norm, out_var_test_mean, out_var_test_std = RS_normalize_samples_y(target_test,  target_train, target_val, target_test,'testing', testing_path)

    input_train_norm = input_train_norm[:,:,0,:,:]
    input_val_norm = input_val_norm[:,:,0,:,:]
    input_test_norm = input_test_norm[:,:,0,:,:]

    input_train_norm_down, target_train_norm_down, mask_train_down = RS_downsize_data(input_train_norm, target_train_norm, mask_train, samplesize)
    input_val_norm_down, target_val_norm_down, mask_val_down = RS_downsize_data(input_val_norm, target_val_norm, mask_val, samplesize)
    input_test_norm_down, target_test_norm_down, mask_test_down = RS_downsize_data(input_test_norm, target_test_norm, mask_test, samplesize)

    train_loader = DataLoader(CustomDataset(input_train_norm_down, target_train_norm_down, mask_train_down), batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(CustomDataset(input_val_norm_down, target_val_norm_down, mask_val_down), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CustomDataset(input_test_norm_down, target_test_norm_down, mask_test_down), batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

def RS_reload_data(log_dir, samplesize, batch_size):
    input_train_norm = np.load('%s/inp_training_norm_arr.npy' %log_dir)
    target_train_norm = np.load('%s/tar_training_norm_arr.npy' %log_dir)
    mask_train = np.load('%s/mask_training.npy' %log_dir)   
    input_val_norm = np.load('%s/inp_validation_norm_arr.npy' %log_dir)
    target_val_norm = np.load('%s/tar_validation_norm_arr.npy' %log_dir)
    mask_val = np.load('%s/mask_validation.npy' %log_dir)
    input_test_norm = np.load('%s/inp_testing_norm_arr.npy' %log_dir)
    target_test_norm = np.load('%s/tar_testing_norm_arr.npy' %log_dir)
    mask_test = np.load('%s/mask_testing.npy' %log_dir)

    input_train_norm_down, target_train_norm_down, mask_train_down = RS_downsize_data(input_train_norm, target_train_norm, mask_train, samplesize)
    input_val_norm_down, target_val_norm_down, mask_val_down = RS_downsize_data(input_val_norm, target_val_norm, mask_val, samplesize)
    input_test_norm_down, target_test_norm_down, mask_test_down = RS_downsize_data(input_test_norm, target_test_norm, mask_test, samplesize)

    train_loader = DataLoader(CustomDataset(input_train_norm_down, target_train_norm_down, mask_train_down), batch_size=batch_size, shuffle=False)
    print('train loader check', next(iter(train_loader))[0].shape)
    sys.stdout.flush()
    validation_loader = DataLoader(CustomDataset(input_val_norm_down, target_val_norm_down, mask_val_down), batch_size=batch_size, shuffle=False)
    print('val loader check', next(iter(validation_loader))[0].shape)
    sys.stdout.flush()
    test_loader = DataLoader(CustomDataset(input_test_norm_down, target_test_norm_down, mask_test_down), batch_size=batch_size, shuffle=False)
    print('test loader check', next(iter(test_loader))[0].shape)
    return train_loader, validation_loader, test_loader



'''full random data prep'''
def RS_cnn_full_sample_prep(cnn_sample_path, log_dir, params_mf):
    sample_arrays = []
    for param in params_mf[:]:
        print(f'param: {param}, %s/full_array_%s.npy' % (cnn_sample_path, param))
        samples_load = np.load('%s/full_array_%s.npy' % (cnn_sample_path, param))
        # print(samples_load.shape)
        #check if sample has nan or inf values and replace them with 0
        if param == 'globgm-wtd':
            print('create delta wtd')
            t_0 = samples_load[:,:1,0, :, :]
            t_min1 = samples_load[:,1:,0,:, :]
            target = t_0 - t_min1 #this is the delta wtd
            #include only the previous timestep for wtd info in input data
            # samples_load_sel = target #original testing had delta wtd also as input
            samples_load_sel = samples_load[:,1:,0, :, :]  #extra test run with wtd as input as well 
            print(param, samples_load_sel.shape) 
            print('create mask')
            # print(target)
            mask = np.nan_to_num(target, copy=False, nan=0)
            mask = np.where(mask==0, 0, 1)
            # print(mask)
            mask_bool = mask.astype(bool)
            # print(mask_bool)
        else:
            #include only the previous timestep
            samples_load_sel = samples_load[:,1:,:]
            print(param, samples_load_sel.shape)

        if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
            print(f'nan or inf values in {param}')
            samples_load_sel = np.nan_to_num(samples_load_sel, copy=False, nan=0)
            samples_load_sel = np.where(samples_load_sel==np.nan, 0, samples_load_sel)
            samples_load_sel = np.where(samples_load_sel==np.inf, 0, samples_load_sel)
            if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                print(f'nan or inf values STILL in {param}')

        sample_arrays.append(samples_load_sel)
    # stack the arrays together
    sample_arrays = np.stack(sample_arrays, axis=1)
   

    np.save('%s/full_input.npy' %(log_dir), sample_arrays)
    np.save('%s/full_target.npy' %(log_dir), target)
    np.save('%s/full_mask.npy' %(log_dir), mask_bool)
    return sample_arrays, target, mask_bool

def RS_normalize_full_samples_X(data, log_dir):
    inp_var_mean = [] # list to store normalisation information for denormalisation later
    inp_var_std = []
    X_norm = []
    X = data
    # print('X nan', np.isnan(X).any())
    # print('X inf', np.isinf(X).any())
    # print('X', X)
    for i in range(X.shape[1])[:]:
        print(i)
        # gather the same layer from train, val and test set and combine to one array
        mean = X[:, i, :, :, :].mean()
        std = X[:, i, :, :, :].std()
        # print('mean', mean, 'std', std)
        # print('X max', temp.max())
        # print('X min', temp.min())
        # check if every value in array is 0, if so, skip normalisation
        if X[:, i, :, :, :].max() == 0 and X[:, i, :, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            X_temp = X[:, i,:, :, :]
        else:
            X_temp = (X[:, i, :, :, :] - mean) / std
        if np.isnan(X_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(X_temp, copy=False, nan=0)

      
        # print(mean, std, X_temp)
        X_norm.append(X_temp)
        inp_var_mean.append(mean)
        inp_var_std.append(std)

    #from list to array
    X_norm_arr = np.array(X_norm)
    X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3, 4)
    np.save('%s/full_inp_norm_arr.npy'%(log_dir), X_norm_arr)
    np.save('%s/full_inp_var_mean.npy'%(log_dir), inp_var_mean)
    np.save('%s/full_inp_var_std.npy'%(log_dir), inp_var_std)

    return X_norm_arr, inp_var_mean, inp_var_std

def RS_normalize_full_samples_y(data, log_dir):
    out_var_mean = []
    out_var_std = []
    y_norm = []
    y = data
    # print('X nan', np.isnan(y).any())
    # print('X inf', np.isinf(y).any())
    for i in range(y.shape[1]):
        # gather the same layer from train, val and test set and combine to one array
        mean = y[:, i, :, :].mean()
        std = y[:, i, :, :].std()
        # print('mean', mean, 'std', std)
        # print('X max', temp.max())
        # print('X min', temp.min())
        # check if every value in array is 0, if so, skip normalisation
        if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            y_temp = y[:, i, :, :, :]
        else:
            y_temp = (y[:, i, :, :] - mean) / std

        if np.isnan(y_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(y_temp, copy=False, nan=0)


        y_norm.append(y_temp)
        out_var_mean.append(mean)
        out_var_std.append(std)
    y_norm_arr = np.array(y_norm)
    y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    np.save('%s/full_tar_norm_arr.npy'%(log_dir), y_norm_arr)
    np.save('%s/full_out_var_mean.npy'%(log_dir), out_var_mean)
    np.save('%s/full_out_var_std.npy'%(log_dir), out_var_std)
    return y_norm_arr, out_var_mean, out_var_std


def RS_full_dataprep(cnn_sample_path, testing_path, batch_size):
    '''load the modflow files and prepare the data for input'''
    params_modflow = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
                       'bed_conductance_used', 
                        'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
                        'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
                        'surface_water_bed_elevation_used',
                        'surface_water_elevation', 'net_RCH', 
                        'bottom_lowermost_layer', 'bottom_uppermost_layer', 
                        'drain_conductance', 
                        'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
                        'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
                        'top_uppermost_layer',
                        'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer',
                        'globgm-wtd']
    print('defined params modflow')
    print('starting data prep')
    input, target, mask = RS_cnn_full_sample_prep(cnn_sample_path, testing_path, params_modflow)
    input_norm, inp_var_mean, inp_var_std = RS_normalize_full_samples_X(input, testing_path)
    target_norm, out_var_mean, out_var_std = RS_normalize_full_samples_y(target, testing_path)
    input_norm = input_norm[:,:,0,:,:]

    full_run_loader = DataLoader(CustomDataset(input_norm, target_norm, mask), batch_size=batch_size, shuffle=False)

    return full_run_loader





''''random samples prep wtd'''
def RS_cnn_sample_prep_target_wtd(sample, params_mf,  cnn_sample_path, log_dir):
    print(f'sample: {sample}')
    sample_arrays = []
    for param in params_mf[:]:
        samples_load = np.load('%s/%s_array_%s.npy' % (cnn_sample_path, sample, param))
        if param == 'globgm-wtd':
            # print('create delta wtd')
            t_0 = samples_load[:,:1,0, :, :] #wtd
            t_min1 = samples_load[:,1:,0,:, :] #wtd
            target = t_0 #this is the delta wtd
            print('target shape', target.shape) 
            samples_load_sel = t_min1 #
            print(param, samples_load_sel.shape)
            mask = np.nan_to_num(target, copy=False, nan=0)
            mask = np.where(mask==0, 0, 1)
            mask_bool = mask.astype(bool)

        else:
            #every other parameter
            samples_load_sel = samples_load[:,1:,:]
            print(param, samples_load_sel.shape)

        if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
            print(f'nan or inf values in {sample} {param}')
            samples_load_sel = np.nan_to_num(samples_load_sel, copy=False, nan=0)
            samples_load_sel = np.where(samples_load_sel==np.nan, 0, samples_load_sel)
            samples_load_sel = np.where(samples_load_sel==np.inf, 0, samples_load_sel)
            if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                print(f'nan or inf values STILL in {sample} {param}')

        sample_arrays.append(samples_load_sel)
    # stack the arrays together
    
    sample_arrays = np.stack(sample_arrays, axis=1)
    print('sample arrays shape', sample_arrays.shape)
    sys.stdout.flush()

    np.save('%s/input_%s.npy' %(log_dir, sample), sample_arrays)
    np.save('%s/target_%s.npy' %(log_dir, sample), target)
    np.save('%s/mask_%s.npy' %(log_dir, sample), mask_bool)
    return sample_arrays, target, mask_bool


def RS_normalize_samples_X_wtd(paramslimited, data, sample, log_dir):
    X_norm = []
    X = data
    variables = range(X.shape[1])
    # print('variables', variables)
    for i, param in zip(variables, paramslimited):
        print('X shape', X.shape)
        print('param', param, 'i', i)
        mean = np.load('%s/full_inp_var_mean_%s.npy' %(log_dir,param))
        std = np.load('%s/full_inp_var_std_%s.npy' %(log_dir,param))

        # check if every value in array is 0, if so, skip normalisation
        if X[:, i, :, :, :].max() == 0 and X[:, i, :, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            X_temp = X[:, i,:, :, :]
        else:
            X_temp = (X[:, i, :, :, :] - mean) / std
        if np.isnan(X_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(X_temp, copy=False, nan=0)      
        # print(mean, std, X_temp)
        X_norm.append(X_temp)

    #from list to array
    X_norm_arr = np.array(X_norm)
    # print('X norm shape before shaping', X_norm_arr.shape)
    # sys.stdout.flush()
    X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3, 4)
    # print('X norm shape after shapine', X_norm_arr.shape)
    # sys.stdout.flush()
    np.save('%s/inp_%s_norm_arr.npy'%(log_dir, sample), X_norm_arr)
 
    return X_norm_arr

def RS_normalize_samples_y_wtd(paramslimited, data, sample, log_dir):
    y_norm = []
    y = data
    mean = np.load('%s/full_inp_var_mean_globgm-wtd.npy'%(log_dir))
    std = np.load('%s/full_inp_var_std_globgm-wtd.npy'%(log_dir))
    for i in range(y.shape[1]):
        print('y shape', y.shape)
        # check if every value in array is 0, if so, skip normalisation
        if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            y_temp = y[:, i, :, :, :]
        else:
            y_temp = (y[:, i, :, :] - mean) / std

        if np.isnan(y_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            y_temp = np.nan_to_num(y_temp, copy=False, nan=0)
        y_norm.append(y_temp)

    y_norm_arr = np.array(y_norm)
    # print('y norm shape before shaping', y_norm_arr.shape)
    # sys.stdout.flush()
    y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    # print('y norm shape after shapine', y_norm_arr.shape)
    # sys.stdout.flush()
    np.save('%s/tar_%s_norm_arr.npy'%(log_dir, sample), y_norm_arr)
    return y_norm_arr

def RS_cnn_data_prep_target_wtd(cnn_sample_path, paramslimited, testing_path, batch_size):
    '''load the modflow files and prepare the data for input'''
    #check if training, testing, validation data already exists
    if os.path.exists('%s/input_training.npy' %testing_path):
        print('training, testing, val data already exists')
        sys.stdout.flush()
        input_train = np.load('%s/input_training.npy' %testing_path)
        target_train = np.load('%s/target_training.npy' %testing_path)
        mask_train = np.load('%s/mask_training.npy' %testing_path)
        input_val = np.load('%s/input_validation.npy' %testing_path)
        target_val = np.load('%s/target_validation.npy' %testing_path)
        mask_val = np.load('%s/mask_validation.npy' %testing_path)
        input_test = np.load('%s/input_testing.npy' %testing_path)
        target_test = np.load('%s/target_testing.npy' %testing_path)
        mask_test = np.load('%s/mask_testing.npy' %testing_path)
        print('data loaded')
        sys.stdout.flush()
    else:
        print('training, testing, val data does not exist') 
        sys.stdout.flush()
        input_train, target_train, mask_train = RS_cnn_sample_prep_target_wtd('training', paramslimited, cnn_sample_path, testing_path)
        input_val, target_val, mask_val = RS_cnn_sample_prep_target_wtd('validation', paramslimited, cnn_sample_path, testing_path)
        input_test, target_test, mask_test = RS_cnn_sample_prep_target_wtd('testing', paramslimited, cnn_sample_path, testing_path)

    print('normalising data')
    input_train_norm = RS_normalize_samples_X_wtd(paramslimited, input_train, 'training', testing_path)
    input_val_norm = RS_normalize_samples_X_wtd(paramslimited,  input_val, 'validation', testing_path)
    input_test_norm = RS_normalize_samples_X_wtd(paramslimited,  input_test, 'testing', testing_path)

    target_train_norm = RS_normalize_samples_y_wtd(paramslimited, target_train, 'training', testing_path)
    target_val_norm = RS_normalize_samples_y_wtd(paramslimited,  target_val, 'validation', testing_path)
    target_test_norm = RS_normalize_samples_y_wtd(paramslimited,  target_test,'testing', testing_path)

    input_train_norm = input_train_norm[:,:,0,:,:]
    input_val_norm = input_val_norm[:,:,0,:,:]
    input_test_norm = input_test_norm[:,:,0,:,:]
    print('creating dataloaders')
    train_loader = DataLoader(CustomDataset(input_train_norm, target_train_norm, mask_train), batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(CustomDataset(input_val_norm, target_val_norm, mask_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CustomDataset(input_test_norm, target_test_norm, mask_test), batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

def RS_reload_data_wtd(log_dir, batch_size):
    input_train_norm = np.load('%s/inp_training_norm_arr.npy' %log_dir)
    target_train_norm = np.load('%s/tar_training_norm_arr.npy' %log_dir)
    mask_train = np.load('%s/mask_training.npy' %log_dir)   
    input_val_norm = np.load('%s/inp_validation_norm_arr.npy' %log_dir)
    target_val_norm = np.load('%s/tar_validation_norm_arr.npy' %log_dir)
    mask_val = np.load('%s/mask_validation.npy' %log_dir)
    input_test_norm = np.load('%s/inp_testing_norm_arr.npy' %log_dir)
    target_test_norm = np.load('%s/tar_testing_norm_arr.npy' %log_dir)
    mask_test = np.load('%s/mask_testing.npy' %log_dir)

    train_loader = DataLoader(CustomDataset(input_train_norm, target_train_norm, mask_train), batch_size=batch_size, shuffle=False)
    print('train loader check', next(iter(train_loader))[0].shape)
    sys.stdout.flush()
    validation_loader = DataLoader(CustomDataset(input_val_norm, target_val_norm, mask_val), batch_size=batch_size, shuffle=False)
    print('val loader check', next(iter(validation_loader))[0].shape)
    sys.stdout.flush()
    test_loader = DataLoader(CustomDataset(input_test_norm, target_test_norm, mask_test), batch_size=batch_size, shuffle=False)
    print('test loader check', next(iter(test_loader))[0].shape)
    return train_loader, validation_loader, test_loader





''''random samples prep deltawtd'''
def RS_cnn_sample_prep_target_deltawtd(sample, params_mf,  cnn_sample_path, log_dir):
    print(f'sample: {sample}')
    sample_arrays = []
    for param in params_mf[:]:
        samples_load = np.load('%s/%s_array_%s.npy' % (cnn_sample_path, sample, param))
        if param == 'globgm-wtd':
            t_0 = samples_load[:,:1,0, :, :] #wtd
            t_min1 = samples_load[:,1:,0,:, :] #wtd
            target = t_0 - t_min1 #this is the delta wtd
            print('target shape', target.shape) 
            samples_load_sel = samples_load[:,1:,0, :, :]  #
            print(param, samples_load_sel.shape)
            mask = np.nan_to_num(target, copy=False, nan=0)
            mask = np.where(mask==0, 0, 1)
            mask_bool = mask.astype(bool)

        else:
            #every other parameter
            samples_load_sel = samples_load[:,1:,:]
            print(param, samples_load_sel.shape)

        if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
            print(f'nan or inf values in {sample} {param}')
            samples_load_sel = np.nan_to_num(samples_load_sel, copy=False, nan=0)
            samples_load_sel = np.where(samples_load_sel==np.nan, 0, samples_load_sel)
            samples_load_sel = np.where(samples_load_sel==np.inf, 0, samples_load_sel)
            if np.isnan(samples_load_sel).any() or np.isinf(samples_load_sel).any():
                print(f'nan or inf values STILL in {sample} {param}')

        sample_arrays.append(samples_load_sel)
    # stack the arrays together
    
    sample_arrays = np.stack(sample_arrays, axis=1)
    print('sample arrays shape', sample_arrays.shape)
    sys.stdout.flush()

    np.save('%s/input_%s.npy' %(log_dir, sample), sample_arrays)
    np.save('%s/target_%s.npy' %(log_dir, sample), target)
    np.save('%s/mask_%s.npy' %(log_dir, sample), mask_bool)
    return sample_arrays, target, mask_bool


def RS_normalize_samples_X_deltawtd(paramslimited, data, sample, log_dir):
    X_norm = []
    X = data
    variables = range(X.shape[1])
    # print('variables', variables)
    for i, param in zip(variables, paramslimited):
        print('X shape', X.shape)
        print('param', param, 'i', i)
        mean = np.load('%s/full_inp_var_mean_%s.npy' %(log_dir,param))
        std = np.load('%s/full_inp_var_std_%s.npy' %(log_dir, param))

        # check if every value in array is 0, if so, skip normalisation
        if X[:, i, :, :, :].max() == 0 and X[:, i, :, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            X_temp = X[:, i,:, :, :]
        else:
            X_temp = (X[:, i, :, :, :] - mean) / std
        if np.isnan(X_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            X_temp = np.nan_to_num(X_temp, copy=False, nan=0)      
        # print(mean, std, X_temp)
        X_norm.append(X_temp)

    #from list to array
    X_norm_arr = np.array(X_norm)
    # print('X norm shape before shaping', X_norm_arr.shape)
    # sys.stdout.flush()
    X_norm_arr = X_norm_arr.transpose(1, 0, 2, 3, 4)
    # print('X norm shape after shapine', X_norm_arr.shape)
    # sys.stdout.flush()
    np.save('%s/inp_%s_norm_arr.npy'%(log_dir, sample), X_norm_arr)
 
    return X_norm_arr

def RS_normalize_samples_y_deltawtd(paramslimited, data, sample, log_dir):
    y_norm = []
    y = data
    mean = np.load('%s/full_out_var_mean.npy'%log_dir)
    std = np.load('%s/full_out_var_std.npy'%log_dir)
    for i in range(y.shape[1]):
        print('y shape', y.shape)
        # check if every value in array is 0, if so, skip normalisation
        if y[:, i, :, :].max() == 0 and y[:, i, :, :].min() == 0:
            print('skipped normalisation for array %s' %i)
            y_temp = y[:, i, :, :, :]
        else:
            y_temp = (y[:, i, :, :] - mean) / std

        if np.isnan(y_temp).any():
            print('nan in array %s' %i)
            print(i, mean, std)
            print('replace nan values with 0')
            y_temp = np.nan_to_num(y_temp, copy=False, nan=0)
        y_norm.append(y_temp)

    y_norm_arr = np.array(y_norm)
    # print('y norm shape before shaping', y_norm_arr.shape)
    # sys.stdout.flush()
    y_norm_arr = y_norm_arr.transpose(1, 0, 2, 3)
    # print('y norm shape after shapine', y_norm_arr.shape)
    # sys.stdout.flush()
    np.save('%s/tar_%s_norm_arr.npy'%(log_dir, sample), y_norm_arr)
    return y_norm_arr

def RS_cnn_data_prep_target_deltawtd(cnn_sample_path, paramslimited, testing_path, batch_size):
    '''load the modflow files and prepare the data for input'''
    #check if training, testing, validation data already exists
    if os.path.exists('%s/input_training.npy' %testing_path):
        print('training, testing, val data already exists')
        sys.stdout.flush()
        input_train = np.load('%s/input_training.npy' %testing_path)
        target_train = np.load('%s/target_training.npy' %testing_path)
        mask_train = np.load('%s/mask_training.npy' %testing_path)
        input_val = np.load('%s/input_validation.npy' %testing_path)
        target_val = np.load('%s/target_validation.npy' %testing_path)
        mask_val = np.load('%s/mask_validation.npy' %testing_path)
        input_test = np.load('%s/input_testing.npy' %testing_path)
        target_test = np.load('%s/target_testing.npy' %testing_path)
        mask_test = np.load('%s/mask_testing.npy' %testing_path)
        print('data loaded')
        sys.stdout.flush()
    else:
        print('training, testing, val data does not exist') 
        sys.stdout.flush()
        input_train, target_train, mask_train = RS_cnn_sample_prep_target_deltawtd('training', paramslimited, cnn_sample_path, testing_path)
        input_val, target_val, mask_val = RS_cnn_sample_prep_target_deltawtd('validation', paramslimited, cnn_sample_path, testing_path)
        input_test, target_test, mask_test = RS_cnn_sample_prep_target_deltawtd('testing', paramslimited, cnn_sample_path, testing_path)

    print('normalising data')
    input_train_norm = RS_normalize_samples_X_deltawtd(paramslimited, input_train, 'training', testing_path)
    input_val_norm = RS_normalize_samples_X_deltawtd(paramslimited,  input_val, 'validation', testing_path)
    input_test_norm = RS_normalize_samples_X_deltawtd(paramslimited,  input_test, 'testing', testing_path)

    target_train_norm = RS_normalize_samples_y_deltawtd(paramslimited, target_train, 'training', testing_path)
    target_val_norm = RS_normalize_samples_y_deltawtd(paramslimited,  target_val, 'validation', testing_path)
    target_test_norm = RS_normalize_samples_y_deltawtd(paramslimited,  target_test,'testing', testing_path)

    input_train_norm = input_train_norm[:,:,0,:,:]
    input_val_norm = input_val_norm[:,:,0,:,:]
    input_test_norm = input_test_norm[:,:,0,:,:]
    print('creating dataloaders')
    train_loader = DataLoader(CustomDataset(input_train_norm, target_train_norm, mask_train), batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(CustomDataset(input_val_norm, target_val_norm, mask_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CustomDataset(input_test_norm, target_test_norm, mask_test), batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

def RS_reload_data_deltawtd(log_dir, batch_size):
    input_train_norm = np.load('%s/inp_training_norm_arr.npy' %log_dir)
    target_train_norm = np.load('%s/tar_training_norm_arr.npy' %log_dir)
    mask_train = np.load('%s/mask_training.npy' %log_dir)   
    input_val_norm = np.load('%s/inp_validation_norm_arr.npy' %log_dir)
    target_val_norm = np.load('%s/tar_validation_norm_arr.npy' %log_dir)
    mask_val = np.load('%s/mask_validation.npy' %log_dir)
    input_test_norm = np.load('%s/inp_testing_norm_arr.npy' %log_dir)
    target_test_norm = np.load('%s/tar_testing_norm_arr.npy' %log_dir)
    mask_test = np.load('%s/mask_testing.npy' %log_dir)

    train_loader = DataLoader(CustomDataset(input_train_norm, target_train_norm, mask_train), batch_size=batch_size, shuffle=False)
    print('train loader check', next(iter(train_loader))[0].shape)
    sys.stdout.flush()
    validation_loader = DataLoader(CustomDataset(input_val_norm, target_val_norm, mask_val), batch_size=batch_size, shuffle=False)
    print('val loader check', next(iter(validation_loader))[0].shape)
    sys.stdout.flush()
    test_loader = DataLoader(CustomDataset(input_test_norm, target_test_norm, mask_test), batch_size=batch_size, shuffle=False)
    print('test loader check', next(iter(test_loader))[0].shape)
    return train_loader, validation_loader, test_loader

