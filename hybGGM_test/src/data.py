import glob
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


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

