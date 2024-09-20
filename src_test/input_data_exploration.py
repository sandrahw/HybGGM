#exploring the input data separately
import pandas as pd 
import glob
import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
random.seed(10)
print(random.random())

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

# lon_bounds = (5, 10) #CH bounds(5,10)
# lat_bounds = (45, 50)#CH bounds(45,50)

lon_bounds = (3,6) #NL bounds(3,6)
lat_bounds = (50,54)#NL bounds(50,54)

#create mask (for land/ocean)
map_tile = xr.open_dataset(r'..\data\temp\wtd.nc')
map_cut = map_tile.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
mask = map_cut.to_array().values
# mask = map_tile.to_array().values
# mask where everything that is nan is 0 and everything else is 1
mask = np.nan_to_num(mask, copy=False, nan=0)
mask = np.where(mask==0, 0, 1)
mask = mask[0, :, :]
mask = np.flip(mask, axis=1)
mask_na = np.where(mask==0, np.nan, 1)



inFiles = glob.glob(r'..\data\temp\*.nc') #load all input files in the folder

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


for file in inFiles[-3:-2]:
    print(file)
    param = file.split('\\')[-1].split('.')[0]

    data = xr.open_dataset(file)
    data_cut = data.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
    # data_cut = data.copy()
    data_array = data_cut.to_array().values
    if param in params_monthly:
        data_array_flipped = np.flip(data_array, axis=2)
    else:
        data_array_flipped = np.flip(data_array, axis=1)


    # plot mask, data and distribution of data
    plt.figure(figsize=(40, 40))
    plt.rcParams['font.size'] = '30'

    ax1= plt.subplot(3, 2, 5)
    if np.isinf(data_array[0, :, :]).any():
        mask_inf = np.where(data_array[0, :, :].flatten()==np.inf, 0, 1)
        mask_inf = mask_inf.reshape(data_array[0, :, :].shape)
        mask_flipped = np.flip(mask_inf, axis=0)
        plt.imshow(mask_flipped)
        plt.title('inf values in data', fontsize=30)
        plt.colorbar(shrink=0.8,  ticks=[0, 1])
        plt.tight_layout()
    else:   
        plt.imshow(mask[0, :, :])
        plt.title('ocean/land mask', fontsize=30)
        plt.colorbar(shrink=0.8,  ticks=[0, 1])
        plt.tight_layout()

    ax2=plt.subplot(3, 2, 1)
    if param in params_monthly:
        plt.imshow(data_array_flipped[0,0, :, :])
    else:
        plt.imshow(data_array_flipped[0, :, :])
    plt.title('%s' %param, fontsize=30)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()

    ax3=plt.subplot(3, 2, 3)
    if param in params_monthly:
        plt.imshow(data_array_flipped[0,0, :, :]*mask_na[0, :, :])
    else:
        plt.imshow(data_array_flipped[0, :, :]*mask_na[0, :, :])
    plt.title('%s \n masked' %param, fontsize=30)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()

    ax4=plt.subplot(3, 2, 2)
    data_array = data_cut.to_array().values
    if param in params_monthly:
        data_array_flipped = np.flip(data_array, axis=2)
        data_min = np.nanmin(data_array_flipped[0,0, :, :].flatten())
        data_max = np.nanmax(data_array_flipped[0,0, :, :].flatten())
        perc_nan_data = np.sum(np.isnan(data_array_flipped[0,0, :, :].flatten()))/len(data_array_flipped[0,0, :, :].flatten())
        perc_inf_data = np.sum(data_array_flipped[0,0, :, :].flatten()==np.inf)/len(data_array_flipped[0,0, :, :].flatten())
        perc_zero_data = np.sum(data_array_flipped[0,0, :, :].flatten()==0)/len(data_array_flipped[0,0, :, :].flatten())
        plt.hist(data_array_flipped[0,0, :, :].flatten(), bins=100)
    else:
        data_array_flipped = np.flip(data_array, axis=1)
        perc_nan_data = np.sum(np.isnan(data_array_flipped[0, :, :].flatten()))/len(data_array_flipped[0, :, :].flatten())
        perc_inf_data = np.sum(data_array_flipped[0, :, :].flatten()==np.inf)/len(data_array_flipped[0, :, :].flatten())
        perc_zero_data = np.sum(data_array_flipped[0, :, :].flatten()==0)/len(data_array_flipped[0, :, :].flatten())
        data_min = np.nanmin(data_array_flipped[0, :, :].flatten())
        data_max = np.nanmax(data_array_flipped[0, :, :].flatten())
        if data_max == np.inf:
            #replace inf with nan value 
            data_array_flipped[0, :, :][data_array_flipped[0, :, :]==np.inf] = np.nan
        plt.hist(data_array_flipped[0, :, :].flatten(), bins=100)      
    ax4.text(0.3, 0.8, 'min: %.3f, max: %.3f,\nperc nan: %.3f, perc inf: %.3f\n perc zeros: %.3f' %(data_min, data_max, perc_nan_data, perc_inf_data, perc_zero_data), horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes) 
    plt.title('distribution of data (1 month)', fontsize=30)
    plt.tight_layout()


    ax5=plt.subplot(3, 2, 4)
    data_array = data_cut.to_array().values
    if param in params_monthly:
        data_array_flipped = np.flip(data_array, axis=2)
        data_min = np.nanmin(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten())
        data_max = np.nanmax(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_nan_data = np.sum(np.isnan(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten()))/len(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_inf_data = np.sum(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten()==np.inf)/len(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_zero_data = np.sum(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten()==0)/len(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten())
        plt.hist(data_array_flipped[0,0, :, :].flatten()*mask_na[0, :, :].flatten(), bins=100)
    else:
        data_array_flipped = np.flip(data_array, axis=1)
        data_min = np.nanmin(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten())
        data_max = np.nanmax(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_nan_data = np.sum(np.isnan(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten()))/len(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_inf_data = np.sum(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten()==np.inf)/len(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_zero_data = np.sum(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten()==0)/len(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten())
        if data_max == np.inf:
            #replace inf with nan value 
            data_array_flipped[0, :, :][data_array_flipped[0, :, :]==np.inf] = np.nan
        plt.hist(data_array_flipped[0, :, :].flatten()*mask_na[0, :, :].flatten(), bins=100)
    ax5.text(0.3, 0.8, 'min: %.3f, max: %.3f,\nperc nan: %.3f, perc inf: %.3f\n perc zeros: %.3f' %(data_min, data_max, perc_nan_data, perc_inf_data, perc_zero_data), horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
    plt.title('distribution of masked data (1 month)', fontsize=30)
    plt.tight_layout()
    ax5.sharex(ax4)
    ax5.sharey(ax4)


    ax6=plt.subplot(3, 2, 6)
    data_array = data_cut.to_array().values
    if param in params_monthly:
        data_array_flipped = np.flip(data_array, axis=2)
        data_min = np.nanmin(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten())
        data_max = np.nanmax(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten())
        perc_nan_data = np.sum(np.isnan(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten()))/len(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten())
        perc_inf_data = np.sum(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten()==np.inf)/len(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten())
        perc_zero_data = np.sum(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten()==0)/len(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten()) 
        plt.hist(data_array[:,:, :, :].flatten()*mask_na[:, :, :].flatten(), bins=100) 
    else:
        data_array_flipped = np.flip(data_array, axis=1)
        data_min = np.nanmin(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten())
        data_max = np.nanmax(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_nan_data = np.sum(np.isnan(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten()))/len(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_inf_data = np.sum(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten()==np.inf)/len(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten())
        perc_zero_data = np.sum(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten()==0)/len(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten())
        if data_max == np.inf:
            #replace inf with nan value 
            data_array[0, :, :][data_array[0, :, :]==np.inf] = np.nan
        plt.hist(data_array[0, :, :].flatten()*mask_na[0, :, :].flatten(), bins=100)      
    ax6.text(0.3, 0.8, 'min: %.3f, max: %.3f,\nperc nan: %.3f, perc inf: %.3f\n perc zeros: %.3f' %(data_min, data_max, perc_nan_data, perc_inf_data, perc_zero_data), horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
    plt.title('distribution of masked data all months', fontsize=30)
    plt.tight_layout()

    plt.savefig(r'..\data\temp\input_data_plots\%s_48.png' %param)
    plt.close()

