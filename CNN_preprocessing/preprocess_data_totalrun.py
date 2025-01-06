import pathlib as pl
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

# input_dir = pl.Path('./data')
# save_dir = pl.Path('./saves')
# out_dir = pl.Path('./saves')
tile_example_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Data\GLOBGM\input\tiles_input\tile_048-163\transient'
target_example_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Data\GLOBGM\output\transient_1958-2015'
samples_dir =  r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Data\GLOBGM\input\tiles_input\tile_048-163\transient\cnn_samples'
save_dir = pl.Path('%s/total_samples' % tile_example_path)
out_dir = pl.Path('%s/total_samples' % tile_example_path)
input_dir = pl.Path('%s/netcdf_maps' % tile_example_path)
target_dir = pl.Path('%s/netcdf' % target_example_path)

if not save_dir.exists():
    save_dir.mkdir(parents=True)


#TODO: this sampling is only for CNN without LSTM and temporal component!
# modflow files that are saved monthly
params_monthly = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'surface_water_bed_elevation_used',
 'surface_water_elevation', 'net_RCH']#, 'wtd']
# other modflow files that seem to be static parameters
params_initial = ['bottom_lowermost_layer', 'bottom_uppermost_layer', 
 'drain_conductance', 
 'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
 'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
 'top_uppermost_layer',
 'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer']
target_monthly = ['globgm-wtd']

samples_file = samples_dir +'\samples.csv'
samples = pd.read_csv(samples_file, index_col=0, parse_dates=['date', 'prev_date'])

for param in target_monthly:
        input_files = [f for f in target_dir.glob('%s*'%param) if f.is_file()]

        sample_arrays = []

        sample_index = samples.index[0]
        sample_info = samples.loc[sample_index]
        for sample_index, sample_info in samples.iterrows():
            
            min_lat_bound = sample_info['min_lat_bound']
            max_lat_bound = sample_info['max_lat_bound']
            min_lon_bound = sample_info['min_lon_bound']
            max_lon_bound = sample_info['max_lon_bound']
            
            date_arrays = []
            
            date_key = 'date'
            date = sample_info[date_key]
            for date_key, date in sample_info[['date', 'prev_date']].items():
                
                input_file = [f for f in input_files if date.strftime('%Y%m') in f.stem][0]
                
                with xr.open_dataset(input_file) as da:
                    da = da.sel(lat=slice(min_lat_bound, max_lat_bound),
                                lon=slice(min_lon_bound, max_lon_bound))
                    da = da.drop('crs')
                    #Xarray dataset to xarray array
                    da = da.to_array()
                
                array = da.values
                date_arrays.append(array)
            array = np.stack(date_arrays)
            sample_arrays.append(array)
        array = np.stack(sample_arrays)

        array_out = out_dir / f'array_{param}.npy'
        array_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_out, array)


for param in params_monthly[:]:
    #TODO: loop over all variables
    input_files = [f for f in input_dir.glob('%s_*'%param) if f.is_file()]

    sample_arrays = []

    sample_index = samples.index[0]
    sample_info = samples.loc[sample_index]
    for sample_index, sample_info in samples.iterrows():
        
        min_lat_bound = sample_info['min_lat_bound']
        max_lat_bound = sample_info['max_lat_bound']
        min_lon_bound = sample_info['min_lon_bound']
        max_lon_bound = sample_info['max_lon_bound']
        
        date_arrays = []
        
        date_key = 'date'
        date = sample_info[date_key]
        for date_key, date in sample_info[['date', 'prev_date']].items():
            
            input_file = [f for f in input_files if date.strftime('%Y%m') in f.stem][0]
            
            with xr.open_dataarray(input_file) as da:
                da = da.sel(lat=slice(min_lat_bound, max_lat_bound),
                            lon=slice(min_lon_bound, max_lon_bound))
            
            array = da.values
            date_arrays.append(array)
        array = np.stack(date_arrays)
        sample_arrays.append(array)
    array = np.stack(sample_arrays)

    array_out = out_dir / f'array_{param}.npy'
    array_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_out, array)

for param in params_initial[:]:
    print(param)
    # input_file = [f for f in input_dir.glob('%s.nc'%param) if f.is_file()] #initial files are only one file
    da = xr.open_dataarray('%s/%s.nc'%(input_dir, param))
    sample_arrays = []

    sample_index = samples.index[0]
    sample_info = samples.loc[sample_index]
    for sample_index, sample_info in samples.iterrows():
        
        min_lat_bound = sample_info['min_lat_bound']
        max_lat_bound = sample_info['max_lat_bound']
        min_lon_bound = sample_info['min_lon_bound']
        max_lon_bound = sample_info['max_lon_bound']
        
        date_arrays = []
        
        date_key = 'date'
        date = sample_info[date_key]
        for date_key, date in sample_info[['date', 'prev_date']].items():
            # input_file = [f for f in input_files if date.strftime('%Y%m') in f.stem][0]
            da_sel = da.sel(lat=slice(min_lat_bound, max_lat_bound),
                        lon=slice(min_lon_bound, max_lon_bound))
            
            array = da_sel.values
            date_arrays.append(array)
        array = np.stack(date_arrays)
        sample_arrays.append(array)
    array = np.stack(sample_arrays)

    array_out = out_dir / f'array_{param}.npy'
    array_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_out, array)
