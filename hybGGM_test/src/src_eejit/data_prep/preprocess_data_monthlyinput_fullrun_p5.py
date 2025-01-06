import pathlib as pl
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

# input_dir = pl.Path('./data')
# save_dir = pl.Path('./saves')
# out_dir = pl.Path('./saves')
tile_example_path = "/scratch/depfg/hausw001/data/globgm/tiles_input/tile_048-163/transient/"
target_example_path = "/scratch/depfg/hausw001/data/globgm/output/"
save_dir = pl.Path('%s/cnn_samples' % tile_example_path)
out_dir = pl.Path('%s/cnn_samples' % tile_example_path)
input_dir = pl.Path('%s/netcdf_maps_own' % tile_example_path)
target_dir = pl.Path('%s/netcdf_maps_own' % target_example_path)
# subsets = ['training', 'validation', 'testing']

#TODO: this sampling is only for CNN without LSTM and temporal component!
# modflow files that are saved monthly
params_monthly = ['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'surface_water_bed_elevation_used',
 'surface_water_elevation', 'net_RCH']#, 'wtd']

        
samples_file = save_dir / f'samples.csv'
samples = pd.read_csv(samples_file, index_col=0, parse_dates=['date', 'prev_date'])

for i, param in enumerate(params_monthly[8:10]):
    print(param, i,'out of', len(params_monthly))
    #TODO: loop over all variables
    input_files = [f for f in input_dir.glob('%s_*'%param) if f.is_file()]

    sample_arrays = []

    sample_index = samples.index[0]
    sample_info = samples.loc[sample_index]
    for j, (sample_index, sample_info) in enumerate(samples.iterrows()):
        print(j, 'out of', len(samples))
        
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
    print('stacking')
    array = np.stack(sample_arrays)
    print('saving')
    array_out = out_dir / f'full_array_{param}.npy'
    array_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_out, array)
