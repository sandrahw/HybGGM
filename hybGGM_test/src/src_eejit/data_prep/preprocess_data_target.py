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
save_dir = pl.Path('%s/cnn_samples_30_30_30' % tile_example_path)
out_dir = pl.Path('%s/cnn_samples_30_30_30' % tile_example_path)
input_dir = pl.Path('%s/netcdf_maps_own' % tile_example_path)
target_dir = pl.Path('%s/netcdf_maps_own' % target_example_path)
subsets = ['training', 'validation', 'testing']

#TODO: this sampling is only for CNN without LSTM and temporal component!
# modflow files that are saved monthly
target_monthly = ['globgm-wtd']

for subset in subsets[:]:
    print(f'subset: {subset}')
        
    samples_file = save_dir / f'{subset}_samples.csv'
    samples = pd.read_csv(samples_file, index_col=0, parse_dates=['date', 'prev_date'])

    for param in target_monthly[:]:
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

        array_out = out_dir / f'{subset}_array_{param}.npy'
        array_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_out, array)
