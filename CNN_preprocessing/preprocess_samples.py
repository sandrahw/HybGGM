import pathlib as pl
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

# input_dir = pl.Path('./data')
# out_dir = pl.Path('./saves')

tile_example_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Data\GLOBGM\input\tiles_input\tile_048-163\transient'
input_dir = pl.Path('%s/netcdf_maps' % tile_example_path)
out_dir = pl.Path('%s/cnn_samples' % tile_example_path)
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)

tile_lat_size = 180
tile_lon_size = 180

#TODO: loop over all variables
#TODO: guess not necessary relevant as every variable of the same tile should have same dimensions
#TODO have to create this though for every separate tile!
input_files = [f for f in input_dir.glob('net_RCH_*') if f.is_file()]

with xr.open_dataset(input_files[0]) as ds:
    lats = ds['lat'].values
    lons = ds['lon'].values
lat_resolution = lats[1] - lats[0]
lon_resolution = lons[1] - lons[0]
    
dates = [f.stem.split('_')[-1] for f in input_files]
dates = [dt.datetime.strptime(d, '%Y%m') for d in dates]
dates = np.array(dates)

samples = pd.DataFrame(columns=['date_index', 'date',
                                'date_prev_index', 'prev_date',
                                'tile_index', 'tile_lat_index', 'tile_lon_index',
                                'min_lat_index', 'max_lat_index', 'min_lon_index', 'max_lon_index',
                                'min_lat_bound', 'max_lat_bound', 'min_lon_bound', 'max_lon_bound',
                                'min_lat', 'max_lat', 'min_lon', 'max_lon',])

for date_index in range(1, dates.size):
    
    date = dates[date_index]
    date_prev_index = date_index - 1
    date_prev = dates[date_prev_index]
    
    tile_index = 0
    for tile_lat_index, min_lat_index in enumerate(range(0, lats.size, tile_lat_size)):
        for tile_lon_index, min_lon_index in enumerate(range(0, lons.size, tile_lon_size)):
            
            max_lat_index = min_lat_index + tile_lat_size - 1
            max_lon_index = min_lon_index + tile_lon_size - 1
            
            min_lat = lats[min_lat_index]
            max_lat = lats[max_lat_index]
            min_lon = lons[min_lon_index]
            max_lon = lons[max_lon_index]
            
            min_lat_bound = min_lat - lat_resolution / 2
            max_lat_bound = max_lat + lat_resolution / 2
            min_lon_bound = min_lon - lon_resolution / 2
            max_lon_bound = max_lon + lon_resolution / 2
            
            samples.loc[samples.shape[0]] = [date_index, date,
                                             date_prev_index, date_prev,
                                             tile_index, tile_lat_index, tile_lon_index,
                                             min_lat_index, max_lat_index,
                                             min_lon_index, max_lon_index,
                                             min_lat_bound, max_lat_bound,
                                             min_lon_bound, max_lon_bound,
                                             min_lat, max_lat, min_lon, max_lon]
            
            tile_index += 1
            
samples_out = out_dir / 'samples.csv'
samples_out.parent.mkdir(parents=True, exist_ok=True)
samples.to_csv(samples_out)