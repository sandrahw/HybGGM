
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'
input_path = r'%s\Data\GLOBGM\input\tiles_input' %(general_path)
output_path = r'%s\Data\GLOBGM\output\transient_1958-2015\netcdf' %(general_path)

tile = 'tile_048-163'

i_rch = glob.glob(r'%s\%s\transient\netcdf_maps\net_RCH_*.nc' %(input_path, tile))
# temp_i = xr.open_dataset(i_rch[0])
input_rch = xr.open_mfdataset(i_rch, concat_dim='time', combine='nested')

t_files = glob.glob(r'%s\*.nc' %(output_path))
# temp_o = xr.open_dataset(t_files[0])
target = xr.open_mfdataset(t_files, concat_dim='time', combine='nested')
target = target.drop_vars("crs")

lon_bounds = (0, 15)
lat_bounds = (45, 60)

target_cropped = target.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

test_i = input_rch['Band1'].isel(time=1)
test_o = target_cropped['Band1'].isel(time=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
test_i.plot(ax=ax1)
test_o.plot(ax=ax2)

input_rch.to_netcdf(r'..\data\temp\input_rch.nc')
target_cropped.to_netcdf(r'..\data\temp\target.nc')

'''other model parameters prep'''
# separate layers without monthly data (initial conditions?)
['abstraction_lowermost_layer', 'abstraction_uppermost_layer', 
 'bed_conductance_used', 
 'bottom_lowermost_layer', 'bottom_uppermost_layer', 
 'drain_conductance', 
 'drain_elevation_lowermost_layer', 'drain_elevation_uppermost_layer', 
 'horizontal_conductivity_lowermost_layer', 'horizontal_conductivity_uppermost_layer', 
 'initial_head_lowermost_layer', 'initial_head_uppermost_layer',
 'primary_storage_coefficient_lowermost_layer', 'primary_storage_coefficient_uppermost_layer',
 'surface_water_bed_elevation_used', 
 'surface_water_elevation',
 'top_uppermost_layer',
 'vertical_conductivity_lowermost_layer', 'vertical_conductivity_uppermost_layer']

# monthly data
params = ['abstraction_lowermost_layer_', 'abstraction_uppermost_layer_', 
 'bed_conductance_used_', 
 'drain_elevation_lowermost_layer_', 'drain_elevation_uppermost_layer_', 
 'initial_head_lowermost_layer_', 'initial_head_uppermost_layer_',
 'surface_water_bed_elevation_used_',
 'surface_water_elevation_']
 
for param in params[:]:
    data = glob.glob(r'%s\%s\transient\netcdf_maps\%s*.nc' %(input_path, tile, param))
    dataDF = xr.open_mfdataset(data, concat_dim='time', combine='nested')
    dataDF.to_netcdf(r'..\data\temp\%s.nc' %(param[:-1]))
