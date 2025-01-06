#TODO load modules
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#TODO load one month of tile 48 data
example_tile48 = xr.open_dataset("/scratch/depfg/hausw001/data/globgm/tiles_input/tile_048-163/transient/netcdf_maps_own/abstraction_lowermost_layer.nc")
#TODO extract lat lon boundaries
lat_min = example_tile48.lat.min()
lat_max = example_tile48.lat.max()
lon_min = example_tile48.lon.min()
lon_max = example_tile48.lon.max()
#TODO load wtd data (2003-2004)
globwtd = glob.glob("/scratch/depfg/hausw001/data/globgm/output/netcdf_maps_own/*.nc")
globwtd_example = xr.open_dataset(globwtd[0])
globwtd_europe = globwtd_example.sel(lat=slice(35,72),lon=slice(-25,50))



years = np.arange(2003,2005)
globwtd_sel = [x for x in globwtd if any(str(year) in x for year in years)]
globwtd_sel.sort()
globwtd_sel_load = [xr.open_dataset(x) for x in globwtd_sel]
#TODO cut wtd data to the same lat lon boundaries
globwtd_cut = [x.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max)) for x in globwtd_sel_load]
globwtd_combined = xr.concat(globwtd_cut,dim='time')
globwtd_combined['Band1'][0].plot()
#TODO select specific areas

#TODO plot the timeseries of the selected areas


plt.figure( figsize=(10, 10))
plt.imshow(globwtd_europe['Band1'], cmap='viridis')
plt.gca().invert_yaxis()
plt.colorbar(shrink=0.5)
plt.clim(np.nanpercentile(globwtd_europe['Band1'], 1), np.nanpercentile(globwtd_europe['Band1'], 99))
#remove axis and frame of figure
plt.axis('off')
plt.gca().set_frame_on(False)

plt.figure( figsize=(10, 10))
#plot dataset with viridis colormap but alpha different
plt.imshow(globwtd_europe['Band1'], cmap='viridis', alpha=0.5)
plt.gca().invert_yaxis()
# add square box based on lat lon boundaries of globwtd_combined
plt.gca().add_patch(plt.Rectangle((0, 0), globwtd_combined['Band1'].shape[1], globwtd_combined['Band1'].shape[0], fill=False, edgecolor='red', lw=2))
plt.colorbar(shrink=0.5)
plt.clim(np.nanpercentile(globwtd_europe['Band1'], 1), np.nanpercentile(globwtd_europe['Band1'], 99))
#remove axis and frame of figure
# plt.axis('off')
plt.gca().set_frame_on(False)


plt.figure( figsize=(10, 10))
plt.imshow(globwtd_combined['Band1'][0], cmap='viridis')
plt.gca().invert_yaxis()

