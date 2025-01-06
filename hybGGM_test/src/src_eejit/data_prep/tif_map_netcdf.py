import glob
from osgeo import gdal
import rioxarray
import numpy as np
import os

general_path = '/scratch/depfg/hausw001/data/globgm/tiles_input/'

tilenr = np.arange(0, 164, 1)
tilenames = ['tile_%03d-163' %(i) for i in tilenr]

#change tile name based on folder of interest
for tile in tilenames[48:49]:
    #check if unzipped folder exists
    if not os.path.exists('%s/%s' %(general_path, tile)):
        print('no unzipped folder')
        continue
    else:
        print('tile: %s' %(tile))
    input_maps = '%s/%s/transient/maps' %(general_path, tile)  
    input_netcdfs = '%s/%s/transient/netcdf_maps_own' %(general_path, tile)  

    if not os.path.exists(input_netcdfs):
        os.makedirs(input_netcdfs)  


    files = glob.glob('%s/*.map' %(input_maps))
    for file in files[:1]:
        print(file)
        # open the map file
        fileN = str.split(file, '/')[-1]
        fileN = fileN[:-4]
        # check if file already excist in input_netcdfs folder
        if glob.glob('%s/%s.nc' %(input_netcdfs,fileN)):
            print('file already exists')
            continue

        # Open the file
        temp = rioxarray.open_rasterio(file)    
        # save the values as an index in the xarray
        temp.name = 'Band1'
        temp = temp.rename({"x": "lon", "y": "lat"})
        # Define the output path and filename
        output_path = f'{input_netcdfs}/{fileN}.nc'
        temp.to_netcdf(output_path)
        import xarray as xr
        test = xr.open_dataset(output_path) 
        import matplotlib.pyplot as plt
        plt.imshow(test)

# # selection of output tifs downloaded for example
# output_tif = r'%s\Data\GLOBGM\output\transient_1958-2015\tif' %(general_path)
# output_netcdf = r'%s\Data\GLOBGM\output\transient_1958-2015\netcdf' %(general_path)


# files = glob.glob(r'%s\*wtd-2*.tif' %(output_tif))
# for file in files[:]:
#     print(file)
#     # open the map file
#     fileN = str.split(file, '\\')[-1]
#     fileN = fileN[:-4]
#     tempOut = gdal.Translate(r'%s\%s.nc' %(output_netcdf, fileN), file, format='NetCDF')
