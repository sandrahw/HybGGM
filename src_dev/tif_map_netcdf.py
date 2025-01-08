import glob
from osgeo import gdal

general_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl'

#change tile name based on folder of interest
input_maps = r'%s\Data\GLOBGM\input\tiles_input\tile_048-163\transient\maps' %(general_path)  
input_netcdfs = r'%s\Data\GLOBGM\input\tiles_input\tile_048-163\transient\netcdf_maps' %(general_path)  

files = glob.glob(r'%s\*.map' %(input_maps))
for file in files[:]:
    print(file)
    # open the map file
    fileN = str.split(file, '\\')[-1]
    fileN = fileN[:-4]
    # check if file already excist in input_netcdfs folder
    if glob.glob(r'%s\%s.nc' %(input_netcdfs,fileN)):
        print('file already exists')
        continue
    temp = gdal.Open(file)
    driver = gdal.GetDriverByName('NetCDF')
    temp_copy = driver.CreateCopy(r'%s\%s.nc' %(input_netcdfs,fileN), temp, 0)

    map_temp = gdal.Translate(file, r'%s\%s.nc' %(input_netcdfs,fileN), format='NetCDF')


# selection of output tifs downloaded for example
output_tif = r'%s\Data\GLOBGM\output\transient_1958-2015\tif' %(general_path)
output_netcdf = r'%s\Data\GLOBGM\output\transient_1958-2015\netcdf' %(general_path)


files = glob.glob(r'%s\*wtd-2*.tif' %(output_tif))
for file in files[:]:
    print(file)
    # open the map file
    fileN = str.split(file, '\\')[-1]
    fileN = fileN[:-4]
    tempOut = gdal.Translate(r'%s\%s.nc' %(output_netcdf, fileN), file, format='NetCDF')
