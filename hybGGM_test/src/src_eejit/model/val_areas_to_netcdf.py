#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:40:04 2024

@author: niko
"""
import sys
import numpy as np
import netCDF4 as nc
import os
import datetime
import csv
import glob
import xarray as xr
import matplotlib.pyplot as plt

MV = 1e20
smallNumber = 1E-39

def createNetCDF(ncFileName, varName, varUnits, latitudes, longitudes,\
                                      longName = None):
    
    rootgrp= nc.Dataset(ncFileName,'w', format='NETCDF4')
    
    #-create dimensions - time is unlimited, others are fixed
    rootgrp.createDimension('time',None)
    rootgrp.createDimension('lat',len(latitudes))
    rootgrp.createDimension('lon',len(longitudes))
    
    date_time= rootgrp.createVariable('time','f4',('time',))
    date_time.standard_name= 'time'
    date_time.long_name= 'Days since 1901-01-01'
    
    date_time.units= 'Days since 1901-01-01' 
    date_time.calendar= 'standard'
        
    lat= rootgrp.createVariable('lat','f4',('lat',))
    lat.long_name= 'latitude'
    lat.units= 'degrees_north'
    lat.standard_name = 'latitude'
    
    lon= rootgrp.createVariable('lon','f4',('lon',))
    lon.standard_name= 'longitude'
    lon.long_name= 'longitude'
    lon.units= 'degrees_east'
    
    lat[:]= latitudes
    lon[:]= longitudes
    
    shortVarName = varName
    longVarName  = varName
    if longName != None: longVarName = longName
    var= rootgrp.createVariable(shortVarName,'f4',('time','lat','lon',) ,fill_value=MV,zlib=True)
    var.standard_name = varName
    var.long_name = longVarName
    var.units = varUnits
    rootgrp.title = "simulated CNN outpu"
    rootgrp.institution = "Utrecht University"
    rootgrp.processed = "Processed by Sandra Hauswirth (s.m.Hauswirth@uu.nl)"
    rootgrp.contact = "s.m.hauswirth@uu.nl"
    
    rootgrp.sync()
    rootgrp.close()
  
def data2NetCDF(ncFile,varName,varField,timeStamp,posCnt = None):
  #-write data to netCDF
  rootgrp= nc.Dataset(ncFile,'a')    
  
  shortVarName= varName        
  
  date_time= rootgrp.variables['time']
  if posCnt == None: posCnt = len(date_time)
  
  date_time[posCnt]= nc.date2num(timeStamp,date_time.units,date_time.calendar)
  rootgrp.variables[shortVarName][posCnt,:,:]= (varField)
  
  rootgrp.sync()
  rootgrp.close()


def propertiesCSV(fileName):
  dates = []
  sampleID = []
  min_lon_index = []
  max_lon_index = []
  min_lat_index = []
  max_lat_index = []
  min_lon = []
  max_lon = []
  min_lat = []
  max_lat = []
  with open(fileName) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cnt = 0
    for row in csv_reader:
      if cnt !=0:
        dates.append(row[2])
        sampleID.append(row[5])
        min_lon_index.append(row[10])
        max_lon_index.append(row[11])
        min_lat_index.append(row[8])
        max_lat_index.append(row[9])
        min_lon.append(row[18])
        max_lon.append(row[19])
        min_lat.append(row[16])
        max_lat.append(row[17])
      cnt += 1
  sampleID = np.array(sampleID).astype(int)
  min_lon_index = np.array(min_lon_index).astype(int)
  max_lon_index = np.array(max_lon_index).astype(int)
  min_lat_index = np.array(min_lat_index).astype(int)
  max_lat_index = np.array(max_lat_index).astype(int)
  min_lon = np.array(min_lon).astype(float)
  max_lon = np.array(max_lon).astype(float)
  min_lat = np.array(min_lat).astype(float)
  max_lat = np.array(max_lat).astype(float)
  dy = np.max(max_lat_index)-np.min(min_lat_index)+1
  dx = np.max(max_lon_index)-np.min(min_lon_index)+1
  lats = np.linspace(np.min(min_lat), np.max(max_lat), dy) 
  lons = np.linspace(np.min(min_lon), np.max(max_lon), dx)
  return(dates, lons, lats, dx, dy, sampleID, min_lon_index, max_lon_index, min_lat_index, max_lat_index)

def createMatrix(ncFileName, varName, varUnits, latitudes, longitudes):
  out = np.zeros((len(latitudes), len(longitudes))) + np.nan
  createNetCDF(ncFileName, varName, varUnits, latitudes, longitudes)
  return(out)

def data2Matrix(out, sample, min_lat, max_lat, min_lon, max_lon):
  out[min_lat:(max_lat+1), min_lon:(max_lon+1)] = sample
  return(out)

### Read all tile information
# dates_all, lons_all, lats_all, dx, dy, sampleID, min_lon_index, max_lon_index, min_lat_index, max_lat_index = propertiesCSV("samples.csv")
dates_all, lons_all, lats_all, dx, dy, sampleID, min_lon_index, max_lon_index, min_lat_index, max_lat_index = propertiesCSV("/scratch/depfg/hausw001/data/globgm/tiles_input/tile_048-163/transient/cnn_samples_10_10_10/samples.csv")
### Read information for validation samples
dates, lons, lats, dx, dy, sampleID, min_lon_index, max_lon_index, min_lat_index, max_lat_index = propertiesCSV("/scratch/depfg/hausw001/data/globgm/tiles_input/tile_048-163/transient/cnn_samples_10_10_10/samples.csv")
# print(dates)

### Read spatial validation data (now a dummy to replace later, 180, 180 hardcoded)

print('loading mask')
sys.stdout.flush()
target_mask = np.load("../../data/testing_random_sampling_fulltile_180_101010_deltawtd/full_mask.npy") #mask in 180x180 samples based on wtd
target_mask.shape
mask = target_mask[:,0,:,:]
mask.shape
# mask from true false to 0 1
maks_0_1= np.where(mask==False, 0, 1)
mask_nan = np.where(maks_0_1==0, np.nan, 1)
mask_nan.shape


folders = ['UNet2_50_0.0001_32_1.0']#['UNet2_50_0.0001_4_1.0', 'UNet2_50_0.0001_16_1.0', 'UNet2_50_0.0001_32_1.0', 'UNet6_50_0.0001_4_1.0', 'UNet6_50_0.0001_8_1.0', 'UNet6_50_0.0001_32_1.0']
for fol in folders[:]:
  print('starting with', fol)
  sys.stdout.flush()
  runName = fol
  ncfolder = '/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_fulltile_180_101010_limitedInpSel_deltawtd/%s/ncmaps/' %runName
  if not os.path.exists(ncfolder):
    os.makedirs(ncfolder)
  samples = np.arange(0, 100, 1)
  totrun = []
  print('loading runs')
  sys.stdout.flush()
  for ss in samples[:]:
    run = np.load('../../results/testing/random_sampling_fulltile_180_101010_limitedInpSel_deltawtd/%s/full_pred/y_pred_denorm_full_%s.npy' %(runName,ss))
    totrun.append(run)
  totalrun = np.concatenate(totrun, axis = 0)
  totalrun = totalrun[:,0,:,:]
  totalrun.shape
  # plt.imshow(totalrun[0,:,:])
  # plt.colorbar()
  print('multiplying with mask')
  sys.stdout.flush()
  resultsWmask = totalrun*mask_nan
  del totalrun
  print('saving to netcdf')
  for date in np.unique(dates_all)[:]:
    print(date)
    sys.stdout.flush()
    ncFileName = "%s/results_full_tile_%s.nc" %(ncfolder, date)
    out = createMatrix(ncFileName, "mask", "-", lats_all, lons_all)
    selection = [i for i in range(len(dates)) if dates[i] == date]
    print(selection)
    for i in selection[:]:
      out = data2Matrix(out, resultsWmask[i,:,:], min_lat_index[i],max_lat_index[i],min_lon_index[i],max_lon_index[i])
      # maskout = data2Matrix(out, mask_new_nan[i,:,:], min_lat_index[i],max_lat_index[i],min_lon_index[i],max_lon_index[i])
    timeStamp = datetime.datetime(int(date.split("-")[0]), int(date.split("-")[1]), int(date.split("-")[2]))
    # out = out*maskout
    data2NetCDF(ncFileName, "mask", out, timeStamp, posCnt = 0)
  print('done with', fol)
  del resultsWmask
  sys.stdout.flush()


# import matplotlib.pyplot as plt
# ncfiles = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_fulltile_180_101010_limitedInpSel_wtd/UNet2_50_0.0001_32_1.0/ncmaps/results_*.nc')
# #order by date
# ncfiles.sort()
# for nf in ncfiles[:1]:
#   date = nf.split("_")[-1].split(".")[0]
#   load = xr.open_dataset(nf)
#   plt.figure()
#   plt.imshow(load['mask'][0,:,:])
#   plt.colorbar()
#   plt.title(date)

'''for target data'''

target_deltawtd = np.load('../../data/testing_random_sampling_fulltile_180_101010/full_target_wtd.npy') #target delta wtd in 180x180 samples
target_deltawtd.shape
target = target_deltawtd[:,0,:,:]
target.shape
# plt.imshow(target[0,:,:])
runName = 'UNet2_50_0.0001_32_1.0'
for date in np.unique(dates_all)[:]:
  print(date)
  ncFileName = "/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_fulltile_180_101010_limitedInpSel_wtd/%s/ncmaps/results_full_tile_%s_target.nc" %(runName, date)
  out = createMatrix(ncFileName, "mask", "-", lats_all, lons_all)
  selection = [i for i in range(len(dates)) if dates[i] == date]
  print(selection)
  for i in selection[:]:
    out = data2Matrix(out, target[i,:,:], min_lat_index[i],max_lat_index[i],min_lon_index[i],max_lon_index[i])
  timeStamp = datetime.datetime(int(date.split("-")[0]), int(date.split("-")[1]), int(date.split("-")[2]))
  data2NetCDF(ncFileName, "mask", out, timeStamp, posCnt = 0)














'''test example'''
# validationData = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_180/UNet2_100_0.0001_32_1.0/y_pred_denorm_new.npy' )
# mask_val = np.load('../../data/testing_random_sampling_180/mask_validation.npy') #180
# mask_test_na = np.where(mask_val==0, np.nan, 1)
# target_val = np.load('../../data/testing_random_sampling_180/target_validation.npy')#180

# for date in np.unique(dates_all)[:]:
#   print(date)
#   ncFileName = "/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_180/UNet2_100_0.0001_32_1.0/results_%s.nc" %(date)
#   out = createMatrix(ncFileName, "mask", "-", lats_all, lons_all)
#   selection = [i for i in range(len(dates)) if dates[i] == date]
#   print(selection)
#   for i in selection[:]:
#     out = data2Matrix(out, validationData[i,:,:], min_lat_index[i],max_lat_index[i],min_lon_index[i],max_lon_index[i])
#     maskout = data2Matrix(out, mask_test_na[i,:,:], min_lat_index[i],max_lat_index[i],min_lon_index[i],max_lon_index[i])
#   timeStamp = datetime.datetime(int(date.split("-")[0]), int(date.split("-")[1]), int(date.split("-")[2]))
#   data2NetCDF(ncFileName, "mask", out, timeStamp, posCnt = 0)

# #TODO plotting with target next to it


# import matplotlib.pyplot as plt
# ncfiles = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_180/UNet2_100_0.0001_32_1.0/results_*.nc')
# #order by date
# ncfiles.sort()
# for nf in ncfiles[:10]:
#   date = nf.split("_")[-1].split(".")[0]
#   load = xr.open_dataset(nf)
#   plt.figure()
#   plt.imshow(load['mask'][0,:,:])
#   plt.colorbar()
#   plt.title(date)