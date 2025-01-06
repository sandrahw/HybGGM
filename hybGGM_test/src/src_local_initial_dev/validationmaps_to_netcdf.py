#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:40:04 2024

@author: niko
"""

import numpy as np
import netCDF4 as nc
import os
import datetime
import csv

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


def readSampleFile(fileName):
  columns=['date_index', 'date',
                                'date_prev_index', 'prev_date',
                                'tile_index', 'tile_lat_index', 'tile_lon_index',
                                'min_lat_index', 'max_lat_index', 'min_lon_index', 'max_lon_index',
                                'min_lat_bound', 'max_lat_bound', 'min_lon_bound', 'max_lon_bound',
                                'min_lat', 'max_lat', 'min_lon', 'max_lon']
  csvFile = csv.reader(open(fileName, "rb"))
  for row in csvFile:
    mydictionary[Col1].append(row[0])
  mydictionary[Col2].append(row[1])
  mydictionary[Col3].append(row[2])
  data = pd.read_csv(fileName)
  return(data)

def properties(fileName):
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
  out = np.zeros((len(latitudes), len(longitudes)))
  createNetCDF(ncFileName, varName, varUnits, latitudes, longitudes)
  return(out)

def data2Matrix(out, sample, min_lat, max_lat, min_lon, max_lon):
  out[min_lat:max_lat, min_lon:max_lon] = sample
  return(out)

dates_all, lons_all, lats_all, dx, dy, sampleID, min_lon_index, max_lon_index, min_lat_index, max_lat_index = properties("samples.csv")

dates, lons, lats, dx, dy, sampleID, min_lon_index, max_lon_index, min_lat_index, max_lat_index = properties("validation_samples.csv")
print(dates)

cnt = 0
for date in np.unique(dates_all):
  print(date)
  ncFileName = "results_%s.nc" %(date)
  out = createMatrix(ncFileName, "mask", "-", lats_all, lons_all)
  selection = [i for i in range(len(dates)) if dates[i] == date]
  print(selection)
  for i in selection:
    out = data2Matrix(out, i, min_lat_index[i],max_lat_index[i],min_lon_index[i],max_lon_index[i]) 
  timeStamp = datetime.datetime(int(date.split("-")[0]), int(date.split("-")[1]), int(date.split("-")[2]))
  data2NetCDF(ncFileName, "mask", out, timeStamp, posCnt = cnt)
  cnt += 1

#def createOutputFile(size):
  
