#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:18:06 2024

@author: niko
"""
import numpy as np
import netCDF4 as netCDF

def readNetCDF(fileName):
    with netCDF.Dataset(fileName, mode="r") as nc:
        lon = nc["lon"][:]
        lat = nc["lat"][:]
        vars = list(nc.variables.keys())
        data = nc[vars[-1]][:]
    return data, lat, lon

def calculate_metrics(predictions, target):
    bias = np.nanmean(predictions - target)
    rmse = np.sqrt(np.mean((predictions - target) ** 2))
    nrmse = np.sqrt(np.mean((predictions - target) ** 2))/np.nanmean(predictions)
    valid_mask = ~np.isnan(predictions) & ~np.isnan(target)
    filtered_x = predictions[valid_mask]
    filtered_y = target[valid_mask]
    R = np.corrcoef(filtered_x.flatten(), filtered_y.flatten())[0,1]
    varRatio = np.nanvar(predictions)/np.nanvar(target)
    meanRatio = np.nanmean(predictions)/np.nanmean(target)
    valid_mask = ~np.isnan(np.nanmean(predictions, axis=0)) & ~np.isnan(np.nanmean(target, axis=0))
    filtered_x = np.nanmean(predictions, axis=0)[valid_mask]
    filtered_y = np.nanmean(target, axis=0)[valid_mask]    
    spatialR = np.corrcoef(filtered_x.flatten(), filtered_y.flatten())[0,1]
    print(f"Mean observed: {np.nanmean(target):.4f}")
    print(f"Mean prediction: {np.nanmean(predictions):.4f}")
    print(f"Std observed: {np.nanstd(target):.4f}")
    print(f"Std prediction: {np.nanstd(predictions):.4f}")
    print(f"Bias: {bias:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NRMSE: {nrmse:.4f}")
    print(f"Full R: {R:.4f}")
    print(f"Spatial R: {spatialR:.4f}")
    print(f"Mean Ratio: {meanRatio:.4f}")
    print(f"Variability Ratio: {varRatio:.4f}")
    return bias, rmse, nrmse, R, spatialR, meanRatio, varRatio

def loadData(predFileName, targetFileName):
    predictions, lat, lon = readNetCDF(predFileName)
    target, lat, lon = readNetCDF(targetFileName)
    return predictions, target, lat, lon

### Change fileNames by output data for specific simulation
predictions, target, lat, lon = loadData("predictions_LSTM.nc", "target_LSTM.nc")

calculate_metrics(predictions, target)

predictions, target, lat, lon = loadData("predictions_CNNLSTM.nc", "target_CNNLSTM.nc")

calculate_metrics(predictions, target)
