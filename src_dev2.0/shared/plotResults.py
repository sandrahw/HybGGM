#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:00:11 2024

@author: niko
"""
############ Plotting ############

import glob
import netCDF4 as netCDF
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def readNetCDF(fileName):
    with netCDF.Dataset(fileName, mode="r") as nc:
        lon = nc["lon"][:]
        lat = nc["lat"][:]
        vars = list(nc.variables.keys())
        data = nc[vars[-1]][:]
    return data, lat, lon

model1 = "CNNLSTM"
model2 = "LSTM"

all_predictions1, lat, lon = readNetCDF("all_predictions_%s.nc" %(model1))
target1, lat, lon = readNetCDF("target_%s.nc" %(model1))

all_predictions2, lat, lon = readNetCDF("all_predictions_%s.nc" %(model2))
target2, lat, lon = readNetCDF("target_%s.nc" %(model2))


def plot_predictions_targets_maps(predictions, target, lon, lat, time_step=0, save_path=None):
    """
    Create a four-panel plot: scatter plot, predictions map, targets map, and error map.
    
    Parameters:
    - predictions: 3D array [time, lat, lon], predicted values.
    - target: 3D array [time, lat, lon], true values.
    - lon: 1D array, longitude values.
    - lat: 1D array, latitude values.
    - time_step: int, the time step to plot for the maps.
    """
    # Flatten data for the scatter plot
    pred_flat = predictions[time_step,:,:].flatten()
    target_flat = target[time_step,:,:].flatten()
    
    # Calculate error for the error map
    error = predictions[time_step] - target[time_step]
    
    # Create the multi-panel figure
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # Panel 1: Scatter plot (Predictions vs. Targets)
    ax[0, 0].scatter(target_flat, pred_flat, alpha=0.5, edgecolor='k', s=20)
    ax[0, 0].plot([target_flat.min(), target_flat.max()], [target_flat.min(), target_flat.max()], 
                  color='red', linestyle='--', label='1:1 Line')
    ax[0, 0].set_title("Predictions vs Targets")
    ax[0, 0].set_xlabel("Target")
    ax[0, 0].set_ylabel("Prediction")
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # Panel 2: Predictions Map
    p1 = ax[1, 1].pcolormesh(lon, lat, predictions[time_step], cmap='viridis', shading='auto')
    fig.colorbar(p1, ax=ax[1, 1], label="Predictions")
    ax[1, 1].set_title(f"Predictions Map (Time Step: {time_step})")
    ax[1, 1].set_xlabel("Longitude")
    ax[1, 1].set_ylabel("Latitude")
    
    # Panel 3: Targets Map
    p2 = ax[1, 0].pcolormesh(lon, lat, target[time_step], cmap='viridis', shading='auto')
    fig.colorbar(p2, ax=ax[1, 0], label="Targets")
    ax[1, 0].set_title(f"Targets Map (Time Step: {time_step})")
    ax[1, 0].set_xlabel("Longitude")
    ax[1, 0].set_ylabel("Latitude")
    
    # Panel 4: Error Map
    p3 = ax[0, 1].pcolormesh(lon, lat, error, cmap='coolwarm', shading='auto')
    fig.colorbar(p3, ax=ax[0, 1], label="Prediction Error")
    ax[0, 1].set_title(f"Prediction Error Map (Time Step: {time_step})")
    ax[0, 1].set_xlabel("Longitude")
    ax[0, 1].set_ylabel("Latitude")
    
    # Show the plot
    plt.suptitle("Model Predictions and Error Analysis", fontsize=16, y=1.02)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# Example usage
# Assuming predictions, target, lon, and lat are already defined as NumPy arrays
# predictions and target are [time, lat, lon], and lon/lat are 1D arrays.
time_step_to_plot = 71  # Choose a specific time step to visualize

plot_predictions_targets_maps(all_predictions1, target1, lon, lat, time_step=time_step_to_plot, save_path="/Users/niko/Downloads/LSTM/Figure_%s.png" %(model1))

plot_predictions_targets_maps(all_predictions2, target2, lon, lat, time_step=time_step_to_plot, save_path="/Users/niko/Downloads/LSTM/Figure_%s.png" %(model2))

############

def plot_combined_predictions_maps(
    predictions1, target1, predictions2, target2, lon, lat, model1, model2, time_step=0, save_path=None
):
    """
    Create a combined figure with predictions and error maps for two models.

    Parameters:
    - predictions1: 3D array [time, lat, lon], predictions from Model 1.
    - target1: 3D array [time, lat, lon], true values for Model 1.
    - predictions2: 3D array [time, lat, lon], predictions from Model 2.
    - target2: 3D array [time, lat, lon], true values for Model 2.
    - lon: 1D array, longitude values.
    - lat: 1D array, latitude values.
    - model1: str, name of Model 1.
    - model2: str, name of Model 2.
    - time_step: int, the time step to plot for the maps.
    - save_path: str or None, if provided, saves the plot to the given path.
    """
    # Calculate errors
    error1 = predictions1[time_step] - target1[time_step]
    error2 = predictions2[time_step] - target2[time_step]
    
    # Determine shared color limits for error maps
    error_min = min(error1.min(), error2.min())
    error_max = max(error1.max(), error2.max())
    
    # Create the figure
    fig, ax = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    
    # Model 1
    ax[0, 0].scatter(
        target1[time_step].flatten(), predictions1[time_step].flatten(), alpha=0.5, edgecolor="k", s=20
    )
    ax[0, 0].plot(
        [target1.min(), target1.max()], [target1.min(), target1.max()], color="red", linestyle="--"
    )
    ax[0, 0].set_title(f"{model1}: Predictions vs Targets")
    ax[0, 0].set_xlabel("Target")
    ax[0, 0].set_ylabel("Prediction")
    ax[0, 0].grid(True)
    
    p1 = ax[0, 1].pcolormesh(lon, lat, predictions1[time_step], cmap="viridis", shading="auto")
    fig.colorbar(p1, ax=ax[0, 1], label="Predictions")
    ax[0, 1].set_title(f"{model1}: Predictions Map")
    ax[0, 1].set_xlabel("Longitude")
    ax[0, 1].set_ylabel("Latitude")
    
    p2 = ax[0, 2].pcolormesh(lon, lat, target1[time_step], cmap="viridis", shading="auto")
    fig.colorbar(p2, ax=ax[0, 2], label="Targets")
    ax[0, 2].set_title(f"{model1}: Targets Map")
    ax[0, 2].set_xlabel("Longitude")
    ax[0, 2].set_ylabel("Latitude")
    
    p3 = ax[0, 3].pcolormesh(
        lon, lat, error1, cmap="coolwarm", shading="auto", vmin=error_min, vmax=error_max
    )
    fig.colorbar(p3, ax=ax[0, 3], label="Prediction Error")
    ax[0, 3].set_title(f"{model1}: Error Map")
    ax[0, 3].set_xlabel("Longitude")
    ax[0, 3].set_ylabel("Latitude")
    
    # Model 2
    ax[1, 0].scatter(
        target2[time_step].flatten(), predictions2[time_step].flatten(), alpha=0.5, edgecolor="k", s=20
    )
    ax[1, 0].plot(
        [target2.min(), target2.max()], [target2.min(), target2.max()], color="red", linestyle="--"
    )
    ax[1, 0].set_title(f"{model2}: Predictions vs Targets")
    ax[1, 0].set_xlabel("Target")
    ax[1, 0].set_ylabel("Prediction")
    ax[1, 0].grid(True)
    
    p4 = ax[1, 1].pcolormesh(lon, lat, predictions2[time_step], cmap="viridis", shading="auto")
    fig.colorbar(p4, ax=ax[1, 1], label="Predictions")
    ax[1, 1].set_title(f"{model2}: Predictions Map")
    ax[1, 1].set_xlabel("Longitude")
    ax[1, 1].set_ylabel("Latitude")
    
    p5 = ax[1, 2].pcolormesh(lon, lat, target2[time_step], cmap="viridis", shading="auto")
    fig.colorbar(p5, ax=ax[1, 2], label="Targets")
    ax[1, 2].set_title(f"{model2}: Targets Map")
    ax[1, 2].set_xlabel("Longitude")
    ax[1, 2].set_ylabel("Latitude")
    
    p6 = ax[1, 3].pcolormesh(
        lon, lat, error2, cmap="coolwarm", shading="auto", vmin=error_min, vmax=error_max
    )
    fig.colorbar(p6, ax=ax[1, 3], label="Prediction Error")
    ax[1, 3].set_title(f"{model2}: Error Map")
    ax[1, 3].set_xlabel("Longitude")
    ax[1, 3].set_ylabel("Latitude")
    
    # Overall title
    plt.suptitle(
        f"Comparison of {model1} and {model2} Predictions and Error Analysis", fontsize=18, y=1.02
    )
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

plot_combined_predictions_maps(
    all_predictions1, target1, all_predictions2, target2,
    lon, lat, model1, model2, time_step=71,
    save_path="/Users/niko/Downloads/LSTM/Combined_Figure_%s_%s.png" %(model1, model2)
)


###################


def plot_combined_time_series(
    predictions1, target1, predictions2, target2, lon, lat, location_lon, location_lat, model1, model2, save_path=None
):
    """
    Create a combined time series plot for a single location of predictions and target from two models.
    
    Parameters:
    - predictions1: 3D array [time, lat, lon], predicted values for Model 1.
    - target1: 3D array [time, lat, lon], true values for Model 1.
    - predictions2: 3D array [time, lat, lon], predicted values for Model 2.
    - target2: 3D array [time, lat, lon], true values for Model 2.
    - lon: 1D array, longitude values.
    - lat: 1D array, latitude values.
    - location_lon: float, longitude of the location to plot.
    - location_lat: float, latitude of the location to plot.
    - model1: str, name of Model 1.
    - model2: str, name of Model 2.
    - save_path: str or None, if provided, saves the plot to the given path.
    """
    # Find the closest grid point for the specified location
    lon_idx = np.argmin(np.abs(lon - location_lon))
    lat_idx = np.argmin(np.abs(lat - location_lat))
    
    # Extract time series for the chosen location
    pred_series1 = predictions1[:, lat_idx, lon_idx]
    target_series1 = target1[:, lat_idx, lon_idx]
    pred_series2 = predictions2[:, lat_idx, lon_idx]
    target_series2 = target2[:, lat_idx, lon_idx]
    
    # Create the combined plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
    
    # Model 1
    ax[0].plot(pred_series1, label=f"Predictions ({model1})", marker="o", linestyle="-")
    ax[0].plot(target_series1, label="Target", marker="x", linestyle="--")
    ax[0].set_title(f"{model1}: Time Series\n(Lon: {lon[lon_idx]:.2f}, Lat: {lat[lat_idx]:.2f})")
    ax[0].set_xlabel("Time Steps")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].grid(True)
    
    # Model 2
    ax[1].plot(pred_series2, label=f"Predictions ({model2})", marker="o", linestyle="-", color="orange")
    ax[1].plot(target_series2, label="Target", marker="x", linestyle="--", color="green")
    ax[1].set_title(f"{model2}: Time Series\n(Lon: {lon[lon_idx]:.2f}, Lat: {lat[lat_idx]:.2f})")
    ax[1].set_xlabel("Time Steps")
    ax[1].legend()
    ax[1].grid(True)
    
    # Overall title
    plt.suptitle(f"Time Series Comparison at Location (Lon: {lon[lon_idx]:.2f}, Lat: {lat[lat_idx]:.2f})", fontsize=16)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# Example usage
# Assuming predictions, target, lon, and lat are already defined as NumPy arrays
location_lon = 8.5  # Example longitude
location_lat = 49.2  # Example latitude

plot_combined_time_series(all_predictions1, target1, all_predictions2, target2, lon, lat, location_lon, location_lat, model1, model2, save_path="/Users/niko/Downloads/LSTM/Combined_timeseries_%s_%s.png" %(model1, model2))


##############

def plot_temporal_correlation_comparison(
    target1, predictions1, target2, predictions2, lon, lat, model1, model2, save_path=None
):
    """
    Compute and plot temporal correlation maps for two datasets side by side, 
    with histograms of the correlation distributions below the maps. Histograms 
    share the same y-axis limit, and median correlation is displayed below each histogram.
    
    Parameters:
    - target1: 3D array [time_steps, lat, lon], true values for dataset 1.
    - predictions1: 3D array [time_steps, lat, lon], predicted values for dataset 1.
    - target2: 3D array [time_steps, lat, lon], true values for dataset 2.
    - predictions2: 3D array [time_steps, lat, lon], predicted values for dataset 2.
    - lon: 1D array, longitude values.
    - lat: 1D array, latitude values.
    - model1: str, name of Model 1.
    - model2: str, name of Model 2.
    - save_path: Optional path to save the figure. If None, the plot is shown.
    """
    def compute_temporal_correlation(target, predictions):
        """
        Compute temporal correlation for each spatial location.
        """
        time_steps, n_lat, n_lon = target.shape
        correlation_map = np.zeros((n_lat, n_lon))
        for i in range(n_lat):
            for j in range(n_lon):
                target_series = target[:, i, j]
                predictions_series = predictions[:, i, j]
                if np.std(target_series) > 0 and np.std(predictions_series) > 0:
                    correlation_map[i, j] = np.corrcoef(target_series, predictions_series)[0, 1]
                else:
                    correlation_map[i, j] = np.nan  # Handle cases with no variation
        return correlation_map

    # Compute correlation maps for both datasets
    corr_map1 = compute_temporal_correlation(target1, predictions1)
    corr_map2 = compute_temporal_correlation(target2, predictions2)

    # Flatten maps and remove NaN values for histogram
    corr_flat1 = corr_map1.flatten()
    corr_flat2 = corr_map2.flatten()
    corr_flat1 = corr_flat1[~np.isnan(corr_flat1)]
    corr_flat2 = corr_flat2[~np.isnan(corr_flat2)]

    # Compute medians
    median_corr1 = np.median(corr_flat1)
    median_corr2 = np.median(corr_flat2)

    # Set consistent min and max values
    vmin, vmax = -1, 1

    # Define equal-sized bins for histograms
    bins = np.linspace(vmin, vmax, 30)

    # Get the maximum frequency for shared y-axis in histograms
    hist1, _ = np.histogram(corr_flat1, bins=bins)
    hist2, _ = np.histogram(corr_flat2, bins=bins)
    max_y = max(hist1.max(), hist2.max())

    # Create the figure with 4 subplots: 2 maps on top, 2 histograms below
    fig, ax = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    # Dataset 1 Map
    p1 = ax[0, 0].pcolormesh(lon, lat, corr_map1, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(p1, ax=ax[0, 0], label="Temporal Correlation")
    ax[0, 0].set_title(f"{model1}: Temporal Correlation Map", fontsize=14)
    ax[0, 0].set_xlabel("Longitude")
    ax[0, 0].set_ylabel("Latitude")

    # Dataset 2 Map
    p2 = ax[0, 1].pcolormesh(lon, lat, corr_map2, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(p2, ax=ax[0, 1], label="Temporal Correlation")
    ax[0, 1].set_title(f"{model2}: Temporal Correlation Map", fontsize=14)
    ax[0, 1].set_xlabel("Longitude")
    ax[0, 1].set_ylabel("Latitude")

    # Histogram for Dataset 1
    ax[1, 0].hist(corr_flat1, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    ax[1, 0].set_xlim(vmin, vmax)
    ax[1, 0].set_ylim(0, max_y * 1.1)
    ax[1, 0].set_title(f"{model1}: Correlation Distribution", fontsize=14)
    ax[1, 0].set_xlabel("Temporal Correlation")
    ax[1, 0].set_ylabel("Frequency")
    ax[1, 0].text(0.05, 0.9, f"Median: {median_corr1:.2f}", transform=ax[1, 0].transAxes, fontsize=12)
    ax[1, 0].grid(True)

    # Histogram for Dataset 2
    ax[1, 1].hist(corr_flat2, bins=bins, color="salmon", edgecolor="black", alpha=0.7)
    ax[1, 1].set_xlim(vmin, vmax)
    ax[1, 1].set_ylim(0, max_y * 1.1)
    ax[1, 1].set_title(f"{model2}: Correlation Distribution", fontsize=14)
    ax[1, 1].set_xlabel("Temporal Correlation")
    ax[1, 1].set_ylabel("Frequency")
    ax[1, 1].text(0.05, 0.9, f"Median: {median_corr2:.2f}", transform=ax[1, 1].transAxes, fontsize=12)
    ax[1, 1].grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# Example Usage
plot_temporal_correlation_comparison(
    target1=target1,
    predictions1=all_predictions1,
    target2=target2,
    predictions2=all_predictions2,
    lon=lon,
    lat=lat,
    model1=model1,
    model2=model2,
    save_path="/Users/niko/Downloads/LSTM/temporal_correlation_comparison_%s_%s.png" %(model1, model2),
)

