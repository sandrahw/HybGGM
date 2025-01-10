import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np

stations = pd.read_csv("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year.csv", index_col=0)
#make sure index is in datetime format
stations.index = pd.to_datetime(stations.index, format='%Y-%m-%d')
stations_all_unique = stations.drop_duplicates(subset=['wellID'])

# define lat and lon bounds for tile 48
lon_bounds = (7, 10) #CH bounds(5,10)
lat_bounds = (47, 50)#CH bounds(45,50)

#find stations that are within the bounds
stations_sel = stations[(stations['lon'] >= lon_bounds[0]) & (stations['lon'] <= lon_bounds[1]) & (stations['lat'] >= lat_bounds[0]) & (stations['lat'] <= lat_bounds[1])]
stations_sel.to_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_tile48_noglobgm.pkl")

# df with every station only mentioned once
stations_sel_unique = stations_sel.drop_duplicates(subset=['wellID'])

# Plot the selected stations
plt.figure(figsize=(10, 10))
plt.scatter(stations_sel_unique['lon'], stations_sel_unique['lat'], color='red', label='Selected Stations')   
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Selected Stations within Bounds')

mapwtd = xr.open_dataset("/eejit/home/hausw001/HybGGM/hybGGM_test/data/target/wtd.nc")
map_sel = mapwtd.sel(lon=slice(lon_bounds[0], lon_bounds[1]), lat=slice(lat_bounds[0], lat_bounds[1]))

# plt.figure(figsize=(10, 10))
# map_sel.Band1[0].plot()
# plt.scatter(stations_sel_unique['lon'], stations_sel_unique['lat'], color='red', label='Selected Stations')
# plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area.png")

# plt.figure(figsize=(10, 10))
# mapwtd.Band1[0].plot()
# plt.scatter(stations_all_unique['lon'], stations_all_unique['lat'], color='red', label='Selected Stations')
# plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48.png")


def compute_anomaly_correlation(df, var1, var2):
    """
    Compute the anomaly correlation between two variables in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with time, var1, and var2 columns.
        var1 (str): Column name of the first variable.
        var2 (str): Column name of the second variable.
        time_col (str): Column name of the time variable (should be datetime).

    Returns:
        float: Anomaly correlation coefficient.
    """
      
    # Add a 'month' column for climatology calculation
    df['month'] = df.index.month
    
    # Calculate monthly climatology (mean for each month)
    climatology_var1 = df.groupby('month')[var1].transform('mean')
    climatology_var2 = df.groupby('month')[var2].transform('mean')
    
    # Compute anomalies by subtracting climatology
    df['anomaly_var1'] = df[var1] - climatology_var1
    df['anomaly_var2'] = df[var2] - climatology_var2
    
    # Drop NaN values to ensure valid correlation calculation
    df.dropna(subset=['anomaly_var1', 'anomaly_var2'], inplace=True)
    
    # Calculate the anomaly correlation (Pearson correlation coefficient)
    anomaly_correlation = np.corrcoef(df['anomaly_var1'], df['anomaly_var2'])[0, 1]
    
    return anomaly_correlation

def compute_bias(observed, simulated):
    """
    Compute the bias between observed and simulated data.

    Parameters:
        observed (pd.Series): Observed time series.
        simulated (pd.Series): Simulated time series.

    Returns:
        float: Bias (mean simulated - mean observed).
    """
    # Ensure both series have the same length and no NaN values
    observed, simulated = observed.align(simulated, join='inner')
    observed = observed.dropna()
    simulated = simulated.dropna()
    
    # Calculate bias
    bias = simulated.mean() - observed.mean()
    
    return bias

def compute_kge(observed, simulated):
    """
    Compute the Kling-Gupta Efficiency (KGE) between observed and simulated data,
    handling NaN values.

    Parameters:
        observed (pd.Series): Observed time series.
        simulated (pd.Series): Simulated time series.

    Returns:
        float: KGE value.
    """
    # Align and drop NaN values
    observed, simulated = observed.align(simulated, join='inner')
    mask = observed.notna() & simulated.notna()
    observed = observed[mask]
    simulated = simulated[mask]

    # If no valid data remains after handling NaNs, return NaN
    if len(observed) == 0 or len(simulated) == 0:
        return np.nan

    # Calculate the mean of observed and simulated
    mean_obs = observed.mean()
    mean_sim = simulated.mean()

    # Calculate the standard deviation of observed and simulated
    std_obs = observed.std()
    std_sim = simulated.std()

    # Calculate components of KGE
    r = np.corrcoef(observed, simulated)[0, 1]  # Pearson correlation coefficient
    beta = mean_sim / mean_obs  # Bias ratio
    gamma = (std_sim / mean_sim) / (std_obs / mean_obs)  # Variability ratio

    # KGE calculation
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

    return kge

def compute_low_flow_metrics(observed, simulated, low_flow_threshold):
    """
    Compute bias and KGE with emphasis on low-flow periods (defined by a percentile threshold),
    handling NaN values.

    Parameters:
        observed (pd.Series): Observed time series.
        simulated (pd.Series): Simulated time series.
        low_flow_threshold (float): Threshold below which data is considered low-flow.

    Returns:
        dict: A dictionary with bias and KGE values for low-flow periods.
    """
    # Align observed and simulated data and drop NaNs
    observed, simulated = observed.align(simulated, join='inner')
    mask = observed.notna() & simulated.notna()
    observed = observed[mask]
    simulated = simulated[mask]
    # Apply low-flow mask
    low_flow_mask = observed < low_flow_threshold
    observed_low = observed[low_flow_mask]
    simulated_low = simulated[low_flow_mask]

    # If no valid low-flow data remains, return NaN for metrics
    if len(observed_low) == 0 or len(simulated_low) == 0:
        return {np.nan, np.nan}

    # Calculate bias and KGE for low-flow periods
    bias_low = compute_bias(observed_low, simulated_low)
    kge_low = compute_kge(observed_low, simulated_low)

    return bias_low, kge_low

'''small test area '''
stationsAreaVal = []
stationsAreaCorr = []
for station in stations_sel_unique['wellID'][:]:
    station_data = stations_sel[stations_sel['wellID'] == station]
    #find map values for same lon and lat
    lon = station_data['lon'].values[0]
    lat = station_data['lat'].values[0]
    map_sel_station = map_sel.sel(lon=lon, lat=lat, method='nearest')
    globgm_data = pd.DataFrame(map_sel_station.Band1.values, columns=['globgm_data'], index=pd.date_range(start='2010-01-01', end='2015-12-31', freq='M'))
    #merge globgm_data with station_data
    validation_data = pd.merge(station_data, globgm_data, left_index=True, right_index=True)  
    if validation_data.empty:
        continue
    else:
        anom_corr = compute_anomaly_correlation(validation_data, 'gwh_m', 'globgm_data') 
        corr = validation_data['gwh_m'].corr(validation_data['globgm_data'])
        bias = compute_bias(validation_data['gwh_m'], validation_data['globgm_data'])
        kge = compute_kge(validation_data['gwh_m'], validation_data['globgm_data'])
        bias_low, kge_low = compute_low_flow_metrics(validation_data['gwh_m'], validation_data['globgm_data'], np.percentile(validation_data['gwh_m'], 20))
        score_data = pd.DataFrame({'wellID': station, 'lon': lon, 'lat': lat, 'corr': corr, 'anom_corr': anom_corr, 'bias': bias, 'kge': kge, 'kge_low': kge_low, 'bias_low': bias_low}, index=[0])
        stationsAreaVal.append(validation_data)
        stationsAreaCorr.append(score_data)
stationsAreaVal_df = pd.concat(stationsAreaVal)
stationsAreaCorr_df = pd.concat(stationsAreaCorr)
stationsAreaVal_df.to_pickle("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area_inclglobgm.pkl")
stationsAreaCorr_df.to_pickle("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area_corr.pkl")

'''whole tile area'''
stationsAreaTile = []
stationsAreaTileCorr = []
#drop nan values in stations_all_unique wellID
stations_all_unique = stations_all_unique.dropna(subset=['wellID'])
for i, station in enumerate(stations_all_unique['wellID'][:]):
    print(i, station)
    station_data = stations[stations['wellID'] == station]
    #find map values for same lon and lat
    lon = station_data['lon'].values[0]
    lat = station_data['lat'].values[0]
    map_sel_station = mapwtd.sel(lon=lon, lat=lat, method='nearest')
    globgm_data = pd.DataFrame(map_sel_station.Band1.values, columns=['globgm_data'], index=pd.date_range(start='2010-01-01', end='2015-12-31', freq='M'))
    #merge globgm_data with station_data
    validation_data = pd.merge(station_data, globgm_data, left_index=True, right_index=True)  
    if validation_data.empty:
        print('validation data empty')
        continue
    elif validation_data['globgm_data'].isnull().all():
        print('globgm data empty')
        continue
    else:
        anom_corr = compute_anomaly_correlation(validation_data, 'gwh_m', 'globgm_data') 
        corr = validation_data['gwh_m'].corr(validation_data['globgm_data'])
        bias = compute_bias(validation_data['gwh_m'], validation_data['globgm_data'])
        kge = compute_kge(validation_data['gwh_m'], validation_data['globgm_data'])
        bias_low, kge_low = compute_low_flow_metrics(validation_data['gwh_m'], validation_data['globgm_data'], np.percentile(validation_data['gwh_m'], 20))
        score_data = pd.DataFrame({'wellID': station, 'lon': lon, 'lat': lat, 'corr': corr, 'anom_corr': anom_corr, 'bias': bias, 'kge': kge, 'kge_low': kge_low, 'bias_low': bias_low}, index=[0])
        stationsAreaTile.append(validation_data)
        stationsAreaTileCorr.append(score_data)
stationsAreaTile_df = pd.concat(stationsAreaTile)
stationsAreaTileCorr_df = pd.concat(stationsAreaTileCorr)
stationsAreaTile_df.to_pickle("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_inclglobgm.pkl")
stationsAreaTileCorr_df.to_pickle("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_corr.pkl")


# Plot the selected stations scores and map
plt.figure(figsize=(15, 10))
map_sel.Band1[0].plot(cmap='gray')
plt.scatter(stationsAreaCorr_df['lon'], stationsAreaCorr_df['lat'], c=stationsAreaCorr_df['anom_corr'], cmap='RdBu', label='Anom_Correlation')
plt.legend()
plt.colorbar()
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area_anom_corr.png")

plt.figure(figsize=(15, 10))
mapwtd.Band1[0].plot(cmap='gray')
plt.scatter(stationsAreaTileCorr_df['lon'], stationsAreaTileCorr_df['lat'], c=stationsAreaTileCorr_df['anom_corr'], cmap='RdBu', label='Anom_Correlation')
plt.legend()
plt.colorbar()
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_anom_corr.png")



colors =['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
bounds = [np.percentile(stationsAreaCorr_df['kge'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1]  # Define thresholds: poor, acceptable, good, best
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=len(bounds))
norm = BoundaryNorm(bounds, custom_cmap.N)
plt.figure(figsize=(15, 10))
map_sel.Band1[0].plot(cmap='gray')
scatter = plt.scatter(stationsAreaCorr_df['lon'], stationsAreaCorr_df['lat'], c=stationsAreaCorr_df['kge'], cmap=custom_cmap, norm=norm, label='KGE')
plt.legend()
cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[np.percentile(stationsAreaCorr_df['kge'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.ax.set_yticklabels(['Poor (<0)', 'Acceptable (0-0.5)', 'Good (>0.5)', 'Best (1)'])
cbar.set_label('KGE')
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area_kge.png")

colors =['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
bounds = [np.percentile(stationsAreaTileCorr_df['kge'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1]  # Define thresholds: poor, acceptable, good, best
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=len(bounds))
norm = BoundaryNorm(bounds, custom_cmap.N)
plt.figure(figsize=(15, 10))
mapwtd.Band1[0].plot(cmap='gray')
scatter = plt.scatter(stationsAreaTileCorr_df['lon'], stationsAreaTileCorr_df['lat'], c=stationsAreaTileCorr_df['kge'], cmap=custom_cmap, norm=norm, label='KGE')
plt.legend()
cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[np.percentile(stationsAreaTileCorr_df['kge'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.ax.set_yticklabels(['Poor (<0)', 'Acceptable (0-0.5)', 'Good (>0.5)', 'Best (1)'])
cbar.set_label('KGE')
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_kge.png")


colors = ['#8c510a', '#d8b365', '#f6e8c3', '#f5f5f5', '#c7eae5', '#5ab4ac', '#01665e']
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
minA = np.percentile(stationsAreaCorr_df['bias'],10)
maxA = np.percentile(stationsAreaCorr_df['bias'],90)
lim = max(abs(minA), abs(maxA))
norm = plt.Normalize(vmin=-lim, vmax=lim)

plt.figure(figsize=(15, 10))
map_sel.Band1[0].plot(cmap='gray')
scatter =plt.scatter(stationsAreaCorr_df['lon'], stationsAreaCorr_df['lat'], c=stationsAreaCorr_df['bias'], cmap=custom_cmap, norm=norm, label='Bias')
plt.legend()
cbar = plt.colorbar(scatter)
cbar.set_label('Bias')
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area_bias.png")


minT = np.percentile(stationsAreaTileCorr_df['bias'],10)
maxT = np.percentile(stationsAreaTileCorr_df['bias'],90)
limT = max(abs(minT), abs(maxT))
normT = plt.Normalize(vmin=-lim, vmax=lim)

plt.figure(figsize=(15, 10))
mapwtd.Band1[0].plot(cmap='gray')
scatter = plt.scatter(stationsAreaTileCorr_df['lon'], stationsAreaTileCorr_df['lat'], c=stationsAreaTileCorr_df['bias'], cmap=custom_cmap, norm=norm, label='Bias')
plt.legend()
cbar = plt.colorbar(scatter)
cbar.set_label('Bias')
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_bias.png")




colors =['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
bounds = [np.percentile(stationsAreaCorr_df['kge_low'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1]  # Define thresholds: poor, acceptable, good, best
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=len(bounds))
norm = BoundaryNorm(bounds, custom_cmap.N)
plt.figure(figsize=(15, 10))
map_sel.Band1[0].plot(cmap='gray')
scatter = plt.scatter(stationsAreaCorr_df['lon'], stationsAreaCorr_df['lat'], c=stationsAreaCorr_df['kge_low'], cmap=custom_cmap, norm=norm, label='kge_low')
plt.legend()
cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[np.percentile(stationsAreaCorr_df['kge_low'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.ax.set_yticklabels(['Poor (<0)', 'Acceptable (0-0.5)', 'Good (>0.5)', 'Best (1)'])
cbar.set_label('kge_low')
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_small_area_kge_low.png")


colors =['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
bounds = [np.nanpercentile(stationsAreaTileCorr_df['kge_low'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1]  # Define thresholds: poor, acceptable, good, best
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=len(bounds))
norm = BoundaryNorm(bounds, custom_cmap.N)
plt.figure(figsize=(15, 10))
mapwtd.Band1[0].plot(cmap='gray')
scatter = plt.scatter(stationsAreaTileCorr_df['lon'], stationsAreaTileCorr_df['lat'], c=stationsAreaTileCorr_df['kge_low'], cmap=custom_cmap, norm=norm, label='KGE_low')
plt.legend()
cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[np.nanpercentile(stationsAreaTileCorr_df['kge_low'], 10), 0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.ax.set_yticklabels(['Poor (<0)', 'Acceptable (0-0.5)', 'Good (>0.5)', 'Best (1)'])
cbar.set_label('kge_low')
plt.savefig("/eejit/home/hausw001/HybGGM/hybGGM_test/src/globgm_validation/selected_stations_tile48_kge_low.png")