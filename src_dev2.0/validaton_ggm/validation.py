'''script to validate GLOBGM results for Europe case and focus on droughts'''

import pandas as pd
import glob
import xarray as xr
import numpy as np

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

# globgm data
globgm_raw = "/scratch/depfg/hausw001/data/globgm/output/netcdf_maps_own/"
globgm_raw_files = glob.glob(globgm_raw + "*.nc")
# oberservations data
# obs_raw = "/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered.csv"

# #load obs data into dataframe from csv, parse date, convert wellID to int, lon and lat to float
# obs_df = pd.read_csv(obs_raw, parse_dates=['date'], dtype={'wellID': int, 'lon': float, 'lat': float, 'gwh_m': float})

# #for every unique wellID check if that wellID has at least 10 years of data
# wellIDs = obs_df['wellID'].unique()
# obs_df_sel = []
# for wellID in wellIDs[:]:
#     obs_sel = obs_df[obs_df['wellID'] == wellID]
#     #set date as index for resampling to monthly data
#     obs_sel.set_index('date', inplace=True)
#     # resample gwh_m to monthly data
#     obs_sel = obs_sel.resample('M').mean()
#     #check how many nan values are in the data
#     nan_count = obs_sel['gwh_m'].isna().sum()
#     #check how many non nan values are in the data
#     non_nan_count = obs_sel['gwh_m'].notna().sum()
#     #check how many years of data are in the data
#     years = obs_sel.index.year.unique()
#     #if there are at least 10 year of data, save this df to a new dataframe
#     if non_nan_count >= 120:
#         obs_df_sel.append(obs_sel)
# obs_df_selection = pd.concat(obs_df_sel)
# obs_df_selection = obs_df_selection.dropna(subset=['wellID'])
# obs_df_selection.to_csv("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year.csv",)

reload = pd.read_csv("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year.csv", index_col=0)
#check if there are wellIDs which are nan
test= reload[reload['wellID'].isna()]
#drop all rows where wellID is nan
obs_df_sel = reload.dropna(subset=['wellID'])

df_globgm_obs = obs_df_sel
df_globgm_obs['globgm_data'] = None
df_globgm_obs.index = pd.to_datetime(df_globgm_obs.index, format='%Y-%m-%d')
#for every well, location and date, find the corresponding cell in the globgm data, which is loaded file by file
#for every file in globgm_raw_files
sortfiles = sorted(globgm_raw_files, key=lambda x: x.split("/")[-1].split("_")[0])
full_data = []
#divide 696 files into 4 parts
# sortfiles = sortfiles[:174]
# sortfiles = sortfiles[175:348]
# sortfiles = sortfiles[349:522]
# sortfiles = sortfiles[523:]
for i, file in enumerate(sortfiles[175:348]):
    #define date of file
    fileN = file.split("/")[-1].split("_")[0]
    dateN = fileN.split("-")[2][:-3]
    date = pd.to_datetime(dateN, format='%Y%m%d')
    print(i, "date", date, "in df_globgm_obs: ", df_globgm_obs[df_globgm_obs.index == date].shape[0])
    globgm_data = xr.open_dataset(file) 

    #check if date is in df_globgm_obs or not
    if df_globgm_obs[df_globgm_obs.index == date].empty:
        print('date not in obs')
    else:
        print('date in obs')
        #find all rows where the date is a match
        date_sel = df_globgm_obs[df_globgm_obs.index == date]
        date_sel = date_sel.reset_index()

        #for every row where the date is a match, find the corresponding cell in the globgm data and save the value to a the new column 'globgm'
        for index, row in date_sel[:].iterrows():
            # print(index, row)            #get wellID, lon and lat from row
            lon = row['lon']
            lat = row['lat']
            #find corresponding cell in globgm data
            globgm_data_sel = globgm_data.sel(lon=lon, lat=lat, method='nearest')
            # print(globgm_data_sel.Band1.values)
            #save the globgm data to the new column at the correct row with the same well so that every row has the correct value based on the location
            date_sel.loc[index, 'globgm_data'] = globgm_data_sel.Band1.values
        #append the date_sel to the full_data list
        full_data.append(date_sel)
full_data_df = pd.concat(full_data)
full_data_df.to_csv("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p2.csv")
full_data_df.to_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p2.pkl")



full_p1 = pd.read_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p1.pkl")
full_p2 = pd.read_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p2.pkl")
full_p3 = pd.read_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p3.pkl")
full_p4 = pd.read_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p4.pkl")

full_data_df = pd.concat([full_p1, full_p2, full_p3, full_p4])
unique_wellIDs = full_data_df['wellID'].unique()

#sort fulldata by wellID and date
full_data_df = full_data_df.sort_values(by=['wellID', 'date'])

stationsEuropeCorr = []
for i, well in enumerate(full_data_df['wellID'][:1]):
    validation_data = full_data_df[full_data_df['wellID'] == well]
    anom_corr = compute_anomaly_correlation(validation_data, 'gwh_m', 'globgm_data') 
    if validation_data.empty:
        print('validation data empty')
        continue
    elif validation_data['globgm_data'].isnull().all():
        print('globgm data empty')
        continue
    else:
        corr = validation_data['gwh_m'].corr(validation_data['globgm_data'])
        bias = compute_bias(validation_data['gwh_m'], validation_data['globgm_data'])
        kge = compute_kge(validation_data['gwh_m'], validation_data['globgm_data'])
        bias_low, kge_low = compute_low_flow_metrics(validation_data['gwh_m'], validation_data['globgm_data'], np.percentile(validation_data['gwh_m'], 20))
        score_data = pd.DataFrame({'wellID': well, 'lon': lon, 'lat': lat, 'corr': corr, 'anom_corr': anom_corr, 'bias': bias, 'kge': kge, 'kge_low': kge_low, 'bias_low': bias_low}, index=[0])
        stationsEuropeCorr.append(score_data)
stationsEuropeCorr_df = pd.concat(stationsEuropeCorr)
stationsEuropeCorr_df.to_csv("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p1_corr.csv")
stationsEuropeCorr_df.to_pickle("/scratch/depfg/hausw001/data/globgm/output/observations_Europe/observed_gw_layers_europefiltered_monthly_min10year_globgm_p1_corr.pkl")


