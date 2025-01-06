import xarray as xr
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import pandas as pd
import os
import numpy

target = ['wtd', 'deltawtd']
run = 'UNet2_50_0.0001_32_1.0'
# folders = ['UNet6_50_0.0001_16_1.0', 'UNet2_50_0.0001_8_1.0', 'UNet2_50_0.0001_4_1.0', 'UNet2_50_0.0001_16_1.0', 'UNet2_50_0.0001_32_1.0', 'UNet6_50_0.0001_4_1.0', 'UNet6_50_0.0001_8_1.0']
for tar in target[:1]:
    print(tar)
    ncfolder = '/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_fulltile_180_101010_limitedInpSel_%s/%s/ncmaps/' %(tar,run)
    ncfiles = glob.glob('%s/results_*.nc'%ncfolder)
    #select only results not target
    ncfiles = [f for f in ncfiles if "target" not in f]
    ncfilestarget = glob.glob('%s/results_*target.nc' %ncfolder)
    #order by date
    ncfiles.sort()
    ncfilestarget.sort()

    if os.path.exists('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/total_target_%s_run.npy' %tar):
        print('totat_target.npy exists')
        totaldatat = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/total_target_%s_run.npy' %tar)
        tott = True
    else:
        tott = False
        totaldatat = []
    if os.path.exists('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_total_prediction_%s_run.npy' %(run, tar)):
        print('total_prediction.npy exists')
        totaldata = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_total_prediction_%s_run.npy' %(run, tar))
    else:
        totaldata = []
        for nf, nft in zip(ncfiles[:], ncfilestarget[:]):
            date = nf.split("_")[-1].split(".")[0]
            load = xr.open_dataset(nf)
            totaldata.append(load['mask'])
            loadt = xr.open_dataset(nft)
            if not tott:
                # print('append target')
                totaldatat.append(loadt['mask'])    
        totaldata = xr.concat(totaldata, dim='time')
        np.save('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_total_prediction_%s_run.npy' %(run, tar), totaldata)
        totaldatat = xr.concat(totaldatat, dim='time')
        np.save('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/total_target_%s_run.npy' %(tar), totaldatat) 

    # totaldatat = xr.DataArray(totaldatat, dims=['time', 'lat', 'lon'])
    # totaldata = xr.DataArray(totaldata, dims=['time', 'lat', 'lon'])

    NRMSE = np.sqrt((np.nanmean((totaldatat-totaldata)**2, axis=(1,2))))/np.nanstd(totaldatat, axis=(1,2))
    # plt.plot(NRMSE)
    np.save('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_nrmse_per_timestep_%s_run.npy' %(run, tar), NRMSE)
    
    NRMSE_per_cell = np.sqrt((np.nanmean((totaldatat-totaldata)**2, axis=0)))/np.nanstd(totaldatat, axis=0)
    np.save('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_nrmse_per_cell_%s_run.npy' %(run,tar), NRMSE_per_cell)
    # print(NRMSE_per_cell)
    # print('min', np.nanmin(NRMSE_per_cell), 'mean', np.nanmean(NRMSE_per_cell), 'max',np.nanmax(NRMSE_per_cell))

# for tar in target[:1]:
#     print(tar)

totaldata_sel = totaldata[:, 1100, 1100]
totaldatat_sel = totaldatat[:, 1100, 1100]
NRMSE_per_cell_sel = np.sqrt((np.nanmean((totaldatat_sel-totaldata_sel)**2, axis=0)))
std = np.nanstd(totaldatat_sel)
std = np.nanstd(totaldata_sel)

plt.figure()
plt.plot(totaldata_sel, label='prediction')
plt.plot(totaldatat_sel, label='target')
plt.legend()    

totaldata = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_total_prediction_%s_run.npy' %(run, tar))
totaldata = xr.DataArray(totaldata, dims=['time', 'lat', 'lon'])
annual_mean_wtd = totaldata.mean(dim='time')
annual_mean_wtd.name = 'wtd [m]'
min_value = annual_mean_wtd.min()
max_value = annual_mean_wtd.max()
import matplotlib.colors as mcolors
bounds = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3000]  # Example boundaries
cmap = plt.get_cmap('viridis')  # Choose a colormap
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the data
plt.figure()
ax = annual_mean_wtd.plot(cmap=cmap, norm=norm)
# Remove the axes
ax.axes.set_axis_off()
plt.title('mean water table depth 1958-2015, hybrid model')
# ax.set_title('mean wtd 1958-2015')
# Optionally, also hide the colorbar label and ticks
# ax.colorbar.ax.set_visible(False)
plt.show()
plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/annual_mean_wtd.png')

totaldatat = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/total_target_%s_run.npy' %tar)
totaldatat = xr.DataArray(totaldatat, dims=['time', 'lat', 'lon'])
annual_mean_wtd_target = totaldatat.mean(dim='time')
annual_mean_wtd_target.name = 'wtd [m]'
min_value = annual_mean_wtd_target.min()
max_value = annual_mean_wtd_target.max()

plt.figure()
ax = annual_mean_wtd_target.plot(cmap=cmap, norm=norm)
# Remove the axes
ax.axes.set_axis_off()
plt.title('mean water table depth 1958-2015, GLOBGM v1.0')
plt.show()
plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/annual_mean_wtd_target_globgm.png')

#TODO normalise the difference
diffnorm = (annual_mean_wtd - annual_mean_wtd_target)/ annual_mean_wtd_target
# diff = annual_mean_wtd - annual_mean_wtd_target #ml - physb,  100-110 =-10 (ml is underestimating)
diffnorm.name = 'relative difference in wtd'
min_value = diffnorm.min()
max_value = diffnorm.max()
# #check how many values are in total
# print('total values', diffnorm.count())
# #print how many values are nan
# print('nan values', diffnorm.isnull().sum())
# #print how many times max value present
# print('max value', diffnorm.where(diffnorm == diffnorm.max()).count())
# #print how many times min value present
# print('min value', diffnorm.where(diffnorm == diffnorm.min()).count())

# # get 99 and 1 percentile
# print('99 percentile', diffnorm.quantile(0.99).values)
# print('1 percentile', diffnorm.quantile(0.01).values)

# bounds = [-500, -250, -100, -50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 250, 500]  # Example boundaries
bounds = [-10, -5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 5, 10]  # Example boundaries
cmap = plt.get_cmap('bwr')  # Choose a colormap
norm = mcolors.BoundaryNorm(bounds, cmap.N)

ax = diffnorm.plot(cmap=cmap, norm=norm)
# Remove the axes
ax.axes.set_axis_off()
plt.title('relative difference in water table depth \n hybrid vs GLOBGM v1.0, 1958-2015')
plt.show()
plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/annual_mean_wtd_diff.png')

'''selecting smaller areas for drought analysis'''
totaldatat = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/total_target_%s_run.npy' %tar)
ncfolder = '/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_fulltile_180_101010_limitedInpSel_%s/%s/ncmaps/' %(tar,run)
ncfiles = glob.glob('%s/results_*.nc'%ncfolder)
temp = xr.open_dataset(ncfiles[0])
# transfere coordinates of xarray to totaldatat
totaldatat = xr.DataArray(totaldatat, dims=['time', 'lat', 'lon'])
totaldatat = totaldatat.assign_coords(lon=temp.lon)
totaldatat = totaldatat.assign_coords(lat=temp.lat)
# totaldatat time should be the last day of every month from 1958-2015
totaldatat = totaldatat.assign_coords(time=pd.date_range(start='1958-02-28', end='2015-12-31', freq='M'))
totaldatat.name = 'wtd [m]'
totaldatat[0,:,:].plot.imshow()
# select area based on lat lon
lat = 50, 52
lon = 4, 6
case1 = totaldatat.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))

lat2 = 46, 48	
lon2 = 1,3
case2 = totaldatat.sel(lat=slice(lat2[0], lat2[1]), lon=slice(lon2[0], lon2[1]))
#plot a square of case 1 in totaldatat based on lat lon
import matplotlib.colors as mcolors
bounds = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3000]  # Example boundaries
cmap = plt.get_cmap('viridis')  # Choose a colormap
norm = mcolors.BoundaryNorm(bounds, cmap.N)
plt.figure()
totaldatat[0,:,:].plot.imshow(cmap=cmap, norm=norm)
plt.title(' ')
plt.plot([lon[0], lon[1], lon[1], lon[0], lon[0]], [lat[0], lat[0], lat[1], lat[1], lat[0]], 'orange')
plt.plot([lon2[0], lon2[1], lon2[1], lon2[0], lon2[0]], [lat2[0], lat2[0], lat2[1], lat2[1], lat2[0]], 'b-')

case1_timeseries = case1.mean(dim=('lat', 'lon'))
case2_timeseries = case2.mean(dim=('lat', 'lon'))

#plot timeseries of case1 with time on x axis
#define variable threshold for case1 and case2
thres_c1 = np.percentile(case1_timeseries, 20)
thres_c2 = np.percentile(case2_timeseries, 20)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
case1_timeseries.plot(ax=ax1, color='orange')
case2_timeseries.plot(ax=ax1, color='blue')
ax1.plot(case2_timeseries.time, thres_c1*np.ones(len(case1_timeseries)), 'r--')
ax1.plot(case2_timeseries.time, thres_c2*np.ones(len(case2_timeseries)), 'r--')
ax1.set_ylabel('wtd [GLOBGM v1.0 [m]]')
ax1.set_ylim(10, 30)
# ax1.set_xlabel('time')
ax1.legend(['case1', 'case2'])

#find the time when time periods where values below threshold
case1_timeseries_below = case1_timeseries.where(case1_timeseries < thres_c1)
case2_timeseries_below = case2_timeseries.where(case2_timeseries < thres_c2)

ax2.plot(case2_timeseries.time, case1_timeseries_below, color='orange')
ax2.plot(case2_timeseries.time, case2_timeseries_below, color='blue')
ax2.set_ylim(10, 30)
ax2.set_ylabel('wtd below threshold [GLOBGM v1.0 [m]]')
ax2.set_xlabel('time')


'''making figures'''
rmse_per_date_list = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/*_nrmse_per_time*.npy')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
for rmse_per_date in rmse_per_date_list:
    runName = rmse_per_date.split('/')[-1][:-8]
    rmse_per_date_all = np.load(rmse_per_date)
    if 'wtd' in runName:
        print(runName)
        print(rmse_per_date_all.mean())
        ax1.plot(rmse_per_date_all, alpha=0.5, label=runName)
        ax1.set_title('NRMSE per timestep target wtd')
    else:
        print(runName)
        print(rmse_per_date_all.mean())
        ax2.plot(rmse_per_date_all, alpha=0.5, label=runName)
        ax2.set_title('NRMSE per timestep delta wtd')
ax1.legend()
ax2.legend()
ax1.set_ylim(0, 0.15)
ax2.set_ylim(0, 0.15)
plt.xticks(rotation=45)
plt.xticks(np.arange(0, len(rmse_per_date_all), 60))
plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/rmse_per_date_all.png')

for rmse_per_date in rmse_per_date_list:
    runName = rmse_per_date.split('/')[-1][:-8]
    rmse_per_date_all = np.load(rmse_per_date)
    plt.figure()
    plt.plot(rmse_per_date_all, alpha=0.5, label=runName)
    plt.legend()    
    plt.xticks(rotation=45)
    plt.xticks(np.arange(0, len(rmse_per_date_all), 60))
    plt.ylim(0, 0.15)
    plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_rmse_per_date.png' %runName)

 
rmse_per_cell_list = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/*_nrmse_per_cell*.npy')   
for rmse_cel in rmse_per_cell_list[:]:
    runName = rmse_cel.split('/')[-1][:-8]
    rmse_cell = np.load(rmse_cel)
    vmin = np.nanmin(rmse_cell)
    vmax = np.nanmax(rmse_cell)
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(rmse_cell)
    plt.colorbar(shrink=0.4)
    plt.gca().invert_yaxis()
    plt.clim(vmin, vmax)
    # print first few numbers of vmin and vmax
    # print('rmse min %0.3f max %0.3f' %(vmin, vmax))
    plt.title('rmse min %0.3f max %0.3f' %(vmin, vmax))
    plt.subplot(1,2,2)
    plt.imshow(rmse_cell, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(shrink=0.4)
    plt.gca().invert_yaxis()
    plt.title('%s' %runName)
    plt.tight_layout()
    plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_rmse_cell_new.png' %runName)
    plt.close()



total_prediction_list = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/*_total_prediction.npy')
total_target = np.load('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/total_target.npy')
for tot_pred in total_prediction_list[:]:
    runName = tot_pred.split('/')[-1][:-17]
    total_prediction = np.load(tot_pred)

    vminp = np.nanmin(total_prediction)
    vmaxp = np.nanmax(total_prediction)
    vmint = np.nanmin(total_target)
    vmaxt = np.nanmax(total_target)
    vmin = min(vminp, vmint)
    vmax = max(vmaxp, vmaxt)
 
    plt.scatter(total_prediction[0,:,:].flatten(), total_target[0,:,:].flatten(), marker='o', facecolors='none', edgecolors='r')
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.title('prediction vs target %s' %runName)
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    #plot diagonal line
    plt.plot([vmin, vmax], [vmin, vmax], 'k-')
    plt.savefig('/eejit/home/hausw001/HybGGM/hybGGM_test/results/analysis_tile48/%s_prediction_target_scatter.png' %runName)
    plt.close
