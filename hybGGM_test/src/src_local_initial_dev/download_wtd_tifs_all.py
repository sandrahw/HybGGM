'''create list of dates of last day of the month for years 1958-2016, e.g.19580131, 19580228, 19580331, ...'''
import os
import calendar
import datetime

dates = []
for year in range(1958, 2016):
    for month in range(1, 13):
        dates.append(datetime.datetime(year, month, calendar.monthrange(year, month)[1]).strftime('%Y%m%d'))

download_folder = "/scratch/depfg/hausw001/data/globgm/output_selection"
download_site = 'https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/transient_1958-2015/'
'''Check if file is already there, if not, download it'''
for date in dates[:1]:
    #check if data.tif is already there
    if os.path.exists(download_folder + '/globgm-wtd-' + date + '.tif'):
        print('globgm-wtd-' + date + '.tif already exists')
        continue
    else:
        print('Downloading globgm-wtd-' + date + '.tif')
        downloadfile = download_site + 'globgm-wtd-' + date + '.tif'
        os.system('wget ' + downloadfile)
        print('Downloaded globgm-wtd-' + date + '.tif')


