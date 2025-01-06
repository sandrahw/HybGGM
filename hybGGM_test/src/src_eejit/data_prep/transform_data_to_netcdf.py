import numpy as np
import os
import glob

print(os.getcwd())
general_path = '/scratch/depfg/hausw001/data/globgm/tiles_input/'
#open file connection to generate hyperparam.sh
with open('transformData.slurm', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name="DataToNetCDF"\n')
    f.write('#SBATCH --ntasks 16\n')
    f.write('#SBATCH --time 72:00:00 \n')
    # f.write('#SBATCH --cpus-per-task=8\n')
    f.write('#SBATCH --output=transformData.out\n') # BRAM example: SBATCH --output ./log/hybgmm_test.txt
    f.write('#SBATCH --error=transformData.err\n')# BRAM example: SBATCH --error ./log/hybgmm_test.txt
    f.write('#SBATCH --partition defq\n')
    f.write('\n')
    f.write('echo "SBATCH job"\n')
    f.write('echo "Started $(date \'+%d/%m/%Y %H:%M:%S\')"\n')
    # f.write('echo "Working directory $(pwd)"\n')
    f.write('\n')
    f.write('unset PYTHONHOME\n')
    f.write('module load tools\n')
    f.write('module load GDAL\n')
    f.write('cd /eejit/home/hausw001/HybGGM/hybGGM_test\n') #ejit path to hybGGM_test
    f.write('echo "Working directory $(pwd)"\n')
    f.write('conda activate env_hybGMM \n')

    tilenr = np.arange(0, 164, 1)
    tilenames = ['tile_%03d-163' %(i) for i in tilenr]
    for tile in tilenames[48:49]:
    #check if unzipped folder exists
        if not os.path.exists('%s/%s' %(general_path, tile)):
            f.write('echo "no unzipped folder for tile %s"\n' %(tile))
            continue
        else:
            f.write('echo "tile unzipped: %s"\n' %(tile))
        input_maps = '%s/%s/transient/maps' %(general_path, tile)  
        input_netcdfs = '%s/%s/transient/netcdf_maps_own' %(general_path, tile)  

        if not os.path.exists(input_netcdfs):
            os.makedirs(input_netcdfs)  
            f.write('echo "created folder: %s"\n' %(input_netcdfs))
        else:
            f.write('echo "folder already exists: %s"\n' %(input_netcdfs))
        
        files = glob.glob('%s/*.map' %(input_maps))
        for file in files[:1]:
                fileN = str.split(file, '/')[-1]
                fileN = fileN[:-4]
                # check if file already excist in input_netcdfs folder
                if glob.glob('%s/%s.nc' %(input_netcdfs,fileN)):
                    f.write('echo "file %s already exists" \n' %(fileN))
                    continue
                # Define the output path and filename
                output_path = f'{input_netcdfs}/{fileN}.nc'
                f.write('gdal_translate -of NetCDF %s %s \n' %(file, output_path))
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




