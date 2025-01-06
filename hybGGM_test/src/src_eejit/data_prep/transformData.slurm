#!/bin/bash
#SBATCH --job-name="DataToNetCDF"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=transformData.out
#SBATCH --error=transformData.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
module load tools
module load GDAL
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "tile unzipped: tile_048-163"
echo "folder already exists: /scratch/depfg/hausw001/data/globgm/tiles_input//tile_048-163/transient/netcdf_maps_own"
gdal_translate -of NetCDF /scratch/depfg/hausw001/data/globgm/tiles_input//tile_048-163/transient/maps/surface_water_bed_elevation_used_197006.map /scratch/depfg/hausw001/data/globgm/tiles_input//tile_048-163/transient/netcdf_maps_own/surface_water_bed_elevation_used_197006.nc 

echo "SBATCH job finished"
exit 0
