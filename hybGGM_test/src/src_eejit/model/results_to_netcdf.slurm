#!/bin/bash
#SBATCH --job-name="results_to_netcdf"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=results_to_netcdf.out
#SBATCH --error=results_to_netcdf.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
python src/model/val_areas_to_netcdf.py

echo "SBATCH job finished"
exit 0
