#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_U6_360_ext"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_small_randomarea_U6_360_ext.out
#SBATCH --error=hyperparam_small_randomarea_U6_360_ext.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 1 50 1e-05 32 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 1e-05 32 360 UNet6
echo "UNet6 with 1 50 1e-05 32 360 done at $(date)" 
echo "UNet6 with 1 50 1e-05 64 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 1e-05 64 360 UNet6
echo "UNet6 with 1 50 1e-05 64 360 done at $(date)" 
echo "UNet6 with 1 50 1e-05 128 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 1e-05 128 360 UNet6
echo "UNet6 with 1 50 1e-05 128 360 done at $(date)" 

echo "SBATCH job finished"
exit 0
