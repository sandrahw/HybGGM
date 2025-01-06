#!/bin/bash
#SBATCH --job-name="hybggm_test_extended_area_focusBS"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=hyperparam_extended_area_focusBS.out
#SBATCH --error=hyperparam_extended_area_focusBS.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 16 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 16 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 16 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 32 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 32 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 32 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 64 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 64 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 64 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 128 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 128 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 128 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 256 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 256 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 256 done at $(date)" 

echo "SBATCH job finished"
exit 0
