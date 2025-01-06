#!/bin/bash
#SBATCH --job-name="hybggm_test_small_area_focusBSall"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=hyperparam_small_area_focusBSall.out
#SBATCH --error=hyperparam_small_area_focusBSall.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 32 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 64 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 128 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 256 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 256 small done at $(date)" 

echo "SBATCH job finished"
exit 0
