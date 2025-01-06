#!/bin/bash
#SBATCH --job-name="hybggm_test_halftile"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=hyperparam_halftile_test.out
#SBATCH --error=hyperparam_halftile_test.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 64 half start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 64 UNet6 half 
echo "UNet6 with 0.4 0.3 50 0.0001 64 half done at $(date)" 

echo "SBATCH job finished"
exit 0
