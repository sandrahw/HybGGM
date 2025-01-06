#!/bin/bash
#SBATCH --job-name="hybggm_test"
#SBATCH --ntasks 16"
#SBATCH --time 01:00:00 
#SBATCH --output=hyperparam.out
#SBATCH --error=hyperparam.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
python src/run_hyperparam_testing.py 0.4 0.3 10 0.0001 1 UNet2

echo "SBATCH job finished"
exit 0
