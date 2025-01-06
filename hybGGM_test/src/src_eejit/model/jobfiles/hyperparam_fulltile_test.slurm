#!/bin/bash
#SBATCH --job-name="hybggm_test_fulltile"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_fulltile_test.out
#SBATCH --error=hyperparam_fulltile_test.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 64 full start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 64 UNet6 full 
echo "UNet6 with 0.4 0.3 50 0.0001 64 full done at $(date)" 

echo "SBATCH job finished"
exit 0
