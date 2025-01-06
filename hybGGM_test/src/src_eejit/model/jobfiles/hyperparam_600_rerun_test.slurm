#!/bin/bash
#SBATCH --job-name="hybggm_600_half"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_600_rerun_test.out
#SBATCH --error=hyperparam_600_rerun_test.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 5 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 5 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 5 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 10 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 10 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 10 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 20 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 20 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 20 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 50 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 50 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 50 large done at $(date)"
echo "UNet6 with 0.4 0.3 50 0.001 20 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 20 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 20 large done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 50 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 50 UNet2 large 
echo "UNet2 with 0.4 0.3 50 0.0001 50 large done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 20 large start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 20 UNet2 large 
echo "UNet2 with 0.4 0.3 50 0.001 20 large done at $(date)" 

echo "SBATCH job finished"
exit 0
