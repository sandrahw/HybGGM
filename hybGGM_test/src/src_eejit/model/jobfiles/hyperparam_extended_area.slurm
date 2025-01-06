#!/bin/bash
#SBATCH --job-name="hybggm_test_extended_area"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=hyperparam_extended_area.out
#SBATCH --error=hyperparam_extended_area.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 5 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 5 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 5 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 10 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 20 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 50 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 5 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 5 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 5 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 10 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 20 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 50 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 5 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 5 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 5 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 10 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 20 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 50 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 50 done at $(date)" 

echo "SBATCH job finished"
exit 0
