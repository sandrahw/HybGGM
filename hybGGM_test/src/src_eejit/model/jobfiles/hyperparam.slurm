#!/bin/bash
#SBATCH --job-name="hybggm_test_extended"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=hyperparam.out
#SBATCH --error=hyperparam.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet2 with 0.4 0.3 50 0.0001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 1 UNet2
echo "UNet2 with 0.4 0.3 50 0.0001 1 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 1 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 1 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 1 UNet2
echo "UNet2 with 0.4 0.3 50 0.001 1 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 1 UNet6
echo "UNet6 with 0.4 0.3 50 0.001 1 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 2 UNet2
echo "UNet2 with 0.4 0.3 50 0.0001 2 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 2 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 2 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 2 UNet2
echo "UNet2 with 0.4 0.3 50 0.001 2 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 2 UNet6
echo "UNet6 with 0.4 0.3 50 0.001 2 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 4 UNet2
echo "UNet2 with 0.4 0.3 50 0.0001 4 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 4 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 4 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 4 UNet2
echo "UNet2 with 0.4 0.3 50 0.001 4 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 4 UNet6
echo "UNet6 with 0.4 0.3 50 0.001 4 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 10 UNet2
echo "UNet2 with 0.4 0.3 50 0.0001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 10 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 10 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 10 UNet2
echo "UNet2 with 0.4 0.3 50 0.001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 10 UNet6
echo "UNet6 with 0.4 0.3 50 0.001 10 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 20 UNet2
echo "UNet2 with 0.4 0.3 50 0.0001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 20 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 20 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 20 UNet2
echo "UNet2 with 0.4 0.3 50 0.001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 20 UNet6
echo "UNet6 with 0.4 0.3 50 0.001 20 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 50 UNet2
echo "UNet2 with 0.4 0.3 50 0.0001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 50 UNet6
echo "UNet6 with 0.4 0.3 50 0.0001 50 done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 50 UNet2
echo "UNet2 with 0.4 0.3 50 0.001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 50 UNet6
echo "UNet6 with 0.4 0.3 50 0.001 50 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 1 UNet2
echo "UNet2 with 0.4 0.3 100 0.0001 1 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 1 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 1 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 1 UNet2
echo "UNet2 with 0.4 0.3 100 0.001 1 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 1 UNet6
echo "UNet6 with 0.4 0.3 100 0.001 1 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 2 UNet2
echo "UNet2 with 0.4 0.3 100 0.0001 2 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 2 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 2 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 2 UNet2
echo "UNet2 with 0.4 0.3 100 0.001 2 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 2 UNet6
echo "UNet6 with 0.4 0.3 100 0.001 2 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 4 UNet2
echo "UNet2 with 0.4 0.3 100 0.0001 4 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 4 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 4 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 4 UNet2
echo "UNet2 with 0.4 0.3 100 0.001 4 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 4 UNet6
echo "UNet6 with 0.4 0.3 100 0.001 4 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 10 UNet2
echo "UNet2 with 0.4 0.3 100 0.0001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 10 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 10 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 10 UNet2
echo "UNet2 with 0.4 0.3 100 0.001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 10 UNet6
echo "UNet6 with 0.4 0.3 100 0.001 10 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 20 UNet2
echo "UNet2 with 0.4 0.3 100 0.0001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 20 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 20 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 20 UNet2
echo "UNet2 with 0.4 0.3 100 0.001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 20 UNet6
echo "UNet6 with 0.4 0.3 100 0.001 20 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 50 UNet2
echo "UNet2 with 0.4 0.3 100 0.0001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 50 UNet6
echo "UNet6 with 0.4 0.3 100 0.0001 50 done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 50 UNet2
echo "UNet2 with 0.4 0.3 100 0.001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 50 UNet6
echo "UNet6 with 0.4 0.3 100 0.001 50 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 1 UNet2
echo "UNet2 with 0.4 0.3 200 0.0001 1 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 1 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 1 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 1 UNet2
echo "UNet2 with 0.4 0.3 200 0.001 1 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 1 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 1 UNet6
echo "UNet6 with 0.4 0.3 200 0.001 1 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 2 UNet2
echo "UNet2 with 0.4 0.3 200 0.0001 2 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 2 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 2 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 2 UNet2
echo "UNet2 with 0.4 0.3 200 0.001 2 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 2 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 2 UNet6
echo "UNet6 with 0.4 0.3 200 0.001 2 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 4 UNet2
echo "UNet2 with 0.4 0.3 200 0.0001 4 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 4 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 4 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 4 UNet2
echo "UNet2 with 0.4 0.3 200 0.001 4 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 4 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 4 UNet6
echo "UNet6 with 0.4 0.3 200 0.001 4 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 10 UNet2
echo "UNet2 with 0.4 0.3 200 0.0001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 10 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 10 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 10 UNet2
echo "UNet2 with 0.4 0.3 200 0.001 10 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 10 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 10 UNet6
echo "UNet6 with 0.4 0.3 200 0.001 10 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 20 UNet2
echo "UNet2 with 0.4 0.3 200 0.0001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 20 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 20 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 20 UNet2
echo "UNet2 with 0.4 0.3 200 0.001 20 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 20 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 20 UNet6
echo "UNet6 with 0.4 0.3 200 0.001 20 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 50 UNet2
echo "UNet2 with 0.4 0.3 200 0.0001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 50 UNet6
echo "UNet6 with 0.4 0.3 200 0.0001 50 done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 50 UNet2
echo "UNet2 with 0.4 0.3 200 0.001 50 done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 50 start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.001 50 UNet6
echo "UNet6 with 0.4 0.3 200 0.001 50 done at $(date)" 

echo "SBATCH job finished"
exit 0
