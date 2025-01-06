#!/bin/bash
#SBATCH --job-name="hybggm_rerun_smallU2"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_smallU2_rerun_test.out
#SBATCH --error=hyperparam_smallU2_rerun_test.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet2 with 0.4 0.3 10 0.0001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.0001 1 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.0001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.001 1 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.0001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.0001 4 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.0001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.001 4 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.0001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.0001 10 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.0001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.001 10 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 10 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 10 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 10 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.0001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.0001 1 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.0001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.001 1 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.0001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.0001 4 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.0001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.001 4 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.0001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.0001 10 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.0001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.001 10 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 20 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 20 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 20 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 1 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.0001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 1 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 4 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.0001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 4 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 10 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.0001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 10 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 1 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.0001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 1 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.005 1 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 4 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.0001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 4 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.001 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.005 4 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0001 10 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.0001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.001 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.001 10 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.001 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.005 10 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.005 10 small done at $(date)" 

echo "SBATCH job finished"
exit 0
