#!/bin/bash
#SBATCH --job-name="hybggm_smallU2U6_rerun"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_smallU2U6_rerun.out
#SBATCH --error=hyperparam_smallU2U6_rerun.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet2 with 0.4 0.3 50 0.0005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0005 1 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.0005 1 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0005 1 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0005 1 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 2 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0001 2 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 2 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 2 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.001 2 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 2 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0005 4 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.0005 4 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0005 4 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 50 0.0005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0005 10 UNet2 small
echo "UNet2 with 0.4 0.3 50 0.0005 10 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 50 0.0005 10 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0005 1 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.0005 1 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0005 1 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0005 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0005 4 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.0005 4 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0005 4 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0005 4 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0005 4 small done at $(date)" 
echo "UNet2 with 0.4 0.3 100 0.0005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0005 10 UNet2 small
echo "UNet2 with 0.4 0.3 100 0.0005 10 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0005 10 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 100 0.0005 10 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0005 10 small done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 1 UNet2 small
echo "UNet2 with 0.4 0.3 200 0.0001 1 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0001 1 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 1 small done at $(date)" 
echo "UNet2 with 0.4 0.3 200 0.0005 1 small start at $(date)" 
python src/model/run_hyperparam_testing.py 0.4 0.3 200 0.0005 1 UNet2 small
echo "UNet2 with 0.4 0.3 200 0.0005 1 small done at $(date)" 
echo "SBATCH job finished"
exit 0
