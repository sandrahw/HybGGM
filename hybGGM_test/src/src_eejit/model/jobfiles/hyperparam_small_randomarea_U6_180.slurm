#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_U6_180"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_small_randomarea_U6_180.out
#SBATCH --error=hyperparam_small_randomarea_U6_180.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.1 10 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 10 1e-05 32 180 UNet6
echo "UNet6 with 0.1 10 1e-05 32 180 done at $(date)" 
echo "UNet6 with 0.5 10 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 10 1e-05 32 180 UNet6
echo "UNet6 with 0.5 10 1e-05 32 180 done at $(date)" 
echo "UNet6 with 1 10 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 1e-05 32 180 UNet6
echo "UNet6 with 1 10 1e-05 32 180 done at $(date)" 
echo "UNet6 with 0.1 10 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 10 0.0001 32 180 UNet6
echo "UNet6 with 0.1 10 0.0001 32 180 done at $(date)" 
echo "UNet6 with 0.5 10 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 10 0.0001 32 180 UNet6
echo "UNet6 with 0.5 10 0.0001 32 180 done at $(date)" 
echo "UNet6 with 1 10 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0001 32 180 UNet6
echo "UNet6 with 1 10 0.0001 32 180 done at $(date)" 
echo "UNet6 with 0.1 10 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 10 1e-05 64 180 UNet6
echo "UNet6 with 0.1 10 1e-05 64 180 done at $(date)" 
echo "UNet6 with 0.5 10 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 10 1e-05 64 180 UNet6
echo "UNet6 with 0.5 10 1e-05 64 180 done at $(date)" 
echo "UNet6 with 1 10 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 1e-05 64 180 UNet6
echo "UNet6 with 1 10 1e-05 64 180 done at $(date)" 
echo "UNet6 with 0.1 10 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 10 0.0001 64 180 UNet6
echo "UNet6 with 0.1 10 0.0001 64 180 done at $(date)" 
echo "UNet6 with 0.5 10 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 10 0.0001 64 180 UNet6
echo "UNet6 with 0.5 10 0.0001 64 180 done at $(date)" 
echo "UNet6 with 1 10 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0001 64 180 UNet6
echo "UNet6 with 1 10 0.0001 64 180 done at $(date)" 
echo "UNet6 with 0.1 10 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 10 1e-05 128 180 UNet6
echo "UNet6 with 0.1 10 1e-05 128 180 done at $(date)" 
echo "UNet6 with 0.5 10 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 10 1e-05 128 180 UNet6
echo "UNet6 with 0.5 10 1e-05 128 180 done at $(date)" 
echo "UNet6 with 1 10 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 1e-05 128 180 UNet6
echo "UNet6 with 1 10 1e-05 128 180 done at $(date)" 
echo "UNet6 with 0.1 10 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 10 0.0001 128 180 UNet6
echo "UNet6 with 0.1 10 0.0001 128 180 done at $(date)" 
echo "UNet6 with 0.5 10 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 10 0.0001 128 180 UNet6
echo "UNet6 with 0.5 10 0.0001 128 180 done at $(date)" 
echo "UNet6 with 1 10 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0001 128 180 UNet6
echo "UNet6 with 1 10 0.0001 128 180 done at $(date)" 
echo "UNet6 with 0.1 50 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 50 1e-05 32 180 UNet6
echo "UNet6 with 0.1 50 1e-05 32 180 done at $(date)" 
echo "UNet6 with 0.5 50 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 50 1e-05 32 180 UNet6
echo "UNet6 with 0.5 50 1e-05 32 180 done at $(date)" 
echo "UNet6 with 1 50 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 1e-05 32 180 UNet6
echo "UNet6 with 1 50 1e-05 32 180 done at $(date)" 
echo "UNet6 with 0.1 50 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 50 0.0001 32 180 UNet6
echo "UNet6 with 0.1 50 0.0001 32 180 done at $(date)" 
echo "UNet6 with 0.5 50 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 50 0.0001 32 180 UNet6
echo "UNet6 with 0.5 50 0.0001 32 180 done at $(date)" 
echo "UNet6 with 1 50 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 32 180 UNet6
echo "UNet6 with 1 50 0.0001 32 180 done at $(date)" 
echo "UNet6 with 0.1 50 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 50 1e-05 64 180 UNet6
echo "UNet6 with 0.1 50 1e-05 64 180 done at $(date)" 
echo "UNet6 with 0.5 50 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 50 1e-05 64 180 UNet6
echo "UNet6 with 0.5 50 1e-05 64 180 done at $(date)" 
echo "UNet6 with 1 50 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 1e-05 64 180 UNet6
echo "UNet6 with 1 50 1e-05 64 180 done at $(date)" 
echo "UNet6 with 0.1 50 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 50 0.0001 64 180 UNet6
echo "UNet6 with 0.1 50 0.0001 64 180 done at $(date)" 
echo "UNet6 with 0.5 50 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 50 0.0001 64 180 UNet6
echo "UNet6 with 0.5 50 0.0001 64 180 done at $(date)" 
echo "UNet6 with 1 50 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 64 180 UNet6
echo "UNet6 with 1 50 0.0001 64 180 done at $(date)" 
echo "UNet6 with 0.1 50 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 50 1e-05 128 180 UNet6
echo "UNet6 with 0.1 50 1e-05 128 180 done at $(date)" 
echo "UNet6 with 0.5 50 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 50 1e-05 128 180 UNet6
echo "UNet6 with 0.5 50 1e-05 128 180 done at $(date)" 
echo "UNet6 with 1 50 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 1e-05 128 180 UNet6
echo "UNet6 with 1 50 1e-05 128 180 done at $(date)" 
echo "UNet6 with 0.1 50 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 50 0.0001 128 180 UNet6
echo "UNet6 with 0.1 50 0.0001 128 180 done at $(date)" 
echo "UNet6 with 0.5 50 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 50 0.0001 128 180 UNet6
echo "UNet6 with 0.5 50 0.0001 128 180 done at $(date)" 
echo "UNet6 with 1 50 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 128 180 UNet6
echo "UNet6 with 1 50 0.0001 128 180 done at $(date)" 
echo "UNet6 with 0.1 100 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 100 1e-05 32 180 UNet6
echo "UNet6 with 0.1 100 1e-05 32 180 done at $(date)" 
echo "UNet6 with 0.5 100 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 100 1e-05 32 180 UNet6
echo "UNet6 with 0.5 100 1e-05 32 180 done at $(date)" 
echo "UNet6 with 1 100 1e-05 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 1e-05 32 180 UNet6
echo "UNet6 with 1 100 1e-05 32 180 done at $(date)" 
echo "UNet6 with 0.1 100 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 100 0.0001 32 180 UNet6
echo "UNet6 with 0.1 100 0.0001 32 180 done at $(date)" 
echo "UNet6 with 0.5 100 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 100 0.0001 32 180 UNet6
echo "UNet6 with 0.5 100 0.0001 32 180 done at $(date)" 
echo "UNet6 with 1 100 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0001 32 180 UNet6
echo "UNet6 with 1 100 0.0001 32 180 done at $(date)" 
echo "UNet6 with 0.1 100 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 100 1e-05 64 180 UNet6
echo "UNet6 with 0.1 100 1e-05 64 180 done at $(date)" 
echo "UNet6 with 0.5 100 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 100 1e-05 64 180 UNet6
echo "UNet6 with 0.5 100 1e-05 64 180 done at $(date)" 
echo "UNet6 with 1 100 1e-05 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 1e-05 64 180 UNet6
echo "UNet6 with 1 100 1e-05 64 180 done at $(date)" 
echo "UNet6 with 0.1 100 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 100 0.0001 64 180 UNet6
echo "UNet6 with 0.1 100 0.0001 64 180 done at $(date)" 
echo "UNet6 with 0.5 100 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 100 0.0001 64 180 UNet6
echo "UNet6 with 0.5 100 0.0001 64 180 done at $(date)" 
echo "UNet6 with 1 100 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0001 64 180 UNet6
echo "UNet6 with 1 100 0.0001 64 180 done at $(date)" 
echo "UNet6 with 0.1 100 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 100 1e-05 128 180 UNet6
echo "UNet6 with 0.1 100 1e-05 128 180 done at $(date)" 
echo "UNet6 with 0.5 100 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 100 1e-05 128 180 UNet6
echo "UNet6 with 0.5 100 1e-05 128 180 done at $(date)" 
echo "UNet6 with 1 100 1e-05 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 1e-05 128 180 UNet6
echo "UNet6 with 1 100 1e-05 128 180 done at $(date)" 
echo "UNet6 with 0.1 100 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.1 100 0.0001 128 180 UNet6
echo "UNet6 with 0.1 100 0.0001 128 180 done at $(date)" 
echo "UNet6 with 0.5 100 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 0.5 100 0.0001 128 180 UNet6
echo "UNet6 with 0.5 100 0.0001 128 180 done at $(date)" 
echo "UNet6 with 1 100 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0001 128 180 UNet6
echo "UNet6 with 1 100 0.0001 128 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
