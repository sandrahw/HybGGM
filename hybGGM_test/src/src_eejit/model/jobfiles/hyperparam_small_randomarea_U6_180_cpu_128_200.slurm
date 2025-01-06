#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_U6_180_cpu_128_200"
#SBATCH --ntasks 32
#SBATCH --time 72:00:00 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --output=hyperparam_small_randomarea_U6_180_cpu_128_200.out
#SBATCH --error=hyperparam_small_randomarea_U6_180_cpu_128_200.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 1 200 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_cpu.py 1 200 0.0001 128 180 UNet6
echo "UNet6 with 1 200 0.0001 128 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
