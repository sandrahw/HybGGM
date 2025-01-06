#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_360_gpu_U6_minibadge"
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_small_randomarea_360_gpu_U6_minibadge.out
#SBATCH --error=hyperparam_small_randomarea_360_gpu_U6_minibadge.err

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
echo $PYTORCH_CUDA_ALLOC_CONF 
echo "UNet6 with 1 10 0.0001 4 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_minibadge.py 1 10 0.0001 4 360 UNet6
echo "UNet6 with 1 10 0.0001 4 360 done at $(date)" 
echo "UNet6 with 1 10 0.0001 8 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_minibadge.py 1 10 0.0001 8 360 UNet6
echo "UNet6 with 1 10 0.0001 8 360 done at $(date)" 
echo "UNet6 with 1 10 0.0001 32 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_minibadge.py 1 10 0.0001 32 360 UNet6
echo "UNet6 with 1 10 0.0001 32 360 done at $(date)" 
echo "UNet6 with 1 10 0.0001 64 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_minibadge.py 1 10 0.0001 64 360 UNet6
echo "UNet6 with 1 10 0.0001 64 360 done at $(date)" 
echo "UNet6 with 1 10 0.0001 128 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_minibadge.py 1 10 0.0001 128 360 UNet6
echo "UNet6 with 1 10 0.0001 128 360 done at $(date)" 
echo "UNet6 with 1 10 0.0001 256 360 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_minibadge.py 1 10 0.0001 256 360 UNet6
echo "UNet6 with 1 10 0.0001 256 360 done at $(date)" 

echo "SBATCH job finished"
exit 0
