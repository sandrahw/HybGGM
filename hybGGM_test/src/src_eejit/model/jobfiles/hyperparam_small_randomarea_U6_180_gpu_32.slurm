#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_U6_180_gpu_32"
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00 
#SBATCH --output=hyperparam_small_randomarea_U6_180_gpu_32.out
#SBATCH --error=hyperparam_small_randomarea_U6_180_gpu_32.err

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
echo $PYTORCH_CUDA_ALLOC_CONF 
echo "UNet6 with 1 50 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 32 180 UNet6
echo "UNet6 with 1 50 0.0001 32 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
