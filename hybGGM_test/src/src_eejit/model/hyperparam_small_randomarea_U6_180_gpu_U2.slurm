#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_U6_180_gpu_U2"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00 
#SBATCH --mem=32GB
#SBATCH --output=hyperparam_small_randomarea_U6_180_gpu_U2.out
#SBATCH --error=hyperparam_small_randomarea_U6_180_gpu_U2.err

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
echo $PYTORCH_CUDA_ALLOC_CONF 
echo "UNet2 with 1 50 0.0001 2 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 2 180 UNet2
echo "UNet2 with 1 50 0.0001 2 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 4 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 4 180 UNet2
echo "UNet2 with 1 50 0.0001 4 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 8 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 8 180 UNet2
echo "UNet2 with 1 50 0.0001 8 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 16 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 16 180 UNet2
echo "UNet2 with 1 50 0.0001 16 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 32 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 32 180 UNet2
echo "UNet2 with 1 50 0.0001 32 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 64 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 64 180 UNet2
echo "UNet2 with 1 50 0.0001 64 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 128 180 UNet2
echo "UNet2 with 1 50 0.0001 128 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 256 180 UNet2
echo "UNet2 with 1 50 0.0001 256 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
