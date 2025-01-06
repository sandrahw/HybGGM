#!/bin/bash
#SBATCH --job-name="hyperparam_random_samples_fulltile_U2"
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_random_samples_fulltile_U2.out
#SBATCH --error=hyperparam_random_samples_fulltile_U2.err

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
echo $PYTORCH_CUDA_ALLOC_CONF 
echo "UNet2 with 1 10 0.0001 4 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas_full_tile.py 1 10 0.0001 4 180 UNet2
echo "UNet2 with 1 10 0.0001 4 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
