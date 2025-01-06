#!/bin/bash
#SBATCH --job-name="random_samples_fulltile_U2_4_8_32"
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00 
#SBATCH --output=random_samples_fulltile_U2_4_8_32.out
#SBATCH --error=random_samples_fulltile_U2_4_8_32.err

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
echo $PYTORCH_CUDA_ALLOC_CONF 
echo "UNet2 with 1 50 0.0001 4 180 start at $(date)" 
python src/model/run_random_areas_full_tile.py 1 50 0.0001 4 180 UNet2
echo "UNet2 with 1 50 0.0001 4 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 8 180 start at $(date)" 
python src/model/run_random_areas_full_tile.py 1 50 0.0001 8 180 UNet2
echo "UNet2 with 1 50 0.0001 8 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 32 180 start at $(date)" 
python src/model/run_random_areas_full_tile.py 1 50 0.0001 32 180 UNet2
echo "UNet2 with 1 50 0.0001 32 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
