#!/bin/bash
#SBATCH --job-name="hybgmm_limited_input"
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00 
#SBATCH --output=hybgmm_limited_input.out
#SBATCH --error=hybgmm_limited_input.err


echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
python src/model/hybgmm_limited_input.py

echo "SBATCH job finished"
exit 0
