#!/bin/bash
#SBATCH --job-name="validation_half_rerun"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=validation_half_rerun.out
#SBATCH --error=validation_half_rerun.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 64 half start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 64 UNet6 half 
echo "UNet6 with 0.4 0.3 50 0.0001 64 half done at $(date)" 

echo "SBATCH job finished"
exit 0
