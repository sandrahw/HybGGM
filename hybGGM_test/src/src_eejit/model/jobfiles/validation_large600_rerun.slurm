#!/bin/bash
#SBATCH --job-name="validation_large600_rerun"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=validation_large600_rerun.out
#SBATCH --error=validation_large600_rerun.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 5 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 5 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 5 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 5 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 5 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 5 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 10 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 10 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 10 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 10 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 10 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 10 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 20 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 20 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 20 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 20 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 20 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 20 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 50 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 50 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 50 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 50 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 50 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 50 large done at $(date)" 

echo "SBATCH job finished"
exit 0
