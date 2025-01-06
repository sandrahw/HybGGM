#!/bin/bash
#SBATCH --job-name="validation_large720_rerun"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=validation_large720_rerun.out
#SBATCH --error=validation_large720_rerun.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 50 0.0001 16 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 16 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 16 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 16 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 16 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 16 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 32 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 32 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 32 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 32 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 32 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 32 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 64 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 64 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 64 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 64 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 64 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 64 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 128 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 128 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 128 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 128 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 128 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 128 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 256 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 256 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.0001 256 large done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 256 large start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 256 UNet6 large 
echo "UNet6 with 0.4 0.3 50 0.001 256 large done at $(date)" 

echo "SBATCH job finished"
exit 0
