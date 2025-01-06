#!/bin/bash
#SBATCH --job-name="validation_small360_rerun"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=validation_small360_rerun.out
#SBATCH --error=validation_small360_rerun.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "UNet6 with 0.4 0.3 10 1e-05 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 1e-05 16 UNet6 small 
echo "UNet6 with 0.4 0.3 10 1e-05 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.0001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.0001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.0001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.01 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.01 16 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.01 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 1e-05 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 1e-05 20 UNet6 small 
echo "UNet6 with 0.4 0.3 10 1e-05 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.0001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.0001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.0001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.01 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.01 20 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.01 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 1e-05 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 1e-05 32 UNet6 small 
echo "UNet6 with 0.4 0.3 10 1e-05 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.0001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.0001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.0001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.01 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.01 32 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.01 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 1e-05 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 1e-05 64 UNet6 small 
echo "UNet6 with 0.4 0.3 10 1e-05 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.0001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.0001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.0001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.01 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.01 64 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.01 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 1e-05 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 1e-05 128 UNet6 small 
echo "UNet6 with 0.4 0.3 10 1e-05 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.0001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.0001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.0001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.01 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.01 128 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.01 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 1e-05 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 1e-05 256 UNet6 small 
echo "UNet6 with 0.4 0.3 10 1e-05 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.0001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.0001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.0001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 10 0.01 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 10 0.01 256 UNet6 small 
echo "UNet6 with 0.4 0.3 10 0.01 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 1e-05 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 1e-05 16 UNet6 small 
echo "UNet6 with 0.4 0.3 50 1e-05 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.01 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.01 16 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.01 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 1e-05 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 1e-05 20 UNet6 small 
echo "UNet6 with 0.4 0.3 50 1e-05 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.01 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.01 20 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.01 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 1e-05 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 1e-05 32 UNet6 small 
echo "UNet6 with 0.4 0.3 50 1e-05 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.01 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.01 32 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.01 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 1e-05 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 1e-05 64 UNet6 small 
echo "UNet6 with 0.4 0.3 50 1e-05 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.01 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.01 64 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.01 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 1e-05 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 1e-05 128 UNet6 small 
echo "UNet6 with 0.4 0.3 50 1e-05 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.01 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.01 128 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.01 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 1e-05 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 1e-05 256 UNet6 small 
echo "UNet6 with 0.4 0.3 50 1e-05 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.0001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.0001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.0001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 50 0.01 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 50 0.01 256 UNet6 small 
echo "UNet6 with 0.4 0.3 50 0.01 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 1e-05 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 1e-05 16 UNet6 small 
echo "UNet6 with 0.4 0.3 100 1e-05 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.0001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.01 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.01 16 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.01 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 1e-05 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 1e-05 20 UNet6 small 
echo "UNet6 with 0.4 0.3 100 1e-05 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.0001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.01 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.01 20 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.01 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 1e-05 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 1e-05 32 UNet6 small 
echo "UNet6 with 0.4 0.3 100 1e-05 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.0001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.01 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.01 32 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.01 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 1e-05 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 1e-05 64 UNet6 small 
echo "UNet6 with 0.4 0.3 100 1e-05 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.0001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.01 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.01 64 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.01 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 1e-05 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 1e-05 128 UNet6 small 
echo "UNet6 with 0.4 0.3 100 1e-05 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.0001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.01 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.01 128 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.01 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 1e-05 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 1e-05 256 UNet6 small 
echo "UNet6 with 0.4 0.3 100 1e-05 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.0001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.0001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.0001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 100 0.01 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 100 0.01 256 UNet6 small 
echo "UNet6 with 0.4 0.3 100 0.01 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 1e-05 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 1e-05 16 UNet6 small 
echo "UNet6 with 0.4 0.3 150 1e-05 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.0001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.0001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.0001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.01 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.01 16 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.01 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 1e-05 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 1e-05 20 UNet6 small 
echo "UNet6 with 0.4 0.3 150 1e-05 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.0001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.0001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.0001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.01 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.01 20 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.01 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 1e-05 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 1e-05 32 UNet6 small 
echo "UNet6 with 0.4 0.3 150 1e-05 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.0001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.0001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.0001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.01 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.01 32 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.01 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 1e-05 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 1e-05 64 UNet6 small 
echo "UNet6 with 0.4 0.3 150 1e-05 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.0001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.0001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.0001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.01 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.01 64 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.01 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 1e-05 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 1e-05 128 UNet6 small 
echo "UNet6 with 0.4 0.3 150 1e-05 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.0001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.0001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.0001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.01 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.01 128 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.01 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 1e-05 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 1e-05 256 UNet6 small 
echo "UNet6 with 0.4 0.3 150 1e-05 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.0001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.0001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.0001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 150 0.01 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 150 0.01 256 UNet6 small 
echo "UNet6 with 0.4 0.3 150 0.01 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 1e-05 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 1e-05 16 UNet6 small 
echo "UNet6 with 0.4 0.3 200 1e-05 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.0001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.001 16 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.001 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.01 16 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.01 16 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.01 16 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 1e-05 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 1e-05 20 UNet6 small 
echo "UNet6 with 0.4 0.3 200 1e-05 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.0001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.001 20 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.001 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.01 20 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.01 20 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.01 20 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 1e-05 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 1e-05 32 UNet6 small 
echo "UNet6 with 0.4 0.3 200 1e-05 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.0001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.001 32 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.001 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.01 32 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.01 32 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.01 32 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 1e-05 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 1e-05 64 UNet6 small 
echo "UNet6 with 0.4 0.3 200 1e-05 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.0001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.001 64 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.001 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.01 64 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.01 64 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.01 64 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 1e-05 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 1e-05 128 UNet6 small 
echo "UNet6 with 0.4 0.3 200 1e-05 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.0001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.001 128 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.001 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.01 128 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.01 128 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.01 128 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 1e-05 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 1e-05 256 UNet6 small 
echo "UNet6 with 0.4 0.3 200 1e-05 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.0001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.0001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.0001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.001 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.001 256 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.001 256 small done at $(date)" 
echo "UNet6 with 0.4 0.3 200 0.01 256 small start at $(date)" 
python src/model/run_validation.py 0.4 0.3 200 0.01 256 UNet6 small 
echo "UNet6 with 0.4 0.3 200 0.01 256 small done at $(date)" 

echo "SBATCH job finished"
exit 0
