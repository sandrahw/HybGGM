#!/bin/bash
#SBATCH --job-name="hyperparam_small_randomarea_180_gpu_U2_Test"
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00 
#SBATCH --output=hyperparam_small_randomarea_180_gpu_U2_Test.out
#SBATCH --error=hyperparam_small_randomarea_180_gpu_U2_Test.err

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
echo $PYTORCH_CUDA_ALLOC_CONF 
echo "UNet2 with 1 10 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0001 128 180 UNet2
echo "UNet2 with 1 10 0.0001 128 180 done at $(date)" 
echo "UNet2 with 1 10 0.0005 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0005 128 180 UNet2
echo "UNet2 with 1 10 0.0005 128 180 done at $(date)" 
echo "UNet2 with 1 10 0.001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.001 128 180 UNet2
echo "UNet2 with 1 10 0.001 128 180 done at $(date)" 
echo "UNet2 with 1 10 0.005 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.005 128 180 UNet2
echo "UNet2 with 1 10 0.005 128 180 done at $(date)" 
echo "UNet2 with 1 10 0.0001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0001 256 180 UNet2
echo "UNet2 with 1 10 0.0001 256 180 done at $(date)" 
echo "UNet2 with 1 10 0.0005 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.0005 256 180 UNet2
echo "UNet2 with 1 10 0.0005 256 180 done at $(date)" 
echo "UNet2 with 1 10 0.001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.001 256 180 UNet2
echo "UNet2 with 1 10 0.001 256 180 done at $(date)" 
echo "UNet2 with 1 10 0.005 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 10 0.005 256 180 UNet2
echo "UNet2 with 1 10 0.005 256 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 128 180 UNet2
echo "UNet2 with 1 50 0.0001 128 180 done at $(date)" 
echo "UNet2 with 1 50 0.0005 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0005 128 180 UNet2
echo "UNet2 with 1 50 0.0005 128 180 done at $(date)" 
echo "UNet2 with 1 50 0.001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.001 128 180 UNet2
echo "UNet2 with 1 50 0.001 128 180 done at $(date)" 
echo "UNet2 with 1 50 0.005 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.005 128 180 UNet2
echo "UNet2 with 1 50 0.005 128 180 done at $(date)" 
echo "UNet2 with 1 50 0.0001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0001 256 180 UNet2
echo "UNet2 with 1 50 0.0001 256 180 done at $(date)" 
echo "UNet2 with 1 50 0.0005 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.0005 256 180 UNet2
echo "UNet2 with 1 50 0.0005 256 180 done at $(date)" 
echo "UNet2 with 1 50 0.001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.001 256 180 UNet2
echo "UNet2 with 1 50 0.001 256 180 done at $(date)" 
echo "UNet2 with 1 50 0.005 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 50 0.005 256 180 UNet2
echo "UNet2 with 1 50 0.005 256 180 done at $(date)" 
echo "UNet2 with 1 100 0.0001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0001 128 180 UNet2
echo "UNet2 with 1 100 0.0001 128 180 done at $(date)" 
echo "UNet2 with 1 100 0.0005 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0005 128 180 UNet2
echo "UNet2 with 1 100 0.0005 128 180 done at $(date)" 
echo "UNet2 with 1 100 0.001 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.001 128 180 UNet2
echo "UNet2 with 1 100 0.001 128 180 done at $(date)" 
echo "UNet2 with 1 100 0.005 128 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.005 128 180 UNet2
echo "UNet2 with 1 100 0.005 128 180 done at $(date)" 
echo "UNet2 with 1 100 0.0001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0001 256 180 UNet2
echo "UNet2 with 1 100 0.0001 256 180 done at $(date)" 
echo "UNet2 with 1 100 0.0005 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.0005 256 180 UNet2
echo "UNet2 with 1 100 0.0005 256 180 done at $(date)" 
echo "UNet2 with 1 100 0.001 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.001 256 180 UNet2
echo "UNet2 with 1 100 0.001 256 180 done at $(date)" 
echo "UNet2 with 1 100 0.005 256 180 start at $(date)" 
python src/model/run_hyperparam_testing_random_areas.py 1 100 0.005 256 180 UNet2
echo "UNet2 with 1 100 0.005 256 180 done at $(date)" 

echo "SBATCH job finished"
exit 0
