#!/bin/bash
#SBATCH --job-name="hybggm_test"
#SBATCH --ntasks 1"
#SBATCH --time 01:00:00 
#SBATCH --output=hyperparam.out
#SBATCH --error=hyperparam.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"
echo "Working directory $(pwd)"

module load python/3.7
cd hybGGM_test
python src/run_hyperparam_testing.py 0.4 0.3 2 0.001 1 UNet2
python src/run_hyperparam_testing.py 0.4 0.3 2 0.001 1 ConvExample
python src/run_hyperparam_testing.py 0.4 0.3 4 0.001 1 UNet2
python src/run_hyperparam_testing.py 0.4 0.3 4 0.001 1 ConvExample
python src/run_hyperparam_testing.py 0.4 0.3 6 0.001 1 UNet2
python src/run_hyperparam_testing.py 0.4 0.3 6 0.001 1 ConvExample
python src/run_hyperparam_testing.py 0.4 0.3 8 0.001 1 UNet2
python src/run_hyperparam_testing.py 0.4 0.3 8 0.001 1 ConvExample
python src/run_hyperparam_testing.py 0.4 0.3 10 0.001 1 UNet2
python src/run_hyperparam_testing.py 0.4 0.3 10 0.001 1 ConvExample
cp -r $SLURM_TMPDIR/hybGGM_test $HOME/

exit 0
