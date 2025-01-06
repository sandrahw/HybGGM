#!/bin/bash
#SBATCH --job-name="sampleprep_monthlyinput_tile48_180_fullp4"
#SBATCH --ntasks 16
#SBATCH --time 72:00:00 
#SBATCH --output=sampleprep_monthlyinput_tile48_180_fullp4.out
#SBATCH --error=sampleprep_monthlyinput_tile48_180_fullp4.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "start sample prep target_tile48_180_full" 
python src/data_prep/preprocess_data_monthlyinput_fullrun_p4.py 
echo "finished sample prep target_tile48_180_full" 

echo "SBATCH job finished"
exit 0
