#!/bin/bash
#SBATCH --job-name="sampleprep_monthlyinput_tile48_180"
#SBATCH --ntasks 16
#SBATCH --time 48:00:00 
#SBATCH --output=sampleprep_monthlyinput_tile48_180.out
#SBATCH --error=sampleprep_monthlyinput_tile48_180.err
#SBATCH --partition defq

echo "SBATCH job"
echo "Started $(date '+%d/%m/%Y %H:%M:%S')"

unset PYTHONHOME
cd /eejit/home/hausw001/HybGGM/hybGGM_test
echo "Working directory $(pwd)"
conda activate env_hybGMM 
echo "start sample prep monthlyinput til48 180x180" 
python src/data_prep/preprocess_data_monthlyinput.py 
echo "finished sample prep monthlyinput til48 180x180" 

echo "SBATCH job finished"
exit 0
