#open file connection to generate hyperparam.sh
filename = 'sampleprep_initialparams_tile48_180_fullp1.slurm'
outputname = 'sampleprep_initialparams_tile48_180_fullp1.out'
errorname = 'sampleprep_initialparams_tile48_180_fullp1.err'
jobname = 'sampleprep_initialparams_tile48_180_fullp1'

with open('%s'%filename, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name="%s"\n'%jobname)
    f.write('#SBATCH --ntasks 16\n')
    f.write('#SBATCH --time 72:00:00 \n')
    # f.write('#SBATCH --cpus-per-task=8\n')
    f.write('#SBATCH --output=%s\n'%outputname) # BRAM example: SBATCH --output ./log/hybgmm_test.txt
    f.write('#SBATCH --error=%s\n'%errorname)# BRAM example: SBATCH --error ./log/hybgmm_test.txt
    f.write('#SBATCH --partition defq\n')
    f.write('\n')
    f.write('echo "SBATCH job"\n')
    f.write('echo "Started $(date \'+%d/%m/%Y %H:%M:%S\')"\n')
    # f.write('echo "Working directory $(pwd)"\n')
    f.write('\n')
    f.write('unset PYTHONHOME\n')
    f.write('cd /eejit/home/hausw001/HybGGM/hybGGM_test\n') #ejit path to hybGGM_test
    f.write('echo "Working directory $(pwd)"\n')
    f.write('conda activate env_hybGMM \n')
    f.write('echo "start sample prep target_tile48_180_full" \n')
    f.write('python src/data_prep/preprocess_data_initialparams_fullrun_p1.py \n')
    f.write('echo "finished sample prep target_tile48_180_full" \n')
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




