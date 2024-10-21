#open file connection to generate hyperparam.sh
with open('hyperparam.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name="hybggm_test"\n')
    f.write('#SBATCH --ntasks 1"\n')
    f.write('#SBATCH --time 01:00:00 \n')
    # f.write('#SBATCH --cpus-per-task=8\n')
    f.write('#SBATCH --output=hyperparam.out\n') # BRAM example: SBATCH --output ./log/hybgmm_test.txt
    f.write('#SBATCH --error=hyperparam.err\n')# BRAM example: SBATCH --error ./log/hybgmm_test.txt
    f.write('#SBATCH --partition defq\n')
    f.write('\n')
    f.write('echo "SBATCH job"\n')
    f.write('echo "Started $(date \'+%d/%m/%Y %H:%M:%S\')"\n')
    f.write('echo "Working directory $(pwd)"\n')
    f.write('\n')
    f.write('module load python/3.7\n')
    f.write('cd hybGGM_test\n') #ejit path to hybGGM_test
    for i in [2, 4, 6, 8, 10]:
        f.write('python src/run_hyperparam_testing.py 0.4 0.3 %s 0.001 1 UNet2\n' % i)
        f.write('python src/run_hyperparam_testing.py 0.4 0.3 %s 0.001 1 ConvExample\n' % i)
    f.write('cp -r $SLURM_TMPDIR/hybGGM_test $HOME/\n')
    f.write('\n')
    f.write('exit 0\n')


