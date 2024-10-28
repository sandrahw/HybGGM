#open file connection to generate hyperparam.sh
with open('hyperparam.slurm', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name="hybggm_test_extended"\n')
    f.write('#SBATCH --ntasks 16\n')
    f.write('#SBATCH --time 72:00:00 \n')
    # f.write('#SBATCH --cpus-per-task=8\n')
    f.write('#SBATCH --output=hyperparam.out\n') # BRAM example: SBATCH --output ./log/hybgmm_test.txt
    f.write('#SBATCH --error=hyperparam.err\n')# BRAM example: SBATCH --error ./log/hybgmm_test.txt
    f.write('#SBATCH --partition defq\n')
    f.write('\n')
    f.write('echo "SBATCH job"\n')
    f.write('echo "Started $(date \'+%d/%m/%Y %H:%M:%S\')"\n')
    # f.write('echo "Working directory $(pwd)"\n')
    f.write('\n')
    f.write('unset PYTHONHOME\n')
    f.write('cd HybGGM/hybGGM_test\n') #ejit path to hybGGM_test
    f.write('echo "Working directory $(pwd)"\n')
    f.write('conda activate env_hybGMM \n')
    for i in [50, 100, 200]:
        for j in [1, 4, 10]:
            for k in [0.0001, 0.0005, 0.001, 0.005]:
                f.write('echo "UNet2 with 0.4 0.3 %s %s %s start at $(date)" \n' % (i, k, j))
                f.write('python src/run_hyperparam_testing.py 0.4 0.3 %s %s %s UNet2\n' % (i, k, j))
                f.write('echo "UNet2 with 0.4 0.3 %s %s %s done at $(date)" \n' % (i, k, j))
                f.write('echo "UNet6 with 0.4 0.3 %s %s %s start at $(date)" \n' % (i, k, j))
                f.write('python src/run_hyperparam_testing.py 0.4 0.3 %s %s %s UNet6\n' % (i, k, j))
                f.write('echo "UNet6 with 0.4 0.3 %s %s %s done at $(date)" \n' % (i, k, j))
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




