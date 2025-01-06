#open file connection to generate hyperparam.sh
filename = 'hyperparam_small360_rerun.slurm'
outputname = 'hyperparam_small360_rerun.out'
errorname = 'hyperparam_small360_rerun.err'
jobname = 'hyperparam_small360_rerun'

with open('%s'%filename, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name="%s"\n'%jobname)
    f.write('#SBATCH --ntasks 16\n')
    f.write('#SBATCH --time 48:00:00 \n')
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
    for i in [10, 50, 100, 150, 200][:]:
        for j in [16, 20, 32, 64, 128, 256][:]:
            for k in [0.00001, 0.0001, 0.001, 0.01][:]:
                for area in ['small', 'large', 'half', 'full'][:1]:
                    # f.write('echo "UNet2 with 0.4 0.3 %s %s %s %s start at $(date)" \n' % (i, k, j, area))
                    # f.write('python src/model/run_hyperparam_testing.py 0.4 0.3 %s %s %s UNet2 %s\n' % (i, k, j, area))
                    # f.write('echo "UNet2 with 0.4 0.3 %s %s %s %s done at $(date)" \n' % (i, k, j, area))
                    f.write('echo "UNet6 with 0.4 0.3 %s %s %s %s start at $(date)" \n' % (i, k, j, area))
                    f.write('python src/model/run_hyperparam_testing.py 0.4 0.3 %s %s %s UNet6 %s \n' % (i, k, j, area))
                    f.write('echo "UNet6 with 0.4 0.3 %s %s %s %s done at $(date)" \n' % (i, k, j, area))
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




