#open file connection to generate hyperparam.sh
filename = 'hyperparam_small_randomarea_U6_360_ext.slurm'
outputname = 'hyperparam_small_randomarea_U6_360_ext.out'
errorname = 'hyperparam_small_randomarea_U6_360_ext.err'
jobname = 'hyperparam_small_randomarea_U6_360_ext'

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
    for i in [50][:]:
        for j in [32, 64, 128][:]:
            for k in [0.00001][:]:
                for area in [180, 360][1:2]:
                    for sampsize in [0.1, 0.5, 1][2:3]:
                        f.write('echo "UNet6 with %s %s %s %s %s start at $(date)" \n' % (sampsize, i, k, j, area))
                        f.write('python src/model/run_hyperparam_testing_random_areas.py %s %s %s %s %s UNet6\n' % (sampsize, i, k, j, area))
                        f.write('echo "UNet6 with %s %s %s %s %s done at $(date)" \n' % (sampsize, i, k, j, area))
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




