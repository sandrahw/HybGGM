#open file connection to generate hyperparam.sh
filename = 'validation_large600_rerun.slurm'
outputname = 'validation_large600_rerun.out'
errorname = 'validation_large600_rerun.err'
jobname = 'validation_large600_rerun'

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
        for j in [5, 10, 20, 50][:]:
            for k in [0.0001, 0.001][:]:
                for area in ['small', 'large', 'half', 'full'][1:2]:
                    # f.write('echo "UNet2 with 0.4 0.3 %s %s %s %s start at $(date)" \n' % (i, k, j, area))
                    # f.write('python src/model/run_validation.py 0.4 0.3 %s %s %s UNet2 %s\n' % (i, k, j, area))
                    # f.write('echo "UNet2 with 0.4 0.3 %s %s %s %s done at $(date)" \n' % (i, k, j, area))
                    f.write('echo "UNet6 with 0.4 0.3 %s %s %s %s start at $(date)" \n' % (i, k, j, area))
                    f.write('python src/model/run_validation.py 0.4 0.3 %s %s %s UNet6 %s \n' % (i, k, j, area))
                    f.write('echo "UNet6 with 0.4 0.3 %s %s %s %s done at $(date)" \n' % (i, k, j, area))
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




