#open file connection to generate hyperparam.sh
filename = 'hyperparam_small_randomarea_U6_180_cpu_128_200.slurm'
outputname = 'hyperparam_small_randomarea_U6_180_cpu_128_200.out'
errorname = 'hyperparam_small_randomarea_U6_180_cpu_128_200.err'
jobname = 'hyperparam_small_randomarea_U6_180_cpu_128_200'

with open('%s'%filename, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name="%s"\n'%jobname)
    f.write('#SBATCH --ntasks 32\n')
    f.write('#SBATCH --time 72:00:00 \n')
    f.write('#SBATCH --nodes=4\n')
    f.write('#SBATCH --ntasks-per-node=48\n')
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
    for i in [200][:]:
        for j in [128][:]:
            for k in [0.0001][:]:
                for area in [180, 360][:1]:
                    for sampsize in [1][:]:
                        f.write('echo "UNet6 with %s %s %s %s %s start at $(date)" \n' % (sampsize, i, k, j, area))
                        f.write('python src/model/run_hyperparam_testing_random_areas_cpu.py %s %s %s %s %s UNet6\n' % (sampsize, i, k, j, area))
                        f.write('echo "UNet6 with %s %s %s %s %s done at $(date)" \n' % (sampsize, i, k, j, area))
    f.write('\n')
    f.write('echo "SBATCH job finished"\n')
    f.write('exit 0\n')




