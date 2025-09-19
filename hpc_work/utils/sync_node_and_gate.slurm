#!/bin/bash
#SBATCH --job-name=sync_node_and_gate
#SBATCH --clusters=srf_cpu_01
#SBATCH --nodelist=swan02.cpu.stats.ox.ac.uk
#SBATCH --partition=standard-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:20:00
##SBATCH --export=ALL


rsync -avzhPue "ssh -i ~/.ssh/gate" /homes/ayan/sero/multipathogen-sero/outputs/from_hpc/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/outputs/from_hpc/

rsync -avzhPue "ssh -i ~/.ssh/gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/src/multipathogen_sero/ /homes/ayan/sero/multipathogen-sero/src/multipathogen_sero/
## retrieve these by using rsync to copy from gate to local:
# rsync -avzhPue "ssh -i ~/.ssh/ayan_at_gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/outputs/from_hpc/ /home/ayan/sero/multipathogen-sero/outputs/from_hpc/
# rsync -avzhPue "ssh -i ~/.ssh/ayan_at_gate" /homes/ayan/sero/multipathogen-sero/src/multipathogen_sero/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/src/multipathogen_sero/

# in case of stan file changes, recompile the models
module load conda
conda init bash
source ~/.bashrc
conda activate multipathogen-sero
python ~/sero/multipathogen-sero/src/multipathogen_sero/models/compile_stan.py
