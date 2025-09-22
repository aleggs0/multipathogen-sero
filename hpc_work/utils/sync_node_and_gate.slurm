#!/bin/bash
#SBATCH --job-name=sync_node_and_gate
#SBATCH --clusters=srf_cpu_01
#SBATCH --nodelist=swan03.cpu.stats.ox.ac.uk
#SBATCH --partition=standard-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:20:00
##SBATCH --export=ALL
#SBATCH --output=/homes/ayan/sero/multipathogen-sero/outputs/from_hpc/log/%x_%A.out
#SBATCH --error=/homes/ayan/sero/multipathogen-sero/outputs/from_hpc/log/%x_%A.err


rsync -avzhPue "ssh -i ~/.ssh/gate" /homes/ayan/sero/multipathogen-sero/outputs/from_hpc/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/outputs/from_hpc/
rsync -avzhPue "ssh -i ~/.ssh/gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/hpc_work/ /homes/ayan/sero/multipathogen-sero/hpc_work/
## rsync -avzhPue "ssh -i ~/.ssh/gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/src/multipathogen_sero/ /homes/ayan/sero/multipathogen-sero/src/multipathogen_sero/

## in case of stan file changes, recompile the models
module load conda
conda init bash
source ~/.bashrc
conda activate multipathogen-sero
python ~/sero/multipathogen-sero/src/multipathogen_sero/models/compile_stan.py
