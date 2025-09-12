#!/bin/bash
#SBATCH --job-name=copy_outputs_to_gate
#SBATCH --clusters=srf_cpu_01
#SBATCH --nodelist=swan02.cpu.stats.ox.ac.uk
#SBATCH --partition=standard-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:20:00
##SBATCH --export=ALL

rsync -avzhPue "ssh -i ~/.ssh/gate" /homes/ayan/log/ ayan@gate.stats.ox.ac.uk:~/log/
rsync -avzhPue "ssh -i ~/.ssh/gate" /homes/ayan/sero/multipathogen-sero/model_fits/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/model-fits/