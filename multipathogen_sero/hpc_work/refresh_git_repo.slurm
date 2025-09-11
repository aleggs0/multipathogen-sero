#!/bin/bash
#SBATCH --job-name=my_first_array_job
#SBATCH --array=1-2
#SBATCH --clusters=srf_cpu_01
#SBATCH --nodelist=swan02.cpu.stats.ox.ac.uk
#SBATCH --partition=standard-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:20:00
#SBATCH --mail-user=alexander.yan@stats.ox.ac.uk
#SBATCH --mail-type=END
##SBATCH --export=ALL
#SBATCH --output=/homes/ayan/log/%A_%a.out
#SBATCH --error=/homes/ayan/log/%A_%a.err

idx=${SLURM_ARRAY_TASK_ID}

echo $SLURM_ARRAY_TASK_ID

cd ~/sero/multipathogen-sero/
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github

git pull