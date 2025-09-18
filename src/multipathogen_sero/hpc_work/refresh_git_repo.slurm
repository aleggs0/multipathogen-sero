#!/bin/bash
#SBATCH --job-name=repo_setup
#SBATCH --clusters=srf_cpu_01
#SBATCH --nodelist=swan02.cpu.stats.ox.ac.uk
#SBATCH --partition=standard-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:10:00
#SBATCH --mail-user=alexander.yan@stats.ox.ac.uk
#SBATCH --mail-type=END
##SBATCH --export=ALL

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
cd ~/sero/multipathogen-sero/
git pull

module load conda
conda init bash
source ~/.bashrc
conda activate multipathogen-sero
python ~/sero/multipathogen-sero/src/multipathogen_sero/hpc_work/compile_stan.py
