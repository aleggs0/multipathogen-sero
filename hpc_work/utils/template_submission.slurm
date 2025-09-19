#!/bin/bash
#SBATCH --job-name=frailty
#SBATCH --array=1
#SBATCH --clusters=srf_cpu_01
#SBATCH --nodelist=swan03.cpu.stats.ox.ac.uk
#SBATCH --partition=standard-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=4:00:00
#SBATCH --mail-user=alexander.yan@stats.ox.ac.uk
#SBATCH --mail-type=END
##SBATCH --export=ALL
#SBATCH --output=/homes/ayan/sero/multipathogen_sero/outputs/from_hpc/log/%x_%A_%a.out
#SBATCH --error=/homes/ayan/sero/multipathogen_sero/outputs/from_hpc/log/%x_%A_%a.err

idx=${SLURM_ARRAY_TASK_ID}

echo $SLURM_ARRAY_TASK_ID

cd ~/sero/multipathogen-sero/
module load conda
conda init bash
source ~/.bashrc
conda activate multipathogen-sero
python multipathogen_sero/hpc_work/frailty_known.py

