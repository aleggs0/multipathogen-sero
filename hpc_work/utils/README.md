$ Useful commands

```
$ ssh ayan@gate.stats.ox.ac.uk
$ ssh ayan@slurm-hn02.stats.ox.ac.uk
$ squeue
$ sinfo -M all -N -l  # for nodes
$ sinfo -M all -l  # for partitions
$ sbatch frailty.slurm
$ srun --pty -t 0:30:00 -M srf_cpu_01 --partition=swan02-debug --mem=16G bash -l
```

File transfer between node and gate
```
ssh-keygen -t ed25519 -C "swan02_to_gate"  # call the file gate
ssh-copy-id -i ~/.ssh/gate.pub
rsync -avzhPue "ssh -i ~/.ssh/gate" /homes/ayan/sero/multipathogen-sero/outputs/from_hpc/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/outputs/from_hpc/
rsync -avzhPue "ssh -i ~/.ssh/gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/hpc_work/ /homes/ayan/sero/multipathogen-sero/hpc_work/
# rsync -avzhPue "ssh -i ~/.ssh/gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/src/multipathogen_sero/ /homes/ayan/sero/multipathogen-sero/src/multipathogen_sero/
```

File transfer between local and gate
```
rsync -avzhPue "ssh -i ~/.ssh/ayan_at_gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/outputs/from_hpc/ /home/ayan/sero/multipathogen-sero/outputs/from_hpc/
rsync -avzhPue "ssh -i ~/.ssh/ayan_at_gate" /home/ayan/sero/multipathogen-sero/hpc_work/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/hpc_work/
# rsync -avzhPue "ssh -i ~/.ssh/ayan_at_gate" /home/ayan/sero/multipathogen-sero/src/multipathogen_sero/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/src/multipathogen_sero/
```

Github
```
$ git config --global user.name "Alex Yan"
$ git config --global user.email "alexyan88@hotmail.com"
$ ssh-keygen -t ed25519 -C "swan02_to_github"
$ cat ~/.ssh/github.pub 
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
$ git clone git@github.com:aleggs0/multipathogen-sero.git
$ git pull
```

Conda
```
module load conda
conda init bash
source ~/.bashrc
conda activate multipathogen-sero
python ~/sero/multipathogen-sero/src/multipathogen_sero/models/compile_stan.py
# conda env create -f environment.yml
# conda env update --file environment.yml --prune
```