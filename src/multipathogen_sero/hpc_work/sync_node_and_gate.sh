
rsync -avzhPue "ssh -i ~/.ssh/gate" /homes/ayan/sero/multipathogen-sero/outputs/from_hpc/ ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/outputs/from_hpc/

rsync -avzhPue "ssh -i ~/.ssh/gate" ayan@gate.stats.ox.ac.uk:~/sero/multipathogen-sero/src/multipathogen_sero/ /homes/ayan/sero/multipathogen-sero/src/multipathogen_sero/