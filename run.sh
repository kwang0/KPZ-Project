#!/bin/bash -l
#SBATCH --account=lr_oppie
#SBATCH --partition=lr6
#SBATCH --qos=condo_oppie
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=kwang98@berkeley.edu
#SBATCH --mem=128GB
#SBATCH --exclusive

#SBATCH --cpus-per-task=40

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS=40
python -u $1 -l $2 -c $3 -b $4 -t $5 > logs/L${2}_chi${3}_beta${4}_dt${5}.txt

exit 0
