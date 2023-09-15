#!/bin/bash -l
#SBATCH --account=lr_oppie
#SBATCH --partition=lr6
#SBATCH --qos=condo_oppie
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=kwang98@berkeley.edu
#SBATCH --mem=128GB
#SBATCH --exclusive

#SBATCH --cpus-per-task=40

export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
julia -t 40 $1 $2 $3 $4 $5 $6 > logs_jl/tdvp_L${2}_chi${3}_beta${4}_dt${5}_Jprime${6}_qnconserved_blocksparse.txt

exit 0
