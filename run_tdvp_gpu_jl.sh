#!/bin/bash -l
#SBATCH --account=pc_gpumoore
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=kwang98@berkeley.edu
#SBATCH --mem=187GB
#SBATCH --exclusive

#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=16
julia $1 $2 $3 $4 $5 $6 > logs_jl/tdvp_gpu_L${2}_chi${3}_beta${4}_dt${5}_Jprime${6}_multithreaded.txt

exit 0
