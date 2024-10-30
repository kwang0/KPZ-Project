#!/bin/bash
#SBATCH -A m3341
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o ./logs_slurm/slurm-%j.out

julia -t 8 --heap-size-hint=400G $1 $2 $3 $4 $5 $6 $7 > logs_jl/tebd_su3_dw_L${2}_chi${3}_beta${4}_dt${5}_U${6}_mu${7}_conserve_threesite_conj.txt

exit 0
