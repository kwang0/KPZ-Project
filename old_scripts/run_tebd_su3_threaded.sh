#!/bin/bash
#SBATCH -A m3341
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00


julia -t $8 --heap-size-hint=400G $1 $2 $3 $4 $5 $6 $7 $8 > logs_jl/tebd_su3_dw_L${2}_chi${3}_beta${4}_dt${5}_U${6}_mu${7}_threaded${8}.txt

exit 0
