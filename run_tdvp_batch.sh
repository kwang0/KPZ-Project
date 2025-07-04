#!/bin/bash
#SBATCH -A m4863_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --licenses=scratch
#SBATCH -o ./logs_slurm/slurm-%j.out

export SLURM_CPU_BIND="cores"

for i in $(seq 0.0 0.1 0.5)
do
    srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G julia heisenberg_ladder_tdvp_coarsegrained_dw_gpu.jl 128 512 0.0 0.1 $i 0.0 0.0 0.001 > logs_jl/tdvp_coarsegrained_dw_gpu_L128_chi512_beta0.0_dt0.1_Jprime${i}_U0.0_Uprime0.0_mu0.001.txt &
done

# for i in $(seq 0.4 0.4 2.0)
# do
#     srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G julia heisenberg_ladder_tdvp_coarsegrained_dw_gpu.jl 128 512 0.0 0.1 0.0 $i 0.0 0.001 > logs_jl/tdvp_coarsegrained_dw_gpu_L128_chi512_beta0.0_dt0.1_Jprime0.0_U${i}_Uprime0.0_mu0.001.txt &
# done

# for i in $(seq 0.4 0.4 2.0)
# do
#     srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G julia heisenberg_ladder_tdvp_coarsegrained_dw_gpu.jl 128 512 0.0 0.1 0.0 0.0 $i 0.001 > logs_jl/tdvp_coarsegrained_dw_gpu_L128_chi512_beta0.0_dt0.1_Jprime0.0_U0.0_Uprime${i}_mu0.001.txt &
# done

# for i in $(seq 0.0 0.1 0.5)
# do
#     srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G julia heisenberg_ladder_tdvp_coarsegrained_dw_gpu.jl 64 600 0.0 0.1 $i 4.0 0.0 0.001 > logs_jl/tdvp_coarsegrained_dw_gpu_L64_chi600_beta0.0_dt0.1_Jprime${i}_U4.0_Uprime0.0_mu0.001.txt &
# done

# for i in $(seq 4.4 0.4 6.0)
# do
#     srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G julia heisenberg_ladder_tdvp_coarsegrained_dw_gpu.jl 64 600 0.0 0.1 0.0 $i 0.0 0.001 > logs_jl/tdvp_coarsegrained_dw_gpu_L64_chi600_beta0.0_dt0.1_Jprime0.0_U${i}_Uprime0.0_mu0.001.txt &
# done

# for i in $(seq 0.4 0.4 2.0)
# do
#     srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G julia heisenberg_ladder_tdvp_coarsegrained_dw_gpu.jl 64 600 0.0 0.1 0.0 4.0 $i 0.001 > logs_jl/tdvp_coarsegrained_dw_gpu_L64_chi600_beta0.0_dt0.1_Jprime0.0_U4.0_Uprime${i}_mu0.001.txt &
# done

wait
