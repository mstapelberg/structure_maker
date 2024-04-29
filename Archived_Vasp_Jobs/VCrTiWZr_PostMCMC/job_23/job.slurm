#!/bin/bash
#SBATCH --job-name=job_23
#SBATCH --output=std-out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=v100s
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS=10
cd $SLURM_SUBMIT_DIR
/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/mpi/bin/mpirun /home/myless/VASP/vasp.6.3.2/bin/vasp_std
exit
