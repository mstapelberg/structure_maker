#!/bin/bash
#
#SBATCH --job-name=test_gpu_assignment
#SBATCH --output=test_gpu_assignment-%j.out
#SBATCH --error=test_gpu_assignment-%j.err
#SBATCH --ntasks=1            # Number of tasks per job, set to 1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2          # Request 2 GPUs per job
#SBATCH --constraint=rtx6000
#SBATCH --time=00:30:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh

# Activate the environment
conda activate chgnet-11.7

# Change to the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=10
export NEQUP_NUM_TASKS=10
export CUDA_LAUNCH_BLOCKING=1

# Run your Python script with full path
python3 /home/myless/Packages/structure_maker/Scripts/test_gpu_script.py /home/myless/Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125/Pre_NEB/batch_1

exit
