#!/bin/bash
#
#SBATCH --job-name=create_neb_jobs
#SBATCH --output=bench-%j.out
#SBATCH --error=bench-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh

# Activate the 'allegro' environment
conda activate mace-11.7

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=10
export NEQUP_NUM_TASKS=10

# Run the benchmark script
#nequip-benchmark stress_test.yaml
#nequip-train stress_test_big_model.yaml
srun python3 make_NEB_jobs.py


exit

