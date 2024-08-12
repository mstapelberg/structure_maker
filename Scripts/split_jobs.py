import os
import math
import shutil

def split_jobs_into_batches(job_dir, num_batches):
    # Get the list of job files
    job_files = [f for f in os.listdir(job_dir) if os.path.isfile(os.path.join(job_dir, f))]
    
    # Calculate the number of jobs per batch
    num_jobs = len(job_files)
    jobs_per_batch = math.ceil(num_jobs / num_batches)
    
    # Create batch directories and distribute job files
    for i in range(num_batches):
        batch_dir = os.path.join(job_dir, f'batch_{i+1}')
        os.makedirs(batch_dir, exist_ok=True)
        
        batch_files = job_files[i * jobs_per_batch:(i + 1) * jobs_per_batch]
        for job_file in batch_files:
            shutil.move(os.path.join(job_dir, job_file), os.path.join(batch_dir, job_file))
        
        # Generate SLURM script for the batch
        generate_slurm_script(batch_dir, i+1)

def generate_slurm_script(batch_dir, batch_num):
    slurm_script = f"""#!/bin/bash
#
#SBATCH --job-name=vac_job_{batch_num}
#SBATCH --output=vac_job_{batch_num}-%j.out
#SBATCH --error=vac_job_{batch_num}-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh

# Activate the environment
conda activate chgnet-11.7

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=10
export NEQUP_NUM_TASKS=10

srun python3 job_script.py {batch_dir}

exit
"""
    slurm_script_path = os.path.join(batch_dir, f'slurm_script_{batch_num}.sh')
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script)

def main():
    job_dir = '../Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125/Vacancies_Fixed'
    num_batches = 12
    split_jobs_into_batches(job_dir, num_batches)

if __name__ == '__main__':
    main()