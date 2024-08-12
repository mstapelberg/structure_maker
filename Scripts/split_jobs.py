import os
import math
import shutil

def split_jobs_into_batches(job_dir, num_batches, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")  # Debugging line

    # Check if the job directory exists and list its contents
    if not os.path.exists(job_dir):
        print(f"Job directory does not exist: {job_dir}")
        return

    print(f"Contents of job directory ({job_dir}): {os.listdir(job_dir)}")  # Debugging line

    # Get the list of job subdirectories
    job_dirs = [d for d in os.listdir(job_dir) if os.path.isdir(os.path.join(job_dir, d))]
    print(f"Job directories: {job_dirs}")  # Debugging line

    # Calculate the number of jobs per batch
    num_jobs = len(job_dirs)
    jobs_per_batch = math.ceil(num_jobs / num_batches)
    print(f"Number of jobs: {num_jobs}, Jobs per batch: {jobs_per_batch}")  # Debugging line

    # Create batch directories and distribute job subdirectories
    for i in range(num_batches):
        batch_dir = os.path.join(output_dir, f'batch_{i+1}')
        os.makedirs(batch_dir, exist_ok=True)
        print(f"Created batch directory: {batch_dir}")  # Debugging line

        batch_dirs = job_dirs[i * jobs_per_batch:(i + 1) * jobs_per_batch]
        print(f"Batch {i+1} directories: {batch_dirs}")  # Debugging line
        for job_subdir in batch_dirs:
            src_path = os.path.join(job_dir, job_subdir)
            dest_path = os.path.join(batch_dir, job_subdir)
            print(f"Copying {src_path} to {dest_path}")  # Debugging line
            if os.path.exists(src_path):
                shutil.copytree(src_path, dest_path)
                print(f"Copied {job_subdir} to {batch_dir}")  # Debugging line
            else:
                print(f"Source path does not exist: {src_path}")  # Debugging line

        # Generate SLURM script for the batch
        generate_slurm_script(batch_dir, i+1)

def generate_slurm_script(batch_dir, batch_num):
    slurm_script = f"""#!/bin/bash
#
#SBATCH --job-name=vac_job_{batch_num}
#SBATCH --output=vac_job_{batch_num}-%j.out
#SBATCH --error=vac_job_{batch_num}-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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

srun python3 /home/myless/Packages/structure_maker/Scripts/job_script.py {batch_dir}

exit
"""
    slurm_script_path = os.path.join(batch_dir, f'slurm_script_{batch_num}.sh')
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script)

def main():
    job_dir = '/home/myless/Packages/structure_maker/Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125/Vacancies_Fixed'
    output_dir = '/home/myless/Packages/structure_maker/Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125/Pre_NEB_Fixed'
    num_batches = 6
    split_jobs_into_batches(job_dir, num_batches, output_dir)

if __name__ == '__main__':
    main()
