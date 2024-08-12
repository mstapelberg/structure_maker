import os
import torch
from chgnet.model.dynamics import CHGNetCalculator, CHGNet

def main(base_directory):
    # Get the number from base_directory, should be like dir_X
    num = base_directory.split('_')[-1]
    
    # Construct full paths
    base_path = os.path.abspath(base_directory)
    job_path = os.path.join(os.path.abspath("/home/myless/Packages/structure_maker/Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125"), f"NEB_fixed_{num}")
    vac_pot_path = os.path.abspath('/home/myless/Packages/structure_maker/Potentials/Vacancy_Train_Results/bestF_epoch89_e2_f28_s55_mNA.pth.tar')
    neb_pot_path = os.path.abspath('/home/myless/Packages/structure_maker/Potentials/Jan_26_100_Train_Results/bestF_epoch75_e3_f23_s23_mNA.pth.tar')

    # Ensure paths are correct
    assert os.path.isfile(vac_pot_path), f"Vacancy potential file not found: {vac_pot_path}"
    assert os.path.isfile(neb_pot_path), f"NEB potential file not found: {neb_pot_path}"

    # Check if CUDA is available and get device count
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you are running on a machine with GPUs.")

    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {device_count}")

    # Assign GPUs
    if device_count >= 2:
        vac_device = torch.device('cuda:0')
        neb_device = torch.device('cuda:1')
    else:
        raise RuntimeError("Not enough GPUs available. At least 2 GPUs are required.")

    print(f"Running vac_potentials on device: {vac_device}")
    print(f"Running neb_potentials on device: {neb_device}")

    # Initialize calculators with the appropriate devices
    vac_calculator = CHGNetCalculator(CHGNet.from_file(vac_pot_path), use_device=vac_device)
    neb_calculator = CHGNetCalculator(CHGNet.from_file(neb_pot_path), use_device=neb_device)

    # Mock function to simulate processing
    def simulate_processing(calculator, job_path):
        print(f"Processing job path {job_path} on {calculator.device}")
        # Add any additional test logic here

    # Run your test processes
    simulate_processing(vac_calculator, job_path)
    simulate_processing(neb_calculator, job_path)

if __name__ == '__main__':
    import sys
    base_directory = sys.argv[1]
    main(base_directory)
