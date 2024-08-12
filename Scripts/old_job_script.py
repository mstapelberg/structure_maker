import os
import torch
from ase.io import read, write
from ase.mep import DyNEB
from ase import Atoms
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import FrechetCellFilter 
from ase.constraints import FixAtoms
from ase.io import Trajectory
import pickle
from ase.atoms import Atoms, units 
import numpy as np 
import json, os
from ase.optimize import LBFGS, FIRE, BFGS, MDMin, QuasiNewton 
from ase.optimize.precon import PreconLBFGS
from pymatgen.core import Structure 
import sys
sys.path.append('/home/myless/Packages/structure_maker/Modules')
from defect_maker import make_defects, return_x_neighbors
from vasp_misc import *
from ModNEB_Barrier import NEB_Barrier

def create_and_run_neb_files(base_directory, job_path, relax=True, num_images=5, vac_calculator=None, neb_calculator=None):
    num_failed = 0
    for subdir in os.listdir(base_directory):
        if subdir.startswith('structure'):
            print(subdir)
            subdir_path = os.path.join(base_directory, subdir)
            files = os.listdir(subdir_path)
            vac_sites = {}
            for file in files:
                if file.startswith('structure_') and file.endswith('.vasp'):
                    parts = file.split('_')
                    comp = parts[3]
                    vac_site = parts[6]
                    if vac_site not in vac_sites:
                        vac_sites[vac_site] = {'start': None, 'end': []}
                    if 'start' in file:
                        vac_sites[vac_site]['start'] = file
                    elif 'end' in file:
                        vac_sites[vac_site]['end'].append(file)
            for vac_site, files in vac_sites.items():
                start_file = files['start']
                end_files = files['end']
                if start_file is None or not end_files:
                    print(f"Skipping vac_site_{vac_site} in {subdir} due to missing start or end files.")
                    continue
                start_structure = read(os.path.join(subdir_path, start_file))
                for end_file in end_files:
                    end_structure = read(os.path.join(subdir_path, end_file))
                    neb_dir = os.path.join(job_path, subdir, f'neb_vac_site_{vac_site}_to_{end_file.split("_")[-1].split(".")[0]}')
                    os.makedirs(neb_dir, exist_ok=True)
                    if os.path.exists(os.path.join(neb_dir, 'results.json')):
                        print(f"NEB interpolation for vac_site_{vac_site} in {subdir} already completed.")
                        continue
                    comp = start_file.split('_')[3]
                    barrier = NEB_Barrier(start=start_structure,
                                          end=end_structure,
                                          vasp_energies=[0, 0],
                                          composition=comp, #start_structure.get_chemical_formula(),
                                          structure_number=int(subdir.split('_')[1]),
                                          defect_number=int(vac_site),
                                          direction=end_file.split("_")[-1].split(".")[0],
                                          root_path=neb_dir)
                    barrier.neb_run(num_images=5,
                                    potential=neb_calculator,
                                    vac_potential=vac_calculator,
                                    run_relax=False,
                                    num_steps=200)
    print(f"NEB interpolation completed with {num_failed} failures.")

def chgnet_relaxer(atoms, calculator, fmax=0.01, steps=250, verbose=False, relax_cell=True, loginterval=1):
    if isinstance(atoms, Structure):
        atoms = AseAtomsAdaptor.get_atoms(atoms)
    new_atoms = atoms.copy()
    new_atoms.calc = calculator 
    if relax_cell:
        ucf = FrechetCellFilter(new_atoms)
        optimizer = PreconLBFGS(ucf)
    else:
        #new_atoms.set_constraint(FixAtoms(mask=[True for atom in new_atoms]))
        #ucf = FrechetCellFilter(new_atoms, constant_volume=True)
        ucf = new_atoms 
        optimizer = PreconLBFGS(ucf)
    optimizer.run(fmax=fmax, steps=steps)
    return new_atoms

#if __name__ == '__main__':
    #import sys
    #base_directory = sys.argv[1]
    ## get the number from base_directory, should be like dir_X
    #num = base_directory.split('_')[-1]
    #job_path = f'../Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125/NEB_fixed_{num}'
    #vac_pot_path = '/home/myless/Packages/structure_maker/Potentials/Vacancy_Train_Results/bestF_epoch89_e2_f28_s55_mNA.pth.tar'
    #neb_pot_path = '/home/myless/Packages/structure_maker/Potentials/Jan_26_100_Train_Results/bestF_epoch75_e3_f23_s23_mNA.pth.tar'
    #vac_calculator = CHGNetCalculator(CHGNet.from_file(vac_pot_path))
    #neb_calculator = CHGNetCalculator(CHGNet.from_file(neb_pot_path))
    #create_and_run_neb_files(base_directory, job_path, relax=True, vac_calculator=vac_calculator, neb_calculator=neb_calculator)


def main(base_directory):
    # Get the number from base_directory, should be like dir_X
    num = base_directory.split('_')[-1]
    job_path = f'/home/myless/Packages/structure_maker/Visualization/Job_Structures/Pre_VASP/VCrTi_Fixed_125/NEB_fixed_{num}'
    vac_pot_path = '/home/myless/Packages/structure_maker/Potentials/Vacancy_Train_Results/bestF_epoch89_e2_f28_s55_mNA.pth.tar'
    neb_pot_path = '/home/myless/Packages/structure_maker/Potentials/Jan_26_100_Train_Results/bestF_epoch75_e3_f23_s23_mNA.pth.tar'

    # Determine and set CUDA device based on environment variable
    cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Initialize calculators with the appropriate device
    #vac_calculator = CHGNetCalculator(CHGNet.from_file(vac_pot_path), use_device=device)
    #neb_calculator = CHGNetCalculator(CHGNet.from_file(neb_pot_path), use_device=device)
    vac_calculator = CHGNet.from_file(vac_pot_path, use_device=device)
    neb_calculator = CHGNet.from_file(neb_pot_path, use_device=device)

    create_and_run_neb_files(base_directory, job_path, relax=True, vac_calculator=vac_calculator, neb_calculator=neb_calculator)

if __name__ == '__main__':
    import sys
    base_directory = sys.argv[1]
    main(base_directory)
