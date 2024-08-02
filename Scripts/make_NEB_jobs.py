import os, sys
import scipy
from ase.io import read, write
from ase.mep import NEB
from ase import Atoms
from nequip.ase import NequIPCalculator
from mace.calculators import MACECalculator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import FrechetCellFilter 
from ase.constraints import FixAtoms
from ase.io import Trajectory
import pickle
from ase.atoms import Atoms, units 
import numpy as np 
import json, os
from ase.optimize import LBFGS, FIRE, BFGS, MDMin, QuasiNewton 
from pymatgen.core import Structure 

sys.path.append('../Modules')
from NEB_Barrier import NEB_Barrier



def _create_neb_files(base_directory, job_path, relax = True, num_images=5):
    # Iterate through each subdirectory in base_directory that starts with "supercell"
    num_failed = 0
    for subdir in os.listdir(base_directory):
        if subdir.startswith('supercell'):
            subdir_path = os.path.join(base_directory, subdir)
            
            # Parse the subdirectory to identify all vac_site files
            files = os.listdir(subdir_path)
            vac_sites = {}

            for file in files:
                if file.startswith('vac_site_') and file.endswith('.vasp'):
                    parts = file.split('_')
                    vac_site = parts[2]
                    if vac_site not in vac_sites:
                        vac_sites[vac_site] = {'start': None, 'end': []}
                    if 'start' in file:
                        vac_sites[vac_site]['start'] = file
                    elif 'end' in file:
                        vac_sites[vac_site]['end'].append(file)

            # Process each vac_site
            for vac_site, files in vac_sites.items():
                start_file = files['start']
                end_files = files['end']
                
                if start_file is None or not end_files:
                    print(f"Skipping vac_site_{vac_site} in {subdir} due to missing start or end files.")
                    continue
                
                # Load the start structure
                start_structure = read(os.path.join(subdir_path, start_file))
                if relax:
                    potential_path = '../Potentials/fin_vcrtiwzr_novirial_efs.pth'
                    species = {'V': 'V', 'Cr': 'Cr', 'Ti': 'Ti', 'W': 'W', 'Zr': 'Zr'}
                    start_structure = allegro_relaxer(start_structure, potential_path, species,relax_cell=False, fmax=0.04, steps=1000)
                
                for end_file in end_files:
                    # Load the end structure
                    end_structure = read(os.path.join(subdir_path, end_file))
                    if relax:
                        potential_path = '../Potentials/fin_vcrtiwzr_novirial_efs.pth'
                        species = {'V': 'V', 'Cr': 'Cr', 'Ti': 'Ti', 'W': 'W', 'Zr': 'Zr'}
                        end_structure = allegro_relaxer(end_structure, potential_path, species,relax_cell=False, fmax=0.04, steps=1000)
                    
                    # Ensure start_structure and end_structure are Atoms objects
                    if not isinstance(start_structure, Atoms) or not isinstance(end_structure, Atoms):
                        print(f"Skipping vac_site_{vac_site} in {subdir} due to invalid structure types.")
                        continue
                    # Create NEB interpolation
                    images = [start_structure]
                    images += [start_structure.copy() for _ in range(num_images)]
                    images.append(end_structure)
                    
                    
                    try:
                        neb = NEB(images)
                        neb.interpolate(mic=True, apply_constraint=False)
                    except Exception as e:
                        failure_message = f"Error interpolating NEB for vac_site_{vac_site} in {subdir}: {e}"
                        with open(os.path.join(job_path, 'failures.txt'), 'a') as f:
                            f.write(failure_message + '\n')
                        num_failed += 1
                        continue
                    # Save the interpolated structures
                    neb_dir = os.path.join(job_path, subdir, f'neb_vac_site_{vac_site}_to_{end_file.split("_")[-1].split(".")[0]}')
                    os.makedirs(neb_dir, exist_ok=True)
                    
                    for i, image in enumerate(images):
                        image_filename = os.path.join(neb_dir, f'POSCAR_{i}.vasp')
                        write(image_filename, image)
                        #print(f"Written NEB image {i} to {image_filename}")
                    
                    if relax:
                        # save the start and end energies as a json in the neb_dir
                        start_energy = start_structure.get_potential_energy()
                        end_energy = end_structure.get_potential_energy()
                        energies = {'start_energy': start_energy, 'end_energy': end_energy}
                        with open(os.path.join(neb_dir, 'energies.json'), 'w') as f:
                            json.dump(energies, f)
                            #print(f"Written start and end energies to energies.json in {neb_dir}")
    print(f"NEB interpolation completed with {num_failed} failures.")

def create_neb_files(base_directory, job_path, output_dir, relax = True, num_images=5, mace_path = '../Potentials/Mace/vcrtiwzr_vac_stress_e1_f10_s100_stagetwo_compiled.model'):
    # Iterate through each subdirectory in base_directory that starts with "supercell"
    num_failed = 0
    for subdir in os.listdir(base_directory):
        if subdir.startswith('supercell'):
            subdir_path = os.path.join(base_directory, subdir)
            
            # Parse the subdirectory to identify all vac_site files
            files = os.listdir(subdir_path)
            vac_sites = {}

            for file in files:
                if file.startswith('vac_site_') and file.endswith('.vasp'):
                    parts = file.split('_')
                    vac_site = parts[2]
                    if vac_site not in vac_sites:
                        vac_sites[vac_site] = {'start': None, 'end': []}
                    if 'start' in file:
                        vac_sites[vac_site]['start'] = file
                    elif 'end' in file:
                        vac_sites[vac_site]['end'].append(file)

            # Process each vac_site
            for vac_site, files in vac_sites.items():
                start_file = files['start']
                end_files = files['end']
                
                if start_file is None or not end_files:
                    print(f"Skipping vac_site_{vac_site} in {subdir} due to missing start or end files.")
                    continue
                
                # Load the start structure
                start_structure = read(os.path.join(subdir_path, start_file))
                if relax:
                    #potential_path = '../Potentials/fin_vcrtiwzr_novirial_efs.pth'
                    #species = {'V': 'V', 'Cr': 'Cr', 'Ti': 'Ti', 'W': 'W', 'Zr': 'Zr'}
                    start_structure = mace_relaxer(atoms = start_structure, 
                                                   model_path= mace_path,
                                                   relax_cell=False, 
                                                   fmax=0.01, 
                                                   steps=1000)
                
                for end_file in end_files:
                    # Load the end structure
                    end_structure = read(os.path.join(subdir_path, end_file))
                    if relax:
                        #potential_path = '../Potentials/fin_vcrtiwzr_novirial_efs.pth'
                        #species = {'V': 'V', 'Cr': 'Cr', 'Ti': 'Ti', 'W': 'W', 'Zr': 'Zr'}
                        end_structure = mace_relaxer(atoms = end_structure, 
                                                   model_path= mace_path,
                                                   relax_cell=False, 
                                                   fmax=0.01, 
                                                   steps=1000)
                    
                    neb_dir = os.path.join(job_path, subdir, f'neb_vac_site_{vac_site}_to_{end_file.split("_")[-1].split(".")[0]}')
                    os.makedirs(neb_dir, exist_ok=True)
                    
                    start_energy = start_structure.get_potential_energy()
                    end_energy = end_structure.get_potential_energy()
                    # save the start and end energies as a json in the neb_dir
                    with open(os.path.join(neb_dir, 'energies.json'), 'w') as f:
                        json.dump({'start_energy': start_energy, 'end_energy': end_energy}, f)
                    
                    # check if the results.json file exists
                    if os.path.exists(os.path.join(neb_dir, 'results.json')):
                        print(f"NEB interpolation for vac_site_{vac_site} in {subdir} already completed.")
                        continue
                    
                    #print(subdir)
                    barrier = NEB_Barrier(start=start_structure,
                                          end=end_structure,
                                          vasp_energies=[start_energy, end_energy],
                                          composition= start_structure.get_chemical_formula(),
                                          structure_number = int(subdir.split('_')[-1]),
                                          defect_number = int(vac_site),
                                          direction = end_file.split("_")[-1].split(".")[0],
                                          root_path = neb_dir
                                          )
                    barrier.neb_run(num_images=num_images,
                                    potential = mace_path,
                                    vac_potential = None,
                                    run_relax = False,
                                    num_steps = 200,
                                    neb_run = False)
                    
                    barrier.create_neb_vasp_job(output_directory = output_dir,
                                                 num_kpts = 3,
                                                 climb = False)
                    # save the name of neb_dir to barrier_path for later use

                    with open(os.path.join(barrier.neb_path, 'pre_neb_path.txt'), 'w') as f:
                        f.write(neb_dir)
                        
    print(f"NEB interpolation completed with {num_failed} failures.")

def mace_relaxer(atoms, model_path, fmax = 0.01, steps = 250, relax_cell=True, optimizer = 'LBFGS', device='cpu'):
    if isinstance(atoms, Structure):
        atoms = AseAtomsAdaptor.get_atoms(atoms)
    new_atoms = atoms.copy()
    new_atoms.calc = MACECalculator(model_paths=[model_path], device=device, default_dtype="float32")

    if relax_cell:
        ucf = FrechetCellFilter(new_atoms)
        #obs = TrajectoryObserver(ucf)
        if optimizer == 'LBFGS':
            optimizer = LBFGS(ucf)
        elif optimizer == 'BFGS':
            optimizer = BFGS(ucf)
        elif optimizer == 'MDMin':
            optimizer = MDMin(ucf)
        elif optimizer == 'QuasiNewton':
            optimizer = QuasiNewton(ucf)
        elif optimizer == 'FIRE':
            optimizer = FIRE(ucf)
        #optimizer.attach(obs, interval=loginterval)

    else:
        #constraints = FixAtoms(mask=[False] * len(new_atoms))  # Allow all atoms to move
        # Add constraints to atoms
        #new_atoms.set_constraint(constraints)
        ucf = new_atoms
        if optimizer == 'LBFGS':
            optimizer = LBFGS(ucf)
        elif optimizer == 'BFGS':
            optimizer = BFGS(ucf)
        elif optimizer == 'MDMin':
            optimizer = MDMin(ucf)
        elif optimizer == 'QuasiNewton':
            optimizer = QuasiNewton(ucf)
        elif optimizer == 'FIRE':
            optimizer = FIRE(ucf)
        #obs = TrajectoryObserver(atoms)
        #optimizer.attach(obs, interval=loginterval)

    optimizer.run(fmax=fmax, steps=steps)
    return new_atoms

def allegro_relaxer(atoms, potential_path, species, fmax = 0.01, steps = 250, verbose=False, relax_cell=True, optimizer = 'LBFGS', loginterval = 1):
    if isinstance(atoms, Structure):
        atoms = AseAtomsAdaptor.get_atoms(atoms)
    new_atoms = atoms.copy()
    new_atoms.calc = NequIPCalculator.from_deployed_model(
        model_path=potential_path,
        species_to_type_name = species
    )
    
    if relax_cell:
        ucf = FrechetCellFilter(new_atoms)
        #obs = TrajectoryObserver(ucf)
        if optimizer == 'LBFGS':
            optimizer = LBFGS(ucf)
        elif optimizer == 'BFGS':
            optimizer = BFGS(ucf)
        elif optimizer == 'MDMin':
            optimizer = MDMin(ucf)
        elif optimizer == 'QuasiNewton':
            optimizer = QuasiNewton(ucf)
        elif optimizer == 'FIRE':
            optimizer = FIRE(ucf)
        #optimizer.attach(obs, interval=loginterval)
        
    else:
        #constraints = FixAtoms(mask=[False] * len(new_atoms))  # Allow all atoms to move
        # Add constraints to atoms
        #new_atoms.set_constraint(constraints)
        print("Relaxing without cell relaxation")
        new_atoms.set_constraint(FixAtoms(mask=[True for atom in new_atoms]))
        ucf = FrechetCellFilter(new_atoms, constant_volume=True)
        if optimizer == 'LBFGS':
            optimizer = LBFGS(ucf)
        elif optimizer == 'BFGS':
            optimizer = BFGS(ucf)
        elif optimizer == 'MDMin':
            optimizer = MDMin(ucf)
        elif optimizer == 'QuasiNewton':
            optimizer = QuasiNewton(ucf)
        elif optimizer == 'FIRE':
            optimizer = FIRE(ucf)
        #obs = TrajectoryObserver(atoms)
        #optimizer.attach(obs, interval=loginterval)
    
    optimizer.run(fmax=fmax, steps=steps)
    return new_atom

# Example usage
base_directory = '../Visualization/Job_Structures/Pre_VASP/VCrTiWZr_Summit/gen_0_4/Pre_NEB_mace'
job_path = '../Visualization/Job_Structures/Pre_VASP/VCrTiWZr_Summit/gen_0_4/NEB_mace'
output_dir = '../Visualization/Job_Structures/Pre_VASP/VCrTiWZr_Summit/gen_0_4/NEB_mace_vasp'
create_neb_files(base_directory, 
                 job_path, 
                 output_dir, 
                 relax=True,
                 num_images=5,
                 mace_path = '../Potentials/Mace/vcrtiwzr_vac_stress_e1_f10_s100_stagetwo_compiled.model',
                 )
