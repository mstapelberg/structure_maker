import os
import json
import sys
import numpy as np
import random
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Outcar
from ase.db import connect
from ase.io import write
from ase.visualize import view
from pymatgen.entries.computed_entries import ComputedStructureEntry
from monty.json import MontyEncoder, MontyDecoder
from pymatgen.io.vasp import Poscar
from ase.filters import FrechetCellFilter 
from pymatgen.transformations.standard_transformations import PerturbStructureTransformation
from mace.calculators import MACECalculator
from ase.optimize import LBFGS, FIRE, BFGS, MDMin, QuasiNewton


sys.path.append('../Modules')
from defect_maker import make_defects, return_x_neighbors
from vasp_misc import *
# Function to load and sort the structure
def load_and_sort_structure(entry):
    return Structure.from_dict(entry.structure.as_dict()).get_sorted_structure()

def percent_difference(value1, value2):
    """
    Calculate the percent difference between two numbers.

    Parameters:
    value1 (float): The first number.
    value2 (float): The second number.

    Returns:
    float: The percent difference between the two numbers.
    """
    try:
        difference = abs(value1 - value2)
        average = (value1 + value2) / 2
        percent_diff = (difference / average) * 100
        return percent_diff
    except ZeroDivisionError:
        return float('inf')  # Return infinity if the average is zero
def read_contcar_direct(contcar_file):
    try:
        with open(contcar_file, 'r') as file:
            lines = file.readlines()
            # Read the lattice constant
            lattice_constant = float(lines[1].strip())
            # Read the lattice vectors
            lattice_vectors = [list(map(float, line.strip().split())) for line in lines[2:5]]
            # Read the elements and their counts
            elements_line = lines[5].strip().split()
            if '/' in elements_line[0]:
                # Handle format where elements are followed by identifiers
                #elements = [element.split('/')[0] for element in elements_line]
                elements = [element.split('/')[0].rstrip('_pv').rstrip('_sv') for element in elements_line]

            else:
                # Handle format where elements are directly listed
                elements = elements_line
            print(elements)
            element_counts = list(map(int, lines[6].strip().split()))
            # Create a list of species that matches the number of coordinates
            species = [element for element, count in zip(elements, element_counts) for _ in range(count)]
            # coordinate type
            coord_type = lines[7].strip()
            if coord_type.startswith(('c','C')):
                cart = True
            elif coord_type.startswith(('d','D')):
                cart = False
            # Read the coordinates
            coordinates_start_index = 8
            coordinates = []
            for line in lines[coordinates_start_index:coordinates_start_index + sum(element_counts)]:
                parts = line.strip().split()
                coordinates.append(list(map(float, parts[:3])))

            # Convert the coordinates from direct to Cartesian
            #cartesian_coordinates = [
                #[sum(a*b for a, b in zip(coord, vector)) for vector in zip(*lattice_vectors)]
                #for coord in coordinates
            #]
            # Create the structure
            contcar = Structure(lattice_vectors, species, coords=coordinates, coords_are_cartesian=cart)
            return contcar
    except Exception as e:
        print(f"Error reading CONTCAR file {contcar_file}: {e}")
        raise e
        return None

def find_target_atoms(structure, N, neighbor_distance=2, cutoff_distance=5):
    all_indices = list(range(len(structure)))
    random.shuffle(all_indices)
    target_atoms = []
    neighbor_sets = []

    print(f"All indices: {all_indices}")

    while all_indices and len(target_atoms) < N:
        index = all_indices.pop()
        neighbors = []
        for distance in range(1, neighbor_distance + 1):
            neighbors_distance, _ = return_x_neighbors(structure, target_atom_index=index, x_neighbor=distance, alat=structure.lattice.a)
            neighbors.extend(neighbors_distance)
        
        print(f"Index: {index}, Neighbors: {neighbors}")

        if not any(set(neighbors).intersection(neighbor_set) for neighbor_set in neighbor_sets):
            # Check if the distance to all existing target atoms is greater than the cutoff distance
            if all(structure.get_distance(index, target_atom) > cutoff_distance for target_atom in target_atoms):
                target_atoms.append(index)
                neighbor_sets.append(set(neighbors))
                print(f"Selected target atom index: {index}")

    #return target_atoms if len(target_atoms) == N else None
    return target_atoms

def old_randomly_pick_sites(structure, n, initial_cutoff=1.25, max_attempts=1000, reduction_factor=0.9):
    """
    Randomly selects a specified number of sites from a given structure.

    Args:
        structure (Structure): The structure from which to randomly select sites.
        n (int): The number of sites to randomly select.
        initial_cutoff (float, optional): The initial cutoff distance for site selection. Defaults to 1.25.
        max_attempts (int, optional): The maximum number of attempts to make for site selection. Defaults to 1000.
        reduction_factor (float, optional): The reduction factor for the cutoff distance after each unsuccessful attempt. Defaults to 0.9.

    Returns:
        list: A list of randomly selected sites from the structure.

    Raises:
        ValueError: If the number of sites to pick is greater than the number of sites in the structure.
    """
    # Ensure that n is not greater than the number of sites in the structure
    if n > len(structure.sites):
        raise ValueError("The number of sites to pick cannot be greater than the number of sites in the structure.")
    
    # Randomly select n sites from the structure with iterative reduction in cutoff
    random_sites = []
    cutoff = initial_cutoff
    while len(random_sites) < n:
        attempts = 0
        while len(random_sites) < n and attempts < max_attempts:
            potential_site = random.choice(structure.sites)
            if all(np.linalg.norm(np.array(potential_site.coords) - np.array(site.coords)) > cutoff * min(structure.lattice.abc) for site in random_sites):
                random_sites.append(potential_site)
            attempts += 1
        
        if len(random_sites) < n:
            cutoff *= reduction_factor
            random_sites = []  # Reset and try again with a reduced cutoff
    
    return random_sites

import random
import numpy as np

def randomly_pick_sites(structure, n, initial_cutoff=1.25, max_attempts=1000, reduction_factor=0.9):
    """
    Randomly selects a specified number of site indices from a given structure.

    Args:
        structure (Structure): The structure from which to randomly select sites.
        n (int): The number of sites to randomly select.
        initial_cutoff (float, optional): The initial cutoff distance for site selection. Defaults to 1.25.
        max_attempts (int, optional): The maximum number of attempts to make for site selection. Defaults to 1000.
        reduction_factor (float, optional): The reduction factor for the cutoff distance after each unsuccessful attempt. Defaults to 0.9.

    Returns:
        list: A list of indices of randomly selected sites from the structure.

    Raises:
        ValueError: If the number of sites to pick is greater than the number of sites in the structure.
    """
    # Ensure that n is not greater than the number of sites in the structure
    if n > len(structure.sites):
        raise ValueError("The number of sites to pick cannot be greater than the number of sites in the structure.")
    
    # Randomly select n site indices from the structure with iterative reduction in cutoff
    random_site_indices = []
    cutoff = initial_cutoff
    while len(random_site_indices) < n:
        attempts = 0
        while len(random_site_indices) < n and attempts < max_attempts:
            potential_index = random.randint(0, len(structure.sites) - 1)
            potential_site = structure.sites[potential_index]
            if all(np.linalg.norm(np.array(potential_site.coords) - np.array(structure.sites[idx].coords)) > cutoff * min(structure.lattice.abc) for idx in random_site_indices):
                random_site_indices.append(potential_index)
            attempts += 1
        
        if len(random_site_indices) < n:
            cutoff *= reduction_factor
            random_site_indices = []  # Reset and try again with a reduced cutoff
    
    return random_site_indices

# Function to select a random neighbor
def select_random_neighbor(structure, target_atom_index, x_neighbor):
    neighbors, _ = return_x_neighbors(structure, target_atom_index, x_neighbor, structure.lattice.a)
    if neighbors:
        return random.choice(neighbors)
    return None

# Function to create and save structures with vacancies
def create_and_save_structures(entries, N, job_path, cutoff_distance=1.25, mace_path = '../Potentials/Mace/vcrtiwzr_vac_stress_e1_f10_s100_stagetwo_compiled.model',device='cpu'):
    for k, entry in enumerate(entries):
        print(f"Processing entry {k+1}/{len(entries)}...")
        #structure = load_and_sort_structure(entry)
        structure = entry[0]
        #print(structure)
        #target_atoms = find_target_atoms(structure, N, neighbor_distance, cutoff_distance)
        target_atoms = randomly_pick_sites(structure, N, initial_cutoff= cutoff_distance)
        if not target_atoms:
            print(f"No suitable target atoms found for entry {k+1}. Skipping...")
            continue
        print(f"Found target atoms for entry {k+1}: {target_atoms}")

        for t, target_atom_index in enumerate(target_atoms):
            print("On Target Atom: ", target_atom_index)
            start_structure, _ = make_defects(structure, target_atom_index, target_atom_index)
            print("Start defect made")
            if start_structure is None:
                print(f"Failed to create start structure for entry {k+1}, target atom {target_atom_index}.")
                continue
            #relax the start structure
            start_relaxed = mace_relaxer(AseAtomsAdaptor.get_atoms(start_structure),
                                            model_path= mace_path ,
                                            fmax=0.01,
                                            relax_cell= True, 
                                            optimizer='FIRE', 
                                            steps = 500, device=device)
            probabilities = [0.8, 0.2]
            selected_sites = []
            rejected_sites = []
            while len(selected_sites) < N: 
            #for x_neighbor in [1, 2]: # removed nextnextnext nearest neighbor
                x_neighbor = random.choices([1, 2], probabilities)[0]
                print("Neigbor distance: ", x_neighbor)
                vac_site = select_random_neighbor(structure, target_atom_index, x_neighbor)
                if vac_site is not None and vac_site not in selected_sites and vac_site not in rejected_sites:
                    
                    _, end_structure = make_defects(structure, target_atom_index, vac_site)
                    if end_structure is None:
                        print(f"Failed to create end structure for entry {k+1}, target atom {target_atom_index}, vac_site {vac_site}.")
                        continue
                    #relax the end structure
                    end_relaxed = mace_relaxer(AseAtomsAdaptor.get_atoms(end_structure),
                                                    model_path= mace_path ,
                                                    fmax=0.01,
                                                    relax_cell= True, 
                                                    optimizer='FIRE', 
                                                    steps = 500, device=device)
                    vol_mismatch = percent_difference(start_relaxed.get_volume(), end_relaxed.get_volume())
                    if vol_mismatch > 5:
                        print(f"Volume difference between start and end structures is too large: {percent_difference(start_relaxed.get_volume(), end_relaxed.get_volume())}%. Skipping...")
                        rejected_sites.append(vac_site)
                        continue

                    selected_sites.append(vac_site)
                    # Create directory and filenames
                    #directory = os.path.join(job_path, f"structure_{k}_vac_site_{n}")
                    directory = os.path.join(job_path, entry[1])
                    os.makedirs(directory, exist_ok=True)
                    print(f"Created directory: {directory}")

                    #start_filename = os.path.join(directory, f"structure_{k}_vac_site_{n}_start.vasp")
                    start_filename = os.path.join(directory, f"vac_site_{t}_start.vasp")
                    end_filename = os.path.join(directory, f"vac_site_{t}_end_site_{vac_site}.vasp")
                    #end_filename = os.path.join(directory, f"structure_{k}_vac_site_{n}_end_site_{vac_site}.vasp")


                    # Write the structures to POSCAR files
                    Poscar(start_structure).write_file(start_filename)
                    print(f"Written start structure to {start_filename}")
                    Poscar(end_structure).write_file(end_filename)
                    print(f"Written end structure to {end_filename}")

                    # write the volume mismatch to a file
                    with open(os.path.join(directory, f"vac_site_{t}_volume_mismatch.txt"), 'w') as f:
                        f.write(f"Volume mismatch between start and end structures: {vol_mismatch}%")


def check_overlapping_atoms(structure, distance_threshold=0.4):
    """
    Check if a pymatgen structure has overlapping atoms.
    
    Parameters:
    structure (Structure): The pymatgen structure to check.
    distance_threshold (float): The distance threshold below which atoms are considered overlapping.
    
    Returns:
    bool: True if there are overlapping atoms, False otherwise.
    """
    distances = structure.distance_matrix
    num_atoms = len(structure)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if distances[i, j] < distance_threshold:
                return True
    return False

def print_min_distance(structure: Structure):
    
    min_distance = float('inf')
    atom1, atom2 = None, None

    for i in range(len(structure)):
        for j in range(i+1, len(structure)):
            distance = structure[i].distance(structure[j])
            if distance < min_distance:
                min_distance = distance
                atom1, atom2 = i, j

    print(f"The minimum distance between any two atoms in the structure is: {min_distance}")
    print(f"The atoms are at indexes {atom1} and {atom2}")
    print(f"The coordinates of the atoms are {structure[atom1].coords} and {structure[atom2].coords}")

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
    return new_atoms






relaxed_perfect_supercells = []
relaxed_perfect_path = '../Visualization/Job_Structures/Pre_VASP/VCrTiWZr_Summit/gen_0_4/Relaxed_Perfect_Structures_mace'
# get the relaxed perfect structures
for file in os.listdir(relaxed_perfect_path):
    if file.endswith('.vasp'):
        structure = Structure.from_file(os.path.join(relaxed_perfect_path, file))
        relaxed_perfect_supercells.append([structure, file.split('.')[0]])
job_path = '../Visualization/Job_Structures/Pre_VASP/VCrTiWZr_Summit/gen_0_4/Pre_NEB_mace'
N = 5
create_and_save_structures(relaxed_perfect_supercells, N, job_path, cutoff_distance=1.25,device='cuda')
