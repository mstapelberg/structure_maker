from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
import numpy as np

def load_and_sort_structure(file_path):
    """
    Loads the structure from a file and sorts it by element type.
    
    Parameters:
    file_path (str): The path to the structure file.
    
    Returns:
    sorted_structure (Structure): The sorted atomic structure.
    """
    structure = Structure.from_file(file_path)
    sorted_structure = structure.get_sorted_structure()
    return sorted_structure

def return_x_neighbors(structure, target_atom_index, x_neighbor, alat, tolerance=1.05):
    """
    Returns the indices and distances of the nearest neighbors of a target atom.
    
    Parameters:
    structure (Structure): The atomic structure (pymatgen Structure object).
    target_atom_index (int): The index of the target atom.
    x_neighbor (int): The number of neighbors to return (1, 2, or 3).
    alat (float): The lattice constant.
    tolerance (float, optional): The tolerance factor for the cutoff distance. Default is 1.05.
    
    Returns:
    nearest_neighbors (list): The indices of the nearest neighbors.
    distances (list): The distances to the nearest neighbors.
    """
    target_site = structure[target_atom_index]

    # Get the cutoff distance for the nearest neighbors
    if x_neighbor == 1:
        cutoff = alat * np.sqrt(3) / 2 * tolerance  # 5% larger than the nearest neighbor distance for tolerance
    elif x_neighbor == 2:
        cutoff = alat * tolerance 
    elif x_neighbor == 3:
        cutoff = alat * np.sqrt(2) * tolerance
    else:
        print('x_neighbor must be 1, 2, or 3')
        return None

    print(f"Cutoff distance: {cutoff}")

    # Manually compute distances to all atoms and filter within cutoff distance
    filtered_neighbors = []
    for i, site in enumerate(structure):
        if i != target_atom_index:  # Skip the target atom itself
            distance = target_site.distance(site)
            if distance <= cutoff:
                filtered_neighbors.append({'site': site, 'index': i, 'distance': distance})

    print(f"Filtered neighbors within cutoff: {filtered_neighbors}")

    # Get the number of indices based on if we need the nearest, next-nearest, or next-next-nearest neighbors
    if x_neighbor == 1:
        sorted_neighbors = sorted(filtered_neighbors, key=lambda x: x['distance'])[:8]
    elif x_neighbor == 2:
        sorted_neighbors = sorted(filtered_neighbors, key=lambda x: x['distance'])[8:14]
    elif x_neighbor == 3:
        sorted_neighbors = sorted(filtered_neighbors, key=lambda x: x['distance'])[14:26]
    else:
        print('x_neighbor must be 1, 2, or 3')
        return None

    print(f"Sorted neighbors: {sorted_neighbors}")

    # Get the indices and distances of the nearest neighbors sorted by distance
    nearest_neighbors = [n['index'] for n in sorted_neighbors]
    distances = [n['distance'] for n in sorted_neighbors]

    print(f"Nearest neighbors: {nearest_neighbors}")
    print(f"Distances: {distances}")

    return nearest_neighbors, distances


def make_defects(structure, target_atom_index, vac_site):
    """
    Creates START and END structures for NEB simulation by making a vacancy defect
    and moving a neighboring atom to the target position.
    
    Parameters:
    structure (Structure): The atomic structure (pymatgen Structure object).
    target_atom_index (int): The index of the atom to remove for the vacancy.
    vac_site (int): The index of the neighboring atom to move to the target position.
    
    Returns:
    start_structure (Structure): The START structure with the vacancy defect.
    end_structure (Structure): The END structure with the moved atom.
    """
    # Create copies of the structure for START and END
    start_structure = structure.copy()
    end_structure = structure.copy()

    # get the target coordinates from the original structure
    target_coords = structure[target_atom_index].coords

    # move the vac_site atom in the end structure to the target position
    end_structure[vac_site].coords = target_coords
    
    # Remove the target atom to create a vacancy in both structures
    start_structure.remove_sites([target_atom_index])
    end_structure.remove_sites([target_atom_index])

    # Move the atom from vac_site to the target_atom_index position in the END structure
    #atom_to_move = end_structure[vac_site].copy()
    #atom_to_move.coords = structure[target_atom_index].coords

    # Add the moved atom to the END structure
    #end_structure.append(atom_to_move)

    # Sort both structures
    #start_structure = start_structure.get_sorted_structure()
    #end_structure = end_structure.get_sorted_structure()

    return start_structure, end_structure

def make_defect(structure, target_atom_index):
    """
    Makes a vacancy defect in the atomic structure.
    
    Parameters:
    structure (Structure): The atomic structure (pymatgen Structure object).
    target_atom_index (int): The index of the atom to remove.
    
    Returns:
    new_structure (Structure): The atomic structure with the vacancy defect.
    """
    # make a copy of the structure
    new_structure = structure.copy()
    new_structure.remove_sites([target_atom_index])
    return new_structure


def write_sorted_poscar(structure, filename='POSCAR'):
    """
    Writes the structure to a POSCAR file sorted by element type.
    
    Parameters:
    structure (Structure): The atomic structure (pymatgen Structure object).
    filename (str): The filename for the POSCAR file.
    """
    # sort the structure by element type
    sorted_structure = structure.get_sorted_structure()
    poscar = Poscar(sorted_structure)
    poscar.write_file(filename)
