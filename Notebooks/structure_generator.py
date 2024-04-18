import json
import os
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.util.coord import pbc_shortest_vectors
from random import shuffle

def calculate_atoms(supercell_size, composition):
    total_sites = np.prod(supercell_size)
    print(composition)
    atoms_count = {element: int(round(fraction * total_sites)) for element, fraction in composition.items()}
    balance_element = max(composition, key=composition.get)  # Assuming the balance element is the one with the highest fraction
    total_calculated = sum(atoms_count.values())
    
    # Adjust to ensure total sites match, prioritizing the balance element
    while total_calculated != total_sites:
        if total_calculated < total_sites:
            atoms_count[balance_element] += 1
        elif total_calculated > total_sites:
            atoms_count[balance_element] -= 1
        total_calculated = sum(atoms_count.values())
    
    # Ensure at least one atom of each element
    for element in composition:
        if atoms_count[element] == 0:
            atoms_count[element] = 1
            # Adjust the balance element to compensate
            atoms_count[balance_element] -= 1
    
    return atoms_count

def generate_unique_supercell(a, supercell_size, composition, json_file="supercells.json"):
    """
    Generate a unique supercell structure based on the given parameters.

    Parameters:
        a (float): The lattice constant.
        supercell_size (int): The size of the supercell.
        composition (dict): A dictionary representing the composition of the supercell.
                            The keys are the element symbols and the values are the number of atoms of each element.
        json_file (str): The path to the JSON file used to store existing supercell structures. Default is "supercells.json".

    Returns:
        Structure: The generated unique supercell structure.

    """
    atoms_count = calculate_atoms(supercell_size, composition)
    
    # Create a list of elements based on their counts
    elements_list = [element for element, count in atoms_count.items() for _ in range(count)]
    shuffle(elements_list)  # Randomize the list
    
    # Create a supercell structure
    lattice = Lattice.cubic(a)  # Arbitrary choice, you can customize this
    structure = Structure(lattice, elements_list, np.random.rand(len(elements_list), 3))
    
    # Check and update JSON
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            existing_structures = json.load(file)
    else:
        existing_structures = []
    
    # Convert structure to dict for JSON compatibility
    structure_dict = structure.as_dict()
    
    # Check uniqueness
    for existing in existing_structures:
        if existing == structure_dict:
            return generate_unique_supercell(supercell_size, composition, json_file)  # Recursion if duplicate
    
    existing_structures.append(structure_dict)
    
    with open(json_file, 'w') as file:
        json.dump(existing_structures, file)
    
    return structure


