from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.core.composition import Composition
import math
import random



def create_cca_primitive(comp_list, a, prim=True):
    """
    Create a primitive or orthorhombic BCC (Body Centered Cubic) structure.

    Args:
        comp_list (list): A list of dictionaries representing the species composition at each site.
                          Each dictionary contains the species and their fraction at that site.
        a (float): The lattice constant.
        prim (bool, optional): Whether to create a primitive or non-primitive structure.
                               Defaults to True.

    Returns:
        Structure: The generated CCA structure.

    """
    # Create the species composition for each site as a Composition object
    # We need to pass the composition as a list of dicts where each dict represents the species and their fraction at that site
    species_composition = Composition(comp_list)

    # Coordinates for the primitive cell, assuming a BCC structure
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]  # Two distinct positions in BCC

    if prim:
        # Define the primitive lattice vectors for BCC
        lattice_vectors = [
            [0.5 * a, 0.5 * a, 0.5 * -a],
            [0.5 * a, 0.5 * -a, 0.5 * a],
            [0.5 * -a, 0.5 * a, 0.5 * a]
        ]

        # Create the lattice
        lattice = Lattice(lattice_vectors)
        # Initialize the structure with the same composition for each of the two sites
        structure = Structure(lattice, [species_composition], [[0,0,0]])
    else:
        # For non-primitive, we use the same approach but might specify a different spacegroup or adjustments
        lattice = Lattice.cubic(a)
        structure = Structure.from_spacegroup("Im-3m", lattice, [species_composition], [[0,0,0]])

    return structure


def closest_composition(comp, num_atoms, bal_element):
    """
    Calculate the closest composition of elements given a target number of atoms and a balance element.

    Args:
        comp (dict): A dictionary representing the composition of elements, where the keys are element symbols and the values are fractions.
        num_atoms (int): The target number of atoms.
        bal_element (str): The element symbol to be used as the balance element.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary represents the number of atoms assigned to each element, excluding the balance element. The second dictionary represents the actual fractions of each element in the composition, rounded to 3 decimals.
    """

    # first sum up the fractions of the composition that are not the balance element
    total_fraction = sum([comp[element] for element in comp if element != bal_element])

    # now set the balance element to the difference between the total fraction and 1
    comp[bal_element] = 1 - total_fraction

    # now we need to normalize the composition to the number of atoms, ignoring the balance element
    # we will round up to the nearest integer for the non balance elements

    # first calculate the number of atoms for each element
    atoms = {element: math.ceil(comp[element] * num_atoms) for element in comp if element != bal_element}

    # then calculate the number of atoms assigned so far

    assigned_atoms = sum(atoms.values())

    # now we need to adjust the balance element to make sure the total number of atoms is correct
    atoms[bal_element] = num_atoms - assigned_atoms

    # now we need to recalculate the actual fractions to make sure they sum to 1
    # let's first start with calculating the fractions of the non balance elements and rounding to 3 decimals
    actual_fractions = {element: round(atoms[element] / num_atoms, 3) for element in atoms if element != bal_element}

    # now we need to calculate the fraction of the balance element by subtracting the sum of the fractions of the non balance elements from 1
    actual_fractions[bal_element] = round(1 - sum(actual_fractions.values()), 3)

    return atoms, actual_fractions

def generate_random_supercells(composition, num_structures, lattice_parameter=3.01, supercell_size=4, supercell_type='cubic', seed=42):
    """
    Generate random supercells based on the given composition.

    Args:
        composition (dict): A dictionary representing the composition of the supercell, where the keys are the element symbols and the values are the number of atoms for each element.
        num_structures (int): The number of supercells to generate.
        lattice_parameter (float, optional): The lattice parameter of the supercell. Defaults to 3.01.
        supercell_size (int, optional): The size of the supercell in each dimension. Defaults to 4.
        supercell_type (str, optional): The type of supercell to generate. Can be 'cubic' or 'prim'. Defaults to 'cubic'.
        seed (int, optional): The seed value for the random number generator. Defaults to 42.

    Returns:
        list: A list of generated supercells, each represented as a `pymatgen.Structure` object.
    """
    random.seed(seed) 
    supercells = []

    comp_list = [key for key in composition]
    for _ in range(num_structures):
        if supercell_type == 'cubic':
            #prim_cell = create_cca_primitive()
            # what is the difference between giving a one element per site, versus giving a partial occupancy for each element? 
            prim_cell = Structure(Lattice.cubic(lattice_parameter), [comp_list[0],comp_list[0]], [[0, 0, 0],[0.5, 0.5, 0.5]])
        elif supercell_type == 'prim':
            prim_cell = Structure(Lattice.cubic(lattice_parameter), [comp_list[0]], [[0, 0, 0]])
        
        supercell = prim_cell * (supercell_size,supercell_size,supercell_size)

        all_indices = list(range(len(supercell.sites)))

        for element, count in composition.items():
            selected_indices = random.sample(all_indices, count)
            
            for index in selected_indices:
                supercell.replace(index,element)
            
            all_indices = [index for index in all_indices if index not in selected_indices]
        
        supercell = supercell.get_sorted_structure()

        supercells.append(supercell)

    return supercells
