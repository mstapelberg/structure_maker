import random
import numpy as np
from monty.serialization import loadfn, dumpfn
from pymatgen.core.structure import Structure
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion, RegressionData
from time import time 
from joblib import Parallel, delayed, cpu_count

def prim_entry_dataloader(prim_path, entry_path):
    """
    Load a primary structure and a list of entries from file paths.

    Args:
        prim_path (str): The file path of the primary structure.
        entry_path (str): The file path of the entries.

    Returns:
        tuple: A tuple containing the loaded primary structure and the list of entries.
    """
    prim = Structure.from_file(prim_path)
    entries = loadfn(entry_path)

    return prim, entries

# then define cluster subspace

def cluster_subspace_creator(prim, cutoffs, verbose=False):
    """
    Create a cluster subspace from a primitive structure.

    Parameters:
        prim (Structure): The primitive structure.
        cutoffs (float or dict): The cutoff distance(s) for creating the cluster subspace.
        verbose (bool, optional): If True, print the cluster subspace. Default is False.

    Returns:
        ClusterSubspace: The created cluster subspace.
    """

    subspace = ClusterSubspace.from_cutoffs(
        prim,
        cutoffs=cutoffs,
        basis='sinusoid',
        supercell_size='num_sites'
    )

    if verbose:
        print(subspace)
    
    return subspace

def structure_wrangler_creator(subspace, entries, ncpu = 1, verbose=False):
    """
    Creates a StructureWrangler object and adds entries to it.

    Args:
        subspace (str): The subspace for the StructureWrangler.
        entries (list): A list of entries to be added to the StructureWrangler.
        verbose (bool, optional): If True, prints the total number of structures that match. 
            Defaults to False.

    Returns:
        StructureWrangler: The created StructureWrangler object.

    """
    wrangler = StructureWrangler(subspace)
    
    if ncpu == -1:
        nprocs = -1 
        batch_size = 'auto'

        start = time()

        with Parallel(n_jobs=nprocs, batch_size=batch_size, verbose=True) as parallel:
            par_entries = parallel(delayed(wrangler.process_entry)(
            entry, verbose=False) for entry in entries
            )
        
        #unpack the items and remove Nones from structure that failed to match 
        par_entries = [entry for entry in par_entries if entry is not None]
        wrangler.append_entries(par_entries)
        end = time()
    elif ncpu == 1:
        for entry in entries:
            wrangler.add_entry(entry, verbose=verbose)
    else:
        nprocs = ncpu
        batch_size = 'auto'
        start = time()
        with Parallel(n_jobs=nprocs, batch_size=batch_size, verbose=True) as parallel:
            par_entries = parallel(delayed(wrangler.process_entry)(
                entry, verbose=False) for entry in entries
            )
        #unpack the items and remove Nones from structure that failed to match 
        par_entries = [entry for entry in par_entries if entry is not None]
        wrangler.append_entries(par_entries)
        end = time()

    if verbose:
        print(f'\nTotal structures that match {wrangler.num_structures}/{len(entries)}')

    return wrangler

