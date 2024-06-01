from pymatgen.core import Lattice, Structure 
import numpy as np 
from pymatgen.core.composition import Element, Composition
from pymatgen.core.periodic_table import Specie
import math
import random
from pymatgen.io.vasp.inputs import Poscar, Kpoints, Potcar,Incar
from pymatgen.io.vasp.outputs import Outcar 
from pymatgen.entries.computed_entries import ComputedStructureEntry
import os
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
import json


def make_vasp_job(supercell, job_path, kpoints_params, vol_relax = False, incar_params=None):
    """
    Create a VASP job for the given supercell.

    Parameters:
        supercell (pymatgen.core.structure.Structure): The supercell for which to create the VASP job.
        job_path (str): The path to the directory where the VASP job should be created.
        incar_params (dict): A dictionary containing the INCAR parameters.
        kpoints_params (list): A tuple containing the kpoints parameters i.e (3,3,3)
        potcar_path (str): The path to the directory containing the POTCAR files.

    Returns:
        None

    """
    # make the POSCAR 
    poscar = Poscar(structure = supercell, comment = str(supercell.composition))
    
    # make the KPOINTS
    kpoints = Kpoints.gamma_automatic(kpts = kpoints_params)
    
    # make the POTCAR
    #potcar = Potcar(symbols=[str(e.symbol)+"_pv" for e in supercell.composition.elements], functional="PBE_64")

    symbols = []
    for e in supercell.composition.elements:
        try:
            symbols.append(str(e.symbol) + "_pv")
            potcar = Potcar(symbols=symbols, functional="PBE_64")
        except:
            symbols[-1] = str(e.symbol) + "_sv"
            potcar = Potcar(symbols=symbols, functional="PBE_64")
    
    potcar = Potcar(symbols=symbols, functional="PBE_64")

    if vol_relax:
        isif = 3
    else:
        isif = 2

    if incar_params is None:
        incar_params = {'ALGO':'Normal', # from FAST
        'ENCUT': 360,
        'EDIFF': 1E-6,
        'EDIFFG': -0.01,
        'ISMEAR': 1,
        'SIGMA': 0.2,
        'LWAVE': '.FALSE.',
        'LCHARG': '.FALSE.',
        'LORBIT': 11,
        'LREAL': '.FALSE.',
        'NELM' : 100,
        'NELMIN' : 5,
        'NSW' : 200,
        'PREC' : 'Accurate',
        'IBRION' : 2,
        #'SMASS' : 1.85,
        'POTIM' : 0.2, # from 0.25
        'ISIF' : isif,
        #'NBANDS' : supercell.num_sites*2*5, # assuming 5 electrons per atom
        } 
    else:
        incar_params['ISIF'] = isif
        
    incar = Incar(incar_params)

    # make slurm file 
    
    # write the files to the job_path
    poscar.write_file(filename=os.path.join(job_path, "POSCAR"))
    kpoints.write_file(filename=os.path.join(job_path, "KPOINTS"))
    potcar.write_file(filename=os.path.join(job_path, "POTCAR"))
    incar.write_file(filename=os.path.join(job_path, "INCAR"))



def make_slurm_file(job_path, i, time="24:00:00", nodes=1, num_gpus=1,omp_threads=1):
    """
    Create a SLURM job for the given VASP job.

    Parameters:
        job_path (str): The path to the directory containing the VASP job.
        job_name (str): The name of the job.
        time (str): The time the job should run for.
        nodes (int): The number of nodes the job should run on.
        ppn (int): The number of processors per node.
        queue (str): The queue to which the job should be submitted.

    Returns:
        None

    """
    slurm_file = open(os.path.join(job_path, "job.slurm"), "w")
    slurm_file.write("#!/bin/bash\n")
    slurm_file.write(f"#SBATCH --job-name=job_{i}\n")
    slurm_file.write("#SBATCH --output=std-out\n")
    slurm_file.write("#SBATCH --time=" + time + "\n")
    slurm_file.write("#SBATCH --nodes=" + str(nodes) + "\n")
    slurm_file.write(f"#SBATCH --ntasks-per-node={num_gpus}\n")
    slurm_file.write("#SBATCH --constraint=v100s\n")
    slurm_file.write(f"#SBATCH --gres=gpu:{num_gpus}\n")
    #slurm_file.write("#SBATCH --partition=" + queue + "\n")
    #slurm_file.write("module load vasp/5.4.4\n")
    if omp_threads > 1:
        slurm_file.write(f"export OMP_NUM_THREADS={omp_threads}\n")
    #slurm_file.write("export ")
    slurm_file.write("cd $SLURM_SUBMIT_DIR\n")
    slurm_file.write("/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/mpi/bin/mpirun /home/myless/VASP/vasp.6.3.2/bin/vasp_std\n")
    slurm_file.write("exit\n")
    #slurm_file.write("mpirun vasp_std\n")
    slurm_file.close()
    
    #!/bin/bash # #SBATCH --job-name=neb_Cr9Ti10V44_Structure_Num_0_Vac_site_num_0_110 #SBATCH --output=std-out #SBATCH --ntasks-per-node=3 #SBATCH --nodes=1 #SBATCH --gres=gpu:3 #SBATCH --constraint=v100s #SBATCH --time=24:00:00 #SBATCH -p regular


def create_computed_entry_from_outcar(outcar,poscar=None):
    """
    Create a ComputedEntry from the given OUTCAR file."""
    # need to test this with an outcar 
    outcar = Outcar(outcar)
    energy = outcar.final_energy
    composition = outcar.structure.composition
    if Poscar is not None:
        poscar = Poscar.from_file(poscar)
        structure = poscar.structure
    else:
        structure = outcar.structure

    return ComputedStructureEntry(composition, energy, structure = structure)

def create_computed_entry_from_chgnet(structure, energy):
    composition = structure.composition
    return ComputedStructureEntry(composition=composition, energy = energy, structure=structure)

