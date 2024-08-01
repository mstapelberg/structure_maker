from __future__ import annotations
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar, Kpoints, Incar, Potcar
from fix_order import fix_order
import numpy as np

#from pymatgen.io.vasp import Poscar
from ase.mep import NEB
from ase.mep import DyNEB
from ase.optimize.fire import FIRE as QuasiNewton
from ase.optimize import BFGS
from ase.optimize import LBFGS
from ase.io import write,read
from ase.io.trajectory import Trajectory
import os, sys, json, re

import matgl
from matgl.ext.ase import M3GNetCalculator, Relaxer
from matgl.apps.pes import Potential
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model import StructOptimizer

# import mace 
#from mace.calculators import MACECalculator
#from MACEStructOptimizer import MACEStructOptimizer

import torch
import warnings


class NEB_Barrier:
    """
        Class to store a NEB barrier and run calculations on it using M3GNet

        Inputs:
            * start: starting structure (This assumes you have run fix_order on the structure)
            * end: ending structure (This assumes you have run fix_order on the structure)
            * energies: list of energies for each image
            * composition: composition of the structure
            * structure_number: number of the structure
            * defect_number: number of the defect
            * direction: direction of the barrier
            * root_path : path with the START, END, and energies.json

        Methods: * run_neb: runs the neb calculation using the M3GNet calculator * create_vasp_folder: creates vasp files for the neb calculation

    """
    # in the future this Barrier class is a structure, vacancy site, direction associated barrier for each composition
    # could add a property called name 
    def __init__(self, start: Structure, end: Structure, vasp_energies: list, composition: str, structure_number: int, defect_number: int, direction: str, root_path):
        self.start = start
        self.end = end
        self.vasp_energies = vasp_energies
        # write a class that stores the results instead since these are a group 
        # class --> matgl_results --> init these variables, add functions to convert, parse, etc as I need them 
        self.matgl_energies = {}  # Initialize as an empty dictionary
        self.matgl_structures = {}  # Initialize as an empty dictionary
        self.matgl_forces = {}  # Initialize as an empty dictionary
        self.matgl_stresses = {}  # Initialize as an empty dictionary
        self.composition = composition
        self.structure_number = structure_number
        self.defect_number = defect_number
        self.direction = direction
        self.traj = None
        self.num_images = None
        self.root_path = root_path
        self.neb_path = None #document this 
        self.neb = None 

    # think about pulling this out of the main run code and putting it into the results class 
    def save_neb_results(self, potential, run_relax,results_file = 'results.json'):
        """
        Saves the NEB results after a NEB run has been completed.

        Parameters:
            potential (object): The potential used for the NEB calculation.
            run_relax (bool): Flag indicating whether to run the relax function after saving the results.

        Returns:
            None
        """
        forces_dict = {}
        stress_dict = {}
        energy_dict = {}
        structure_dict = {}

        # here I want to get the forces for each image, and add them to a dictionary
        if self.neb is not None: 
            for i, image in enumerate(self.neb.images):
                try:
                    forces = image.get_forces()
                    # here we look to see if we use CHGNet or M3GNet as M3GNet does things weird
                    #if isinstance(potential,CHGNet):
                    energy = image.get_potential_energy()
                    #elif isinstance(potential,Potential): 
                        #energy = image.get_potential_energy()[0]
                    stress = image.get_stress()
                    structure = AseAtomsAdaptor.get_structure(image)
                    forces_dict[i] = forces
                    structure_dict[i] = structure
                    stress_dict[i] = stress
                    energy_dict[i] = energy
                except Exception as e:
                    print(f"Missing results for Image {i}")
                    raise e
        else:
            print("NEB object not found")

        #here save the structures, forces, energies, and stresses to the class variables
        self.matgl_structures.update(structure_dict)
        # the energy dict is not being updated correctly
        # self.matgl_energy remains an empty dictionary even after I set this equal to energy_dict
        # I want to insert the energies into the matgl_energies dictionary
        # the energy_dict is a dictionary with the energies for each image
        # the matgl_energies is currently an empty dictionary 
        # I want to insert the energies from the energy_dict into the matgl_energies dictionary
        self.matgl_energies.update(energy_dict)
        #print('\n')
        #print(self.matgl_energies)
        self.matgl_forces.update(forces_dict)
        self.matgl_stresses.update(stress_dict)
        
        # Now run the relax function if run_relax is True
        if run_relax:
            rel_structures, rel_energies, rel_forces , rel_stresses = self.relax(potential=potential)
            structure_dict[0] = rel_structures[0]
            structure_dict[self.num_images + 1] = rel_structures[1]
            energy_dict[0] = rel_energies[0]
            energy_dict[self.num_images + 1] = rel_energies[1]
            forces_dict[0] = rel_forces[0]
            forces_dict[self.num_images + 1] = rel_forces[1]
            stress_dict[0] = rel_stresses[0]
            stress_dict[self.num_images + 1] = rel_stresses[1]
            

        # now i want to dump the structures, energies, forces, and stresses to a json file
        # i want the dictionary results_dict to hold the structures, energies, forces, and stresses
        results_dict = {}
        
        def convert_floats(obj):
            """Reads through a structure object and makes sure no np.float32s exist"""
            if isinstance(obj, dict):
                return {key: convert_floats(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_floats(item) for item in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj

        # check if we are running chgnet
        if isinstance(potential,CHGNet):
            structure_dict_json = {key: convert_floats(value.as_dict()) for key, value in structure_dict.items()} 
            energy_dict_json = {key: float(value) for key, value in energy_dict.items()}
            forces_dict_json = {key: [[float(val) for val in sublist] for sublist in value.tolist()] for key, value in forces_dict.items()}
            stress_dict_json = {key: np.array(value).tolist() for key, value in stress_dict.items()}
        
        if isinstance(potential,str):
            # placeholder for mace 
            structure_dict_json = {key: convert_floats(value.as_dict()) for key, value in structure_dict.items()} 
            energy_dict_json = {key: float(value) for key, value in energy_dict.items()}
            forces_dict_json = {key: [[float(val) for val in sublist] for sublist in value.tolist()] for key, value in forces_dict.items()}
            stress_dict_json = {key: np.array(value).tolist() for key, value in stress_dict.items()}

        # for matgl only 
        elif isinstance(potential,Potential):
            structure_dict_json = {key : value.as_dict() for key, value in structure_dict.items()}
            energy_dict_json = {key: value.tolist() for key, value in energy_dict.items()}
            forces_dict_json = {key: value.tolist() for key, value in forces_dict.items()}
            stress_dict_json = {key: value.tolist() for key, value in stress_dict.items()}

        # Create the results dictionary
        results_dict = {
            'structures': structure_dict_json,
            'energies': energy_dict_json,
            'forces': forces_dict_json,
            'stress': stress_dict_json
        }
        
        #print(results_dict)

        # dumps the results into the root path of the barrier for training or creation later
        results_file_path = os.path.join(self.root_path,results_file)
        json.dump(results_dict, open(results_file_path,'w'))
    
    def relax(self, potential):
            """
            Relaxes the start and end structures using the M3GNet or CHGNet Calculator
            It should be called after neb_run has been run

            Args:
                potential (CHGNet or Potential): The potential used for relaxation

            Returns:
                int: Returns 0 after the relaxation process is complete
            """ 
            # define the relaxer object
            if isinstance(potential,CHGNet):
                #print('Using CHGNet model = {0}'.format(potential))
                relaxer = StructOptimizer(potential)
                start_relaxed = relaxer.relax(atoms = self.start, fmax=0.01,relax_cell=False,verbose=False)
                end_relaxed = relaxer.relax(atoms = self.end, fmax=0.01, relax_cell=False,verbose=False)

                # now I want to get the energy of the start and end relaxed structures 
                start_energy = float(start_relaxed['trajectory'].energies[-1])
                start_forces = start_relaxed['trajectory'].forces[-1]
                start_stresses = start_relaxed['trajectory'].stresses[-1]
                start_structure = start_relaxed['final_structure']

                end_energy = float(end_relaxed['trajectory'].energies[-1])
                end_forces = end_relaxed['trajectory'].forces[-1]
                end_stresses = end_relaxed['trajectory'].stresses[-1]
                end_structure = end_relaxed['final_structure']

            elif isinstance(potential,Potential):
                #print('Using M3GNet model = {0}'.format(potential))
                relaxer = Relaxer(potential=potential,relax_cell=False)

            #get the energy of the start
                relax_start = relaxer.relax(self.start,fmax=0.01)
                start_energy = float(relax_start['trajectory'].energies[-1])
                start_structure = relax_start['final_structure']
                start_forces = relax_start['trajectory'].forces[-1]
                start_stresses = relax_start['trajectory'].stresses[-1]

                relax_end = relaxer.relax(self.end,fmax=0.01)
                end_energy = float(relax_end['trajectory'].energies[-1])
                end_structure = relax_end['final_structure']
                end_forces = relax_end['trajectory'].forces[-1]
                end_stresses = relax_end['trajectory'].stresses[-1]
            
            """
            elif isinstance(potential,str):
                #print('Using MACE model = {0}'.format(potential))
                calculator = MACECalculator(potential,device='cpu')
                relaxer = MACEStructOptimizer(calculator)
                start_relaxed = relaxer.relax(atoms = self.start, fmax=0.01,relax_cell=False,verbose=False)
                end_relaxed = relaxer.relax(atoms = self.start, fmax=0.01, relax_cell=False,verbose=False)

                # now I want to get the energy of the start and end relaxed structures 
                start_energy = float(start_relaxed['trajectory'].energies[-1])
                start_forces = start_relaxed['trajectory'].forces[-1]
                start_stresses = start_relaxed['trajectory'].stresses[-1]
                start_structure = start_relaxed['final_structure']

                end_energy = float(end_relaxed['trajectory'].energies[-1])
                end_forces = end_relaxed['trajectory'].forces[-1]
                end_stresses = end_relaxed['trajectory'].stresses[-1]
                end_structure = end_relaxed['final_structure']
            """
            
            #Update End Points
            self.matgl_energies.update({0: start_energy})
            self.matgl_forces.update({0:start_forces})
            self.matgl_stresses.update({0:start_stresses})
           
            # the structure should actually be from the original start structure
            self.matgl_structures.update({0:start_structure})

            #get the energy of the end
            self.matgl_energies.update({self.num_images+1 : end_energy})
            self.matgl_forces.update({self.num_images + 1 : end_forces})
            self.matgl_stresses.update({self.num_images+1 : end_stresses})
            #print(self.matgl_energies)
            
            #same as beginning, I fix this in make_neb_folders though 
            self.matgl_structures.update({self.num_images + 1 : end_structure})
            
            energies = [start_energy, end_energy]
            forces = [start_forces, end_forces]
            stresses = [start_stresses, end_stresses]
            structures = [start_structure, end_structure]

            return structures, energies, forces, stresses


    def neb_run(self, num_images: int, potential, vac_potential = None , run_relax: bool = True, neb_method: str = 'DyNEB', results_file: str = 'results.json', neb_run: bool = True, fmax = 0.01, num_steps = 300):
        """
        Runs the NEB (Nudged Elastic Band) calculation.

        Args:
            num_images (int): The number of intermediate images to be created between the start and end structures.
            potential: The potential used for the calculation.
            vac_potential: The potential used for the calculation of the vacuum region (optional).
            run_relax (bool): Whether to run relaxation before the NEB calculation (default is True).
            neb_method (str): The NEB method to use ('DyNEB', 'climb', or 'aseneb') (default is 'DyNEB').
            results_file (str): The name of the file to save the NEB results (default is 'results.json').
            neb_run (bool): Whether to run the NEB calculation (default is True).
            fmax: The maximum force tolerance for the NEB calculation (default is 0.01).
            num_steps: The maximum number of steps for the NEB calculation (default is 300).
        """
        # first get the energy of the start and end structures using M3GNet

        #define the relaxer object
        self.num_images = num_images
        # now define the images
        images = [self.start]
        images += [self.start.copy() for i in range(self.num_images)]
        images += [self.end]
        
        #print(self.potential)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', device)
        # now define the calculator for each image
        if isinstance(potential,CHGNet):
            calculator = CHGNetCalculator(potential,use_device=device)
            for image in images:
                image.calc = CHGNetCalculator(potential,use_device=device)
                #image.calc = calculator
            if vac_potential is not None:
                #vac_calculator = CHGNetCalculator(vac_potential,use_device=device)
                #images[0].calc = vac_calculator
                #images[-1].calc = vac_calculator
                images[0].calc = CHGNetCalculator(vac_potential,use_device=device)
                images[-1].calc = CHGNetCalculator(vac_potential,use_device=device)
        elif isinstance(potential,Potential): 
            for image in images:
                image.calc = M3GNetCalculator(potential)
        
        #elif isinstance(potential,str):
            #for image in images:
                #image.calc = MACECalculator(potential,device=device)

        #try to run the neb calculation
        if neb_method == 'DyNEB':
            neb = DyNEB(images, dynamic_relaxation=True, scale_fmax=2.)
        elif neb_method == 'climb':
            neb = NEB(images, k=8, climb=True, method='improvedtangent',allow_shared_calculator=True)
        else:
            neb = NEB(images, method='aseneb',allow_shared_calculator=True)

        neb.interpolate(mic=True)
            # lbfgs = LBFGS(neb, trajectory=os.path.join(self.root_path,'neb.traj'))
        
        if neb_run == True:
            try:
                lbfgs = LBFGS(neb)  # here I am not making a trajectory file
                early_stopping = EarlyStoppingCallback(threshold=1e-5, patience=10)

                lbfgs.run(fmax=fmax, steps=num_steps)
            except Exception as e:
                print("Error in running neb calculation :",e)
                print("Error in Comp - {0} : Structure - {1} : Defect - {2} : Direction - {3}".format(self.composition, self.structure_number, self.defect_number, self.direction))
                raise e
        
            try:
                self.neb = neb
                if vac_potential is not None:
                    print("using vacancy potential")
                    self.save_neb_results(run_relax=run_relax, potential=vac_potential, results_file=results_file)
                else:
                    print("not using vacancy potential")
                    self.save_neb_results(run_relax=run_relax, potential=potential, results_file=results_file)
            except Exception as e:
                print("Error in saving neb_results :", e)
                raise e
        else:
            self.neb = neb 

            # write an empty_results.json file to the directory 
            empty_results = {}
            empty_results_path = os.path.join(self.root_path,'empty_results.json')
            json.dump(empty_results, open(empty_results_path,'w'))
            
            print("Skipping neb calculation")


        # add a raise statement to re-raise the error? 
        # look up raise satements in python and how they work 
        # errors in third party code could be raised as a different error or mess things up 
        # when I catch the exception but still want to save a result (dump to a log file), rather than lose all the data 

        # save the results of the neb calculations to self.matgl_energies, self.matgl_structures, self.matgl_forces, and self.matgl_stresses

    def make_neb_folders(self):
        """
        This function makes the image folders in the self.neb_path directory 
        Outputs:
            Folders with POSCAR files
        """
        for i, image in enumerate(self.neb.images):
            if i < 10: 
                folder_name = str(i).zfill(2)
            else:
                folder_name = str(i)
            folder_path = os.path.join(self.neb_path,folder_name)
            os.makedirs(folder_path,exist_ok=True)
            image.write(os.path.join(folder_path,"POSCAR"),format='vasp',direct=True)

    def make_vasp_inputs(self,incar_params = {}, climb = False, num_kpts = 3):
        """Writes the VASP INCAR, KPOINTS, and POTCAR files to the base_path directory:

            Optional Inputs:
                incar_params = dictionary of incar parameters to be added to the INCAR file
                climb = boolean to determine if the NEB calculation should use the climbing image
                num_kpts = number of kpoints to be added to the KPOINTS file

            Outputs:
                INCAR = creates an INCAR file using pymatgen
                KPOINTS = creates a KPOINTS fiel using pymatgen
                POTCAR = makes a potcar based on the structure file given 
                slurm_file = file that we can run on the cluster with :) 
        """

        #TODO change this to describe how it works, I have done this! no longer todo 
        # 1. Need to read in the first POSCAR file and extract the elements 
        # 2. Then I need to create a POTCAR from the order of the atoms in the POSCAR 
        # 3. Then need to write an INCAR file using the NEB suggestions from Ma 
        # 4. Then create a KPOINTS file - do this using pymatgen 
        # 5. Then write a slurm_file to run on the cluster 

        #1. 
        structure = Structure.from_file(os.path.join(self.neb_path,'00/POSCAR'))
        symbols = []
        for e in structure.composition.elements:
            try:
                symbols.append(str(e.symbol) + "_pv")
                potcar = Potcar(symbols=symbols, functional="PBE_64")
            except:
                symbols[-1] = str(e.symbol) + "_sv"
                potcar = Potcar(symbols=symbols, functional="PBE_64")

        potcar = Potcar(symbols = symbols, functional = "PBE_64")

        potcar.write_file(os.path.join(self.neb_path,'POTCAR'))

        # 3. Write INCAR
        #change these variables 
        encut = 360
        ediff = 1e-6
        ediffg = -0.01
        spring = -8

        # I'd like to check if a incar_params dictionary has been passed to the function
        # if it's empty, then I want to use the default parameters below 
        #create params
        if incar_params == {} and climb == True:
            incar_params = {
                    'SYSTEM': 'NEB-{0}_{1}_{2}_{3}'.format(self.composition,
                                                        self.structure_number,
                                                        self.defect_number,
                                                        self.direction), #insert composition in future
                    'ISTART' : 0,
                    'ALGO' : 'NORMAL',
                    'PREC' : 'Accurate',
                    'ENCUT' : encut,
                    'EDIFF' : ediff,
                    'EDIFFG' : ediffg,
                    'IMAGES' : int(self.num_images),
                    'SPRING' : spring,
                    'LCLIMB' : '.TRUE.',
                    'LCHARGE' : '.FALSE.',
                    'ISMEAR' : 1,
                    'SIGMA' : 0.2,
                    'NSW' : 300,
                    'NELM' : 100,
                    'NELMIN' : 5,
                    'ISIF' : 2,
                    'LREAL' : 'F',
                    'IBRION' : 3,
                    'POTIM' : 0,
                    'LASPH' : 'T',
                    'MAXMOVE' : 0.1,
                    'ILBFGSMEM' : 20,
                    'LGLOBAL' : '.TRUE.',
                    'LAUTOSCALE' : '.TRUE.',
                    'INVCURV' : 0.01,
                    'LLINEOPT' : '.FALSE.',
                    'FDSTEP' : 5E-3,
                    'ISPIN' : 1,
                    'ADDGRID' : 'T',
                    'LWAVE' : 'F',
                    'LORBIT' : 11
                    }
        elif incar_params == {} and climb == False:
            incar_params = {
                    'SYSTEM': 'NEB-{0}_{1}_{2}_{3}'.format(self.composition,
                                                        self.structure_number,
                                                        self.defect_number,
                                                        self.direction), #insert composition in future
                    'ISTART' : 0,
                    'ALGO' : 'NORMAL',
                    'PREC' : 'Accurate',
                    'ENCUT' : encut,
                    'EDIFF' : ediff,
                    'EDIFFG' : ediffg,
                    'IMAGES' : int(self.num_images),
                    'SPRING' : spring,
                    'LCHARGE' : '.FALSE.',
                    'ISMEAR' : 1,
                    'SIGMA' : 0.2,
                    'NSW' : 200,
                    'NELM' : 100,
                    'NELMIN' : 5,
                    'ISIF' : 2,
                    'LREAL' : 'F',
                    'IBRION' : 3,
                    'POTIM' : 0.25,
                    'SMASS' : 1.85,
                    'LASPH' : 'T',
                    'ISPIN' : 1,
                    'ADDGRID' : 'T',
                    'LWAVE' : 'F',
                    'LORBIT' : 11
                    }

        else:
            incar_params = incar_params

        incar = Incar.from_dict(incar_params)
        incar.write_file(os.path.join(self.neb_path,'INCAR'))

        # 4. Write KPOINTs
        kpoint = Kpoints(comment='Auto NEB')
        kp = kpoint.gamma_automatic((num_kpts,num_kpts,num_kpts))
        kp.write_file(os.path.join(self.neb_path,'KPOINTS'))
        

    def create_neb_vasp_job(self,output_directory : str = '', num_kpts : int = 3,climb : bool = False):
        """
        Creates a vasp folder for the neb calculation after neb_run has been run
        
        By default the folder is named neb_composition_structure_defect_direction
        i.e if the composition is 55V4Cr4Ti, structure number is 1, defect number is 1, and direction is 111, 
        the folder name is neb_55V4Cr4Ti_1_1_111
        
        INPUTS:
            output_directory: directory to write the vasp files to, if left blank will create the folder in the directory where the neb calculation was run

        """

        neb_folder = f'neb_{self.composition}_{self.structure_number}_{self.defect_number}_{self.direction}'
        if output_directory == '':
            self.neb_path = os.path.join(self.root_path, neb_folder)
        else:
            self.neb_path = os.path.join(output_directory, neb_folder)

        os.makedirs(self.neb_path, exist_ok=True)
        # make the POSCAR Files
        self.make_neb_folders()
        # make the VASP input files
        self.make_vasp_inputs(climb=climb,num_kpts=num_kpts)
        return 0
    
    def job_info(self):
        """ Returns a string with the job composition, structure number, defect number, and direction
        """
        return f"{self.composition}_{self.structure_number}_{self.defect_number}_{self.direction}"
    
    def job_path(self):
        """Returns the path to where the barrier jobs results are saved"""
        return self.root_path
        
class EarlyStoppingCallback:
    def __init__(self, threshold=1e-5, patience=10):
        self.threshold = threshold
        self.patience = patience
        self.counter = 0
        self.last_energy = None

    def __call__(self, neb):
        energies = neb.get_energies()
        if len(energies) > 1:
            current_energy = energies[-1]
            if self.last_energy is not None:
                energy_diff = abs(current_energy - self.last_energy)
                if energy_diff < self.threshold:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f"Stopping early: Energy improvement {energy_diff} is below the threshold {self.threshold} for {self.patience} consecutive steps")
                        return True
                else:
                    self.counter = 0
            self.last_energy = current_energy
        return False