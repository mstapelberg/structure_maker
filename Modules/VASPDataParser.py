import json
import os, re
import shutil
import numpy as np
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.core.structure import Structure

class VASPDataParser:
    def __init__(self):
        # move this to function arguments instead, this doesn't need to be saved in a class 
        pass 

    def copy_outcar_xdatcar_files(self, data_folder, new_folder):
        """Search through a root folder and copy OUTCAR and XDATCAR files to a new folder.

        Args:
            new_folder (str): The new folder to copy the OUTCAR and XDATCAR files to.
        """
        # Check if new_folder exists. If it doesn't, create it.
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

        # Get the list of subdirectories in the root folder
        subdirectories = os.listdir(data_folder)

        # Loop through the subdirectories
        for subdirectory in subdirectories:
            # Check if the subdirectory begins with 'neb_'
            if subdirectory.startswith('neb_'):
                subdirectory_path = os.path.join(data_folder, subdirectory)
                folders = os.listdir(subdirectory_path)

                # Loop through the folders
                for folder in folders:
                    # Check if the folder is named '00', '01', '02', etc.
                    if folder.isdigit() and 0 <= int(folder) <= 10:
                        folder_path = os.path.join(subdirectory_path, folder)
                        files = os.listdir(folder_path)

                        # Loop through the files
                        for file in files:
                            if 'OUTCAR' in file or 'XDATCAR' in file:
                                # Construct the new filename and copy the file
                                file_type = 'OUTCAR' if 'OUTCAR' in file else 'XDATCAR'
                                new_filename = f'{subdirectory}_Image_{folder}_{file_type}'
                                shutil.copy2(os.path.join(folder_path, file), os.path.join(new_folder, new_filename))
    def get_neb_results(self, data_folder, parse_all=False):
        """
        Search through a root folder and get the NEB results for each subdirectory. 
        The results are saved in the subdirectory as vasp_results.json

        Parameters:
        - data_folder (str): The root folder to search for NEB results.
        - parse_all (bool): Flag indicating whether to parse all data or only the last step. Default is False.

        Returns:
        None
        """
        # Get the list of subdirectories in the root folder
        subdirectories = os.listdir(data_folder)

        # Loop through the subdirectories
        for subdirectory in subdirectories:
            error_occurred = False 
            # Check if the subdirectory begins with 'neb_'
            if subdirectory.startswith('neb_'):
                subdirectory_path = os.path.join(data_folder, subdirectory)
                folders = os.listdir(subdirectory_path)
                structures = {}
                energies = {}
                forces = {}
                stresses = {}

                # Loop through the folders
                for folder in folders:
                    # Check if the folder is named '00', '01', '02', etc.
                    if folder.isdigit() and 0 <= int(folder) <= 10:
                        folder_path = os.path.join(subdirectory_path, folder)
                        
                        # get the number of the folder 
                        
                        # now get teh xdatcar and outcar files 
                        outcar = os.path.join(folder_path,'OUTCAR')
                        xdatcar = os.path.join(folder_path,'XDATCAR')
                        
                        # see if the outcar and xdatcar files exist 
                        if not os.path.exists(outcar):
                            print(f"OUTCAR file does not exist for {subdirectory} {folder}")
                            continue
                        
                        # now get the data from the outcar and xdatcar files
                        try:
                            temp_structures = self.get_structures(filename=xdatcar)
                        except:
                            print(f"Error parsing structure data for {subdirectory} {folder}")
                            error_occurred = True
                            break

                        try:
                            temp_forces = self.get_forces(filename =outcar)
                        except:
                            print(f"Error parsing force data for {subdirectory} {folder}")
                            error_occurred = True
                            break
                       
                        try:
                            temp_energies = self.get_energy_without_entropy(filename=outcar)
                        except:
                            print(f"Error parsing energy data for {subdirectory} {folder}")
                            error_occurred = True
                            break

                        try:
                            temp_stresses = self.get_stresses(filename=outcar)
                        except:
                            print(f"Error parsing stress data for {subdirectory} {folder}")
                            error_occurred = True
                            break

                        # check that all variables are the same length (i.e same number of ionic steps)
                        if len(temp_structures) == len(temp_forces) == len(temp_energies) == len(temp_stresses) and not parse_all:
                            print('not parsing all')
                            structures[str(int(folder))] = temp_structures[-1].as_dict()
                            forces[str(int(folder))] = temp_forces[-1]
                            energies[str(int(folder))] = temp_energies[-1]
                            stresses[str(int(folder))] = temp_stresses[-1]
                            
                        elif len(temp_structures) == len(temp_forces) == len(temp_energies) == len(temp_stresses) and parse_all:
                            print('parsing all')
                            structures[str(int(folder))] = [structure.as_dict() for structure in temp_structures]
                            forces[str(int(folder))] = temp_forces
                            energies[str(int(folder))] = temp_energies
                            stresses[str(int(folder))] = temp_stresses
                        else:
                            print(f"Error: Results are not the same length in {subdirectory} {folder}")
                            print(f"Length of structures: {len(temp_structures)}")
                            print(f"Length of forces: {len(temp_forces)}")
                            print(f"Length of energies: {len(temp_energies)}")
                            print(f"Length of stresses: {len(temp_stresses)}")
                            error_occurred = True
                            break
                # continue to the next subdirectory if an error occurred
                if error_occurred:
                    print(f'Error occurred in {subdirectory}, moving to next one')
                    continue

                # now that we have the data, let's write it to a json file
                vasp_results = {}
                vasp_results['structures'] = structures
                vasp_results['forces'] = forces
                vasp_results['energy_per_atom'] = energies
                vasp_results['stresses'] = stresses
                
                # write the vasp_results to the subdirectory
                if not parse_all:
                    with open(os.path.join(subdirectory_path,'vasp_results.json'), 'w') as f:
                        json.dump(vasp_results, f)
                else:
                    with open(os.path.join(subdirectory_path,'vasp_results_all.json'), 'w') as f:
                        json.dump(vasp_results, f)
    
    def parse_directory(self, root_dir):
        """
        This method is for parsing VASP XDATCAR and OUTCAR files from a batch OLCF VASP Job. 

        Input:
            root_dir (str): The root directory of the VASP job.
        
        Output:
            data (dict): A dictionary containing the parsed data from the VASP job.
        """
        data = {}
        
        for subdir, dirs, files in os.walk(root_dir):
            if 'supercell_' in os.path.basename(subdir):
                subdir_data = {
                    'structures': [],
                    'energies': [],
                    'forces': [],
                    'stresses': []
                }
                
                xdatcar_files = sorted([f for f in files if f.startswith('XDATCAR')])
                outcar_files = sorted([f for f in files if f.startswith('OUTCAR')])
                
                for xdatcar, outcar in zip(xdatcar_files, outcar_files):
                    xdatcar_path = os.path.join(subdir, xdatcar)
                    outcar_path = os.path.join(subdir, outcar)
                    
                    try:
                        structures = self.get_structures(xdatcar_path)
                        forces = self.get_forces(outcar_path)
                        energies = self.get_energy_without_entropy(outcar_path)
                        stresses = self.get_stresses(outcar_path)
                        
                        min_length = min(len(structures), len(forces), len(energies), len(stresses))
                        
                        subdir_data['structures'].extend(structures[:min_length])
                        subdir_data['energies'].extend(energies[:min_length])
                        subdir_data['forces'].extend(forces[:min_length])
                        subdir_data['stresses'].extend(stresses[:min_length])
                    
                    except Exception as e:
                        print(f"Error processing {subdir}: {e}")
                
                data[os.path.basename(subdir)] = subdir_data
        
        return data
                
                



       

    def get_forces(self, filename="OUTCAR", lines=None, n_atoms=None):
        """
            Gets the forces for every ionic step from the OUTCAR file

            Args:
                filename (str): Filename of the OUTCAR file to parse
                lines (list/None): lines read from the file
                n_atoms (int/None): number of ions in OUTCAR

            Returns:

                numpy.ndarray: A Nx3xM array of forces in $eV / \AA$

                where N is the number of atoms and M is the number of time steps
        """
        if n_atoms is None:
            n_atoms = self.get_number_of_atoms(filename=filename, lines=lines)
        trigger_indices, lines = self._get_trigger(
            lines=lines, filename=filename, trigger="TOTAL-FORCE (eV/Angst)"
        )
        return self._get_positions_and_forces_parser(
            lines=lines,
            trigger_indices=trigger_indices,
            n_atoms=n_atoms,
            pos_flag=False,
            force_flag=True,
        )

    def get_energy_without_entropy_from_line(self, line):
        """
        Extracts the energy without entropy value from a given line.

        Args:
            line (str): The line containing the energy value.

        Returns:
            float: The energy value without entropy.

        """
        return float(self._clean_line(line.strip()).split()[3])
    

    def get_energy_without_entropy_from_line(self,line):
        return float(self._clean_line(line.strip()).split()[3])

    def get_energy_without_entropy(self, filename="OUTCAR", lines=None):
        """
        Gets the total energy for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: A 1xM array of the total energies in $eV$

            where M is the number of time steps
        """
        n_atoms = self.get_number_of_atoms(filename=filename, lines=lines)

        trigger_indices, lines = self._get_trigger(
            lines=lines,
            filename=filename,
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        )
        #return np.array(
        return [
                self.get_energy_without_entropy_from_line(lines[j + 4])/n_atoms
                for j in trigger_indices
            ]
        #)

    def get_stresses(self, filename="OUTCAR", lines=None, si_unit=False):
        """

        Get's the stresses for every ionic step from the OUTCAR file

        Args:
            filename (str): Input filename
            lines (list/None): lines read from the file
            si_unit (bool): True SI units are used

        Returns:
            numpy.ndarray: An array of stress values

        """
        trigger_indices, lines = self._get_trigger(
            lines=lines,
            filename=filename,
            trigger="FORCE on cell =-STRESS in cart. coord.  units (eV):",
        )
        stress_lst = []
        for j in trigger_indices:
            # search for '------...' delimiters of the stress table
            # setting a constant offset into `lines` does not work, because the number of stress contributions may vary
            # depending on the VASP configuration (e.g. with or without van der Waals interactions)
            jj = j
            while set(lines[jj].strip()) != {"-"}:
                jj += 1
            jj += 1
            # there's two delimiters, so search again
            while set(lines[jj].strip()) != {"-"}:
                jj += 1
            try:
                if si_unit:
                    stress = [float(l) for l in lines[jj + 1].split()[1:7]]
                else:
                    stress = [float(l) for l in lines[jj + 2].split()[2:8]]
            except ValueError:
                stress = [float("NaN")] * 6
            # VASP outputs the stresses in XX, YY, ZZ, XY, YZ, ZX order
            #                               0,  1,  2,  3,  4,  5
            stressm = np.diag(stress[:3])
            stressm[0, 1] = stressm[1, 0] = stress[3]
            stressm[1, 2] = stressm[2, 1] = stress[4]
            stressm[0, 2] = stressm[2, 0] = stress[5]
            stress_lst.append(stressm.tolist())
        #return np.array(stress_lst)
        return stress_lst

    def get_structures(self, filename="XDATCAR", lines=None):
        """Creates pymatgen structures from the XDATCAR file
    
        Args:
            filename (str): Input filename
            lines (list/None): lines read from the file
            
        Returns:
            list: A list of pymatgen structures
        """
        
        # first get the species and number of each species from the XDATCAR file
        # this is the same for every structure in the XDATCAR file

        # the species are two lines above Direct configuration=     1

        # and the number of each species is one line above Direct configuration=     1
        
        # get the species
        trigger_indices, lines = self._get_trigger(
            lines=lines,
            filename=filename,
            trigger="Direct configuration=     1",
        )
        # grab the species from two lines above the trigger
        species_line = lines[trigger_indices[0] - 2]
        species = species_line.split()
        #print(species)
        
        # grab the number of each species from one line above the trigger
        number_line = lines[trigger_indices[0] - 1]
        number = number_line.split()
        
        # now need to get the lattice vector from the XDATCAR file
        # the lattice vector starts 5 lines above the trigger
        # it is 3 lines long and is a 3 x 3 matrix 

        # get the lattice vector
        lattice_vector = []
        for i in range(3):
            lattice_vector.append(lines[trigger_indices[0] - 5 + i].split())
        lattice_vector = np.array(lattice_vector).astype(float)
        lattice_vector = lattice_vector.tolist()
        #print(lattice_vector)   
        
        # now create a list of the species 
        # i.e if species = ['H', 'O'] and number = ['1', '2'] then species_lst = ['H', 'O', 'O']
        species_lst = []
        for i in range(len(species)):
            for j in range(int(number[i])):
                species_lst.append(species[i]) 

        # now get the structures from the XDATCAR file
        # the positions of the ions are in between Direct configuration= lines 
        
        # get the new trigger

        trigger_indices, lines = self._get_trigger(
            lines=lines,
            filename=filename,
            trigger="Direct configuration=",
        )
        
        # now get the structures by looping through 
        # the trigger indices and getting the positions
        # of the ions between each trigger index
        structures = []
        for i in range(len(trigger_indices)):
            # get the positions of the ions between the trigger indices
            # the positions are below the trigger indices and go for the lengthh of the species_lst
            # i.e if species_lst = ['H', 'O', 'O'] then the positions are 3 lines below the trigger index

            # get the positions
            positions = []
            for j in range(len(species_lst)):
                positions.append(lines[trigger_indices[i] + 1 + j].split())
            # convert the positions to floats
            #print(positions)
            positions = np.array(positions).astype(float)
            positions = positions.tolist()
            # create the structure
            structure = Structure(
                lattice=lattice_vector,
                species=species_lst,
                coords=positions,
                coords_are_cartesian=False,
                to_unit_cell=False,)
            #print(structure)
            structures.append(structure)

        return structures

    def get_number_of_atoms(self, filename="OUTCAR", lines=None):
        """
        Returns the number of ions in the simulation

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            int: The number of ions in the simulation

        """
        ions_trigger = "NIONS ="
        trigger_indices, lines = self._get_trigger(
            lines=lines, filename=filename, trigger=ions_trigger
        )
        if len(trigger_indices) != 0:
            return int(lines[trigger_indices[0]].split(ions_trigger)[-1])
        else:
            raise ValueError()

    def write_data_to_dict(self, mp_id, outcar_file, xdatcar_file, sampling_scheme='gnome'):
        """
        Takes in an mp_id, outcar file, and xdatcar file and creates a dictionary with the main key being the mp_id.
        The values of the mp_id will be the ionic step numbers as strings,
        and the values of the ionic step numbers will be the structure, forces, energies, and stresses for that ionic step.

        Args:
            mp_id (str): The mp_id of the structure.
            outcar_file (str): The outcar file.
            xdatcar_file (str): The xdatcar file.
            sampling_scheme (str, optional): The sampling scheme to determine the ionic steps to include in the dictionary.
                Defaults to 'gnome' but 'all' is another option.

        Returns:
            dict: A dictionary with the main key being the mp_id.
            The values of the mp_id will be the ionic step numbers as strings,
            and the values of the ionic step numbers will be the structure, forces, energies, and stresses for that ionic step.
        """
        
        # get the structures, forces, energies, and stresses
        structures = self.get_structures(filename=xdatcar_file)
        forces = self.get_forces(outcar_file)
        energies = self.get_energy_without_entropy(outcar_file)
        stresses = self.get_stresses(outcar_file)
        
        # create the dictionary
        data_dict = {}
        data_dict[mp_id] = {}
        # if we are using the all sampling scheme, we want to get all the ionic steps
        if sampling_scheme == 'all':
            for i in range(len(structures)):
                data_dict[mp_id][str(i)] = {}
                data_dict[mp_id][str(i)]['structure'] = structures[i].as_dict()
                data_dict[mp_id][str(i)]['force'] = forces[i]
                data_dict[mp_id][str(i)]['energy_per_atom'] = energies[i]
                data_dict[mp_id][str(i)]['stress'] = stresses[i]
        
        # if we are using the gnome sampling scheme, we want to randomly select 25 ionic steps from the first 50 ionic steps
        elif sampling_scheme == 'gnome':
            total_ionic_steps = len(structures)
            if total_ionic_steps > 50:
                # randomly select 25 from the first 50 ionic steps
                indices = np.random.choice(50, 25, replace=False)
                for i in indices:
                    data_dict[mp_id][str(i)] = {}
                    data_dict[mp_id][str(i)]['structure'] = structures[i].as_dict()
                    data_dict[mp_id][str(i)]['force'] = forces[i]
                    data_dict[mp_id][str(i)]['energy_per_atom'] = energies[i]
                    data_dict[mp_id][str(i)]['stress'] = stresses[i]
                
                # randomly select 25 from the rest of the ionic steps
                indices = np.random.choice(total_ionic_steps - 50, 25, replace=False)
                for i in indices:
                    data_dict[mp_id][str(i + 50)] = {}
                    data_dict[mp_id][str(i + 50)]['structure'] = structures[i + 50].as_dict()
                    data_dict[mp_id][str(i + 50)]['force'] = forces[i + 50]
                    data_dict[mp_id][str(i + 50)]['energy_per_atom'] = energies[i + 50]
                    data_dict[mp_id][str(i + 50)]['stress'] = stresses[i + 50]
            else:
                num_samples = min(25, total_ionic_steps)
                # randomly select 25 from the ionic steps
                indices = np.random.choice(total_ionic_steps, num_samples, replace=False)
                for i in indices:
                    data_dict[mp_id][str(i)] = {}
                    data_dict[mp_id][str(i)]['structure'] = structures[i].as_dict()
                    data_dict[mp_id][str(i)]['force'] = forces[i]
                    data_dict[mp_id][str(i)]['energy_per_atom'] = energies[i]
                    data_dict[mp_id][str(i)]['stress'] = stresses[i]

        return data_dict

    def write_data_to_json(self, results_folder, json_file, sampling_scheme='gnome'):
        """This function will take in a root folder of OUTCAR and XDATCAR files and write the data to a json file
        the OUTCAR and XDATCAR files will have the same name except for the presence of OUTCAR and XDATCAR and the end of the filename
        i.e Cr10Ti9V44_Structure_Num_18_Vac_site_num_1_100_Image_01_OUTCAR and Cr10Ti9V44_Structure_Num_18_Vac_site_num_1_100_Image_01_XDATCAR
        the function will loop through the root folder and find the OUTCAR and XDATCAR files with the same name
        it will then call write_data_to_dict for each OUTCAR and XDATCAR file pair
        and write a json containing the mp_id, ionic step number, structure, forces, energies, and stresses for each ionic step
        the mp_id will be the index of the OUTCAR and XDATCAR file pair in the root folder
        i.e if there are 100 OUTCAR and XDATCAR file pairs in the root folder, the mp_id will be 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, etc all the way till 99
        additionally I want to check if the json_file we are writing to already contains data, if it does, I want to append the new data to the json file
        this will require us to get the mp_id of the last entry in the json file and start the mp_id of the new data at the mp_id of the last entry + 1
        
        Args:
            results_folder (str): The root folder of OUTCAR and XDATCAR files
            json_file (str): The json file to write the data to
            
        Returns:
            None
        """
        
        # get the list of files in the results folder
        files = os.listdir(results_folder)

        # Lists to store the paths of OUTCAR and XDATCAR pairs and unmatched files
        file_pairs = []
        unmatched_files = []

        # Filter out all OUTCAR files
        outcar_files = [f for f in files if f.endswith('OUTCAR')]

        # Loop through OUTCAR files and find XDATCAR counterparts
        for outcar in outcar_files:
            # Construct the XDATCAR filename by replacing OUTCAR with XDATCAR
            xdatcar = outcar.replace('OUTCAR', 'XDATCAR')
            # Check if the XDATCAR file exists
            if xdatcar in files:
                # If it exists, store the full paths as a tuple
                file_pairs.append((os.path.join(results_folder, outcar), os.path.join(results_folder, xdatcar)))
                #print(os.path.join(results_folder, outcar), os.path.join(results_folder, xdatcar))
            else:
                # If the XDATCAR does not exist, add the OUTCAR to unmatched files
                unmatched_files.append(os.path.join(results_folder, outcar))

        # Print the results
        print(f'Number of matching OUTCAR and XDATCAR pairs: {len(file_pairs)}')
        for base_name in unmatched_files:
            print(f'Missing pair for base name: {base_name}')

        # check if the json file exists already 
        # if it does exist, check if it is empty
        # if it does not exist, create it
        if os.path.exists(json_file):
                # check if the file is empty
            with open(json_file, 'r') as f:
                try:
                    data_dict = json.load(f)
                    if not data_dict:
                        # if the file is empty, create the dictionary
                        data_dict = {}
                        mp_id_start = 0
                    else:
                        # if the file is not empty, read the data from the json file
                        # get the mp_id of the last entry in the json file
                        # the mp_id will be the last key in the json file
                        # convert the keys to integers
                        # get the maximum key
                        mp_id_start = max([int(key) for key in data_dict.keys()]) + 1

                # if the json file is empty or there's an error opening , create the dictionary
                except json.JSONDecodeError:
                    data_dict = {}
                    mp_id_start = 0

        else:
            # if the file does not exist, create the dictionary
            data_dict = {}
            mp_id_start = 0
            
                
        # create the list of mp_ids that we will make 
        # we need to make a list of strings with the mp_ids starting from mp_id_start and ending at mp_id_start + count_outcar
        # i.e if mp_id_start = 0 and count_outcar = 100, we need to make a list of strings with the numbers 0 to 99
        # we can do this by creating a list of integers from 0 to 99 and then converting the integers to strings
        mp_ids = [str(i) for i in range(mp_id_start, mp_id_start + len(file_pairs))]

        # now that we have the mp_ids, let's loop through outcar and xdatcar file pairs and write the data to the json file
        # loop through the files in the root folder
        
        for i in range(len(file_pairs)):
            
            # get the mp_id
            mp_id = mp_ids[i]

            outcar = file_pairs[i][0]
            xdatcar = file_pairs[i][1]
            
            # get the data from the outcar and xdatcar files

            try:
                data = self.write_data_to_dict(mp_id, outcar, xdatcar, sampling_scheme=sampling_scheme)
                data_dict.update(data)
            except Exception as e:
                print(f"Error writing data {i}/{len(file_pairs)} to dict for {outcar} and {xdatcar}")
                print(e)
                continue
            #print(data)

            
            # append the data to the data_dict


        # write the data_dict to the json file
        # need to check if the data file exist already 
        # if it does exist, check if it is empty
        # if it does not exist, create it
        if os.path.exists(json_file):
            # try to open the file
            try:
                with open(json_file,'r') as f:
                    # load in the existing data from the json file
                    data_dict_old = json.load(f)
                    if not data_dict_old:
                        # if the file is empty, write the data_dict to the json file
                        with open(json_file, 'w') as f:
                            json.dump(data_dict, f)
                    else:
                        # if the file is not empty, update the data_dict_old with the new data
                        data_dict_old.update(data_dict)
                        # write the new json file with the updated data_dict_old
                        with open(json_file, 'w') as f:
                            json.dump(data_dict_old, f)

            # if there is an error reading it for some reason, just overwrite it with our current data
            except json.JSONDecodeError:
                # if the file is empty, write the data_dict to the json file
                print(f"Error reading json file : {json_file}, overwriting with current data")
                with open(json_file, 'w') as f:
                    json.dump(data_dict, f)
        else:
            # if the file does not exist, write the data_dict to the json file
            with open(json_file, 'w') as f:
                json.dump(data_dict, f)
                    
        return None

       
       

    def _get_positions_and_forces_parser(self, lines, trigger_indices, n_atoms, pos_flag=True, force_flag=True):
        """
        Parser to get the forces and or positions for every ionic step from the OUTCAR file

        Args:
            lines (list): lines read from the file
            trigger_indices (list): list of line indices where the trigger was found.
            n_atoms (int): number of atoms
            pos_flag (bool): parse position
            force_flag (bool): parse forces

        Returns:
            [positions, forces] (sequence)
            numpy.ndarray: A Nx3xM array of positions in $\AA$
            numpy.ndarray: A Nx3xM array of forces in $eV / \AA$

            where N is the number of atoms and M is the number of time steps

        """
        positions = []
        forces = []
        for j in trigger_indices:
            pos = []
            force = []
            for line in lines[j + 2 : j + n_atoms + 2]:
                line = line.strip()
                line = self._clean_line(line)
                if pos_flag:
                    pos.append([float(l) for l in line.split()[0:3]])
                if force_flag:
                    force.append([float(l) for l in line.split()[3:]])
            forces.append(force)
            positions.append(pos)
        if pos_flag and force_flag:
            return np.array(positions), np.array(forces)
        elif pos_flag:
            return np.array(positions)
        elif force_flag:
            #print(len(forces))
            #return np.array(forces)
            return forces


    def _clean_line(self, line):
        """
        Cleans a line by replacing all occurrences of "-" with " -".

        Parameters:
        line (str): The line to be cleaned.

        Returns:
        str: The cleaned line.
        """
        return line.replace("-", " -")
    
    def _get_lines_from_file(self, filename, lines=None):
        """
        Reads the contents of a file and returns the lines as a list.

        Args:
            filename (str): The path to the file.
            lines (list, optional): The list of lines to be returned. If not provided, the lines will be read from the file.

        Returns:
            list: The lines of the file as a list.
        """
        if lines is None:
            with open(filename, "r") as f:
                lines = f.readlines()
        return lines

    def _get_trigger(self, trigger, filename=None, lines=None, return_lines=True):
        """
        Get the indices of lines in a file that contain a specific trigger.

        Parameters:
        trigger (str): The trigger to search for in the lines.
        filename (str, optional): The path to the file. If not provided, the lines parameter must be provided.
        lines (list, optional): The list of lines to search in. If not provided, the filename parameter must be provided.
        return_lines (bool, optional): If True, return both the trigger indices and the lines. If False, return only the trigger indices.

        Returns:
        trigger_indices (list): The indices of lines that contain the trigger.
        lines (list, optional): The list of lines from the file, if return_lines is True. None otherwise.
        """
        lines = self._get_lines_from_file(filename=filename, lines=lines)
        trigger_indices = [i for i, line in enumerate(lines) if trigger in line.strip()]
        if return_lines:
            return trigger_indices, lines
        else:
            return trigger_indices

