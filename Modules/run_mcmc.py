import numpy as np
import json
from pymatgen.core.structure import Structure
from smol.io import load_work, save_work
import os
import matplotlib.pyplot as plt

from smol.moca import Ensemble
from smol.moca import Sampler
from smol.capp.generate import generate_random_ordered_occupancy

# first define the supercell matrix and the processor
# this requires an expansion input 

def create_ensemble(expansion, sc_matrix, num_threads):
    # Create the ensemble
    # This specifies the size of the MC simulation domain.
    # this gives a 64 site unit cell
    sc_matrix = np.array([
        [16, 0, 0],
        [0, 16, 0],
        [0, 0, 16]
    ])

# this convenience method will take care of creating the appropriate
# processor for the given cluster expansion.
    os.environ['OMP_NUM_THREADS'] = str(num_threads) 
    ensemble = Ensemble.from_cluster_expansion(expansion, sc_matrix)

# In a real scenario you may want a much larger processor.size
# An MC step is O(1) with the processor.size, meaning it runs at
# the same speed regardless of the size. However, larger sizes
# will need many more steps to reach equilibrium in an MC simulation.
    print(f'The supercell size for the processor is {ensemble.processor.size} prims.')
    print(f'The ensemble has a total of {ensemble.num_sites} sites.')
    print(f'The active sublattices are:')
    for sublattice in ensemble.sublattices:
        print(sublattice)


# define the mcmc sampler
# This will take care of setting the defaults
# for the supplied canonical ensemble
# here we also set the temperature to our operating temperature, in V-Cr-Ti this should be around 900K 

def create_sample(ensemble, temperature):
    sampler = Sampler.from_ensemble(ensemble, temperature)
    print(f"Sampling information: {sampler.samples.metadata}")
    return sampler

T_sample = 973.15
sampler = Sampler.from_ensemble(ensemble, temperature=T_sample)
print(f"Sampling information: {sampler.samples.metadata}")

# define the composition for GC MCMC 
compositions = [sublattice.composition for sublattice in ensemble.sublattices]
# if verbose = True
print(compositions)


# use the generate_random_ordered_occupancy function to generate an initial occupancy for the disordered system

print(dir(sublattice))
compositions = [sublattice.composition for sublattice in ensemble.sublattices]
init_occu = generate_random_ordered_occupancy(processor= ensemble.processor,
                                              composition=compositions,
                                              tol = 0.5,
                                              rng=42)

print(f"The disordered structure has composition: {ensemble.processor.structure.composition}")
print(f"The initial occupancy has composition: {ensemble.processor.structure_from_occupancy(init_occu).composition}")

# more verbose debugging stuf
# The occupancy strings created by the processor
# are by default "encoded" by the indices of the species
# for each given site. You can always see the actual
# species in the occupancy string by decoding it.
print(f'The encoded occupancy is:\n{init_occu}')
print(f'The initial occupancy is:\n {ensemble.processor.decode_occupancy(init_occu)}')


# run the mc mc 
def run_mcmc(sampler, init_occu, iterations):
    sampler.run(
        iterations,
        initial_occupancies=init_occu,
        thin_by=100,
        progress=True
    )
    samples = sampler.samples
    print(f'Fraction of successful steps (efficiency) {sampler.efficiency()}')
    print(f'The last step energy is {samples.get_energies()[-1]} eV')
    print(f'The minimum energy in trajectory is {samples.get_minimum_energy()} eV')
    return samples
# run 1M iterations
# since this is the first run, the initial occupancy must be supplied
sampler.run(
    1000000,
    initial_occupancies=init_occu,
    thin_by=100, # thin_by will save every 100th sample only
    progress=True
) # progress will show progress bar

# Samples are saved in a sample container
samples = sampler.samples

print(f'Fraction of successful steps (efficiency) {sampler.efficiency()}')
print(f'The last step energy is {samples.get_energies()[-1]} eV')
print(f'The minimum energy in trajectory is {samples.get_minimum_energy()} eV')

# You can get the minimum energy structure and current structure
# by using the ensemble processor
curr_s = ensemble.processor.structure_from_occupancy(samples.get_occupancies()[-1])
min_s = ensemble.processor.structure_from_occupancy(samples.get_minimum_energy_occupancy())

#from smol.moca.analysis.convergence import check_property_converged, determine_discard_number

energies = samples.get_energies()
# 100 as an initial guess for amount to discard
#opt_discard = determine_discard_number(property_array=energies, init_discard=100, verbose=True)
#converged = check_property_converged(energies[opt_discard:])
#print(f'Is the energy converged after discarding the first {opt_discard} samples?', converged)
print(energies)

# let's plot the energy trajectories over time 
# convergence plot
runs = np.arange(len(energies))
plt.plot(runs, energies)
plt.show()

# sample results
# Set 100 samples for burn-in, as determined in 5)
discard = 100 # this is in terms of samples so it would be discard*thin_by steps
print(f'A total of {len(samples)} samples taken.')
print(f'A total of {len(samples.get_energies(discard=discard))} samples used for production.')
print(f'The average energy is {samples.mean_energy(discard=discard)} eV')
print(f'The energy variance is {samples.energy_variance(discard=discard)} eV^2')
print(f'The sampling efficiency (acceptance rate) is approximately {samples.sampling_efficiency(discard=discard)}')

# output saving as structure
#write these to cif files
from pymatgen.io.cif import CifWriter
structure_path = '/home/myless/Packages/structure_maker/Visualization/Structures'

initial_structure = samples.get_sampled_structures(indices=[0])[0]
print(initial_structure)

# write the initial structure to a CIF file
cif = CifWriter(initial_structure)
total_initial_path = os.path.join(structure_path, 'big_big_initial_structure.cif')
cif.write_file(total_initial_path)

# print the last structure 
final_structure = samples.get_sampled_structures(indices=[-1])[0]
print(final_structure)

# write the final structure to a CIF file
cif = CifWriter(final_structure)
total_final_path = os.path.join(structure_path, 'big_big_final_structure.cif')
cif.write_file(total_final_path)
