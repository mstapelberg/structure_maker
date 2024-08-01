import ase 
from ase.io import read
import numpy as np 
from ase.geometry import get_distances 

#Atoms_start = read('./111_paths_Aug_15/path_num_75/START',format='vasp')
#Atoms_end = read('./111_paths_Aug_15/path_num_75/END',format='vasp')

def fix_order(Atoms_start,Atoms_end):
    
    distances = []
    mapping = []
    
    for i in range(len(Atoms_end)):
        curr_distances = []
        for j in range(len(Atoms_start)):
            D, D_len = get_distances(Atoms_start.positions[j],Atoms_end.positions[i],cell=Atoms_start.cell,pbc=Atoms_start.pbc)
            curr_distances.append(D_len[0][0])
        distances.append(curr_distances)
    
    distances = np.asarray(distances)
    
    dist_to_end = distances.min(axis=0)
    dist_to_start = distances.min(axis=1)

    N,M = -1,-1
    
    for i in range(len(dist_to_end)):
        if dist_to_end[i]>1.5:
            N = i
        if dist_to_start[i]>1.5:
            M = i
            
    #print(N)
    #print(M)
        
    Initial_Moving_Atom = Atoms_start.pop(N)
    
    Final_Moving_Atom = Atoms_end.pop(M)
    
    Atoms_start.append(Initial_Moving_Atom)
    
    Atoms_end.append(Final_Moving_Atom)
    
    
    ### Check out distances
    
    out_distances = []
    for i in range(len(Atoms_end)):
        D, D_len = get_distances(Atoms_start.positions[i],Atoms_end.positions[i],cell=Atoms_start.cell,pbc=Atoms_start.pbc)
        out_distances.append(D_len[0][0])
    out_distances = []
    for i in range(len(Atoms_end)):
        D, D_len = get_distances(Atoms_start.positions[i],Atoms_end.positions[i],cell=Atoms_start.cell,pbc=Atoms_start.pbc)
        out_distances.append(D_len[0][0])

    if out_distances[-1] > 4.5 or max(out_distances[:-1])>1.0 or out_distances[-1]<1.5:
    # raise an error  
    #print(out_distances)
        return None, None
    else: 
        return Atoms_start, Atoms_end


