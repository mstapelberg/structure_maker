import os

def f(x):
    "User function calling an external script"

    # write variables to file
    f = open('energy_calc/variables.in', 'w')
    for i in range(len(x[0])):
        f.write('%15.7f\n'%(x[0][i]))
    f.close()

    # call bash script for AMBER simulation
    os.system('energy_calc/run_2D.sh')

    # read energy from file and return it
    f = open('energy_calc/energy.out')
    E = float(f.readline())
    f.close()
    return E

