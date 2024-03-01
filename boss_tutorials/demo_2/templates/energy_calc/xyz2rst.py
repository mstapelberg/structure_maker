# Script for converting an xyz file into an rst file
# Usage: python3 xyz2rst.py file.xyz file.rst

import numpy as np
import sys

xyzfile = sys.argv[1]
rstfile = sys.argv[2]

file = open(xyzfile, 'r')
fh = open(rstfile, 'w')

coords = np.array(())

ind = 1
noAtoms = -1
for line in file:
	line = line.split()
	if ind == 1:
		fh.write("TST" + '\n')
		noAtoms = int(line[0])
	elif ind == 2:
		fh.write("%5i\n"%(noAtoms))
	else:
		coords = np.append(coords, float(line[1]))
		coords = np.append(coords, float(line[2]))
		coords = np.append(coords, float(line[3]))

	ind += 1

rowsToRst = int(len(coords) / 6)

#if (len(coords) / 6 % 1 > 0.001):
#	rowsToRst += 1
#extras = len(coords) - 6*(rowsToRst-1)
extras = len(coords) % 6   # ville edit 24.2.2017

ix=0
for i in range(rowsToRst):
	
	fh.write("%12.7f%12.7f%12.7f%12.7f%12.7f%12.7f\n"%(coords[ix], coords[ix + 1], coords[ix + 2], coords[ix + 3], coords[ix + 4], coords[ix + 5]))
	ix += 6
for i in range(extras):
	fh.write("%12.7f"%(coords[ix]))
	ix += 1

file.close()
fh.close()

