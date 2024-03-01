#!/bin/zsh
set -e

# go to calculation directory
here=$(pwd)
cd energy_calc

# input output files
infile="variables.in"
outfile="energy.out"

# copy original structure as z-matrix and modify it based on given variables
cp alanine.gzmat acq.gzmat
d4=$(cat $infile | head -1 | tail -1)
echo "d4 = $d4" >> acq.gzmat
d5=$(($d4+120))
echo "d5 = $d5" >> acq.gzmat
d13=$(cat $infile | head -2 | tail -1)
echo "d13 = $d13" >> acq.gzmat

# run babel run convert modified z-matrix file into xyz file
babel -igzmat acq.gzmat -oxyz acq.xyz &> .dump

# save the xyz structure
cat acq.xyz >> $here/movie.xyz

# transform the xyz into rst coordinates for amber using a python script
python3 xyz2rst.py acq.xyz acq.rst

# run static amber simulation using the rst coordinate file
sander -O -i md.in -o acq.out -c acq.rst -p system.prmtop

# parse amber output file for total energy
eline=($(grep "Etot"  acq.out))
E=$(echo ${eline[3]})
echo $E > $outfile

# save amber output file
cat acq.out >> $here/amber.out

# clean
rm -rf acq*
rm -rf mdinfo
rm -rf mdfrc
rm -rf restrt

# return to original directory
cd $here

