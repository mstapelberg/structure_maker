#!/bin/bash
set -e

svdir=$1
mkdir -p $svdir
mv boss.out $svdir
mv boss.rst $svdir
mv movie.xyz $svdir
mv amber.out $svdir

