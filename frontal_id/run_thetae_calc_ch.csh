#!/bin/bash

### Project name
#PBS -A P54048000

### Job name
#PBS -N python	

### Wallclock time
#PBS -l walltime=12:00:00

### Queue
#PBS -q regular

### Merge output and error files
#PBS -j oe                    

### Select 2 nodes with 36 CPUs, for 72 MPI processes 
#PBS -l select=3:ncpus=36:mpiprocs=36:mem=109GB  

python era5_thetae_calc.py

