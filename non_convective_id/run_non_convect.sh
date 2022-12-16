#!/bin/bash -l 

### Job Name
#PBS -N non_convect

### Charging account
#PBS -A P54048000

### Request one chunk of resources with 1 CPU and 10 GB of memory
#PBS -l select=1:ncpus=2:mem=300GB

### Allow job to run up to 30 minutes
#PBS -l walltime=24:00:00

### Route the job to the casper queue
#PBS -q casper

### Join output and error streams into single file
#PBS -j oe

### export TMPDIR=/glade/scratch/doughert/temp
### mkdir -p $TMPDIR

### Load Python module and activate pangeo environment
bash
source /glade/u/home/doughert/miniconda3/bin/activate pangeo3


###RUN analysis script
python non_convect_classification_py.py
