#!/bin/bash

#SBATCH --partition=full

#SBATCH --job-name=IMT2112

#SBATCH --output=log.out

#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

mpic++ MPI_dot.cpp -std=c++11
mpirun a.out
