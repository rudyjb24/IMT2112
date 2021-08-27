#!/bin/bash

#SBATCH --partition=full

#SBATCH --job-name=IMT2112
#SBATCH --output=log.out

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

mpic++ hello_world.cpp
mpirun --mpi=openmpi a.out
