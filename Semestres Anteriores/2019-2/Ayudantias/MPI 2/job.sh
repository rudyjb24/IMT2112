#!/bin/bash

#SBATCH --partition=full

#SBATCH --job-name=IMT2112
#SBATCH --output=log.out

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

mpic++ producto_punto.cpp
mpirun --mpi=openmpi a.out
