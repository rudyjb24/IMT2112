#!/bin/bash

#SBATCH --partition=full

#SBATCH --job-name=IMT2112
#SBATCH --output=log.out

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

mpic++ hello_world.cpp
srun --mpi=openmpi a.out
