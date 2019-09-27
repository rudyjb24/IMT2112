#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=PruebaRudy
# Archivo de salida
#SBATCH --output=salida.txt
# Cola de trabajo
#SBATCH --partition=full
# Solicitud de cpus
#SBATCH --nodes=4


module load openmpi
mpic++ hello_world.cpp
mpirun a.out