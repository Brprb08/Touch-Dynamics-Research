#!/bin/bash

SBATCH --partition=week
#Partition to submit to

SBATCH --time=0-04:00:00
#Time limit for this job (DD-HH-MM-SS)

SBATCH --nodes=1
#Nodes to be used for this job during runtime

SBATCH --ntasks-per-node=1
#Number of CPU's Per node

SBATCH --mem=150
#Total memory required for this job (1G = 1 Gigabyte)

SBATCH --job-name="Touch Dynamics Research"
#Name of this job in work queue

SBATCH --output=ssample.out
#Output file name

SBATCH --output=ssample.err

SBATCH --mail-user=miraosa3300@uwec.edu

SBATCH --mail-type=ALL

#SBATCH --gpus=#                        # How many GPU cards do you need? (Only needed for GPU-based jobs on BOSE)
