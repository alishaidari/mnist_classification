#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -p dev_q
#SBATCH -A cmda3634_rjh
#SBATCH -o demo_results.out

# Go to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Load the modules we need for openMP
module load gcc

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_DYNAMIC=false

# run knearest_mnist
./knearest_mnist_demo 5 60 10 $OMP_NUM_THREADS

# The script will exit whether we give the "exit" command or not.
exit
