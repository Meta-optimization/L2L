#!/bin/bash -x
#SBATCH --account=structuretofunction
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --mail-user=d.todt@fz-juelich.de
#SBATCH --mail-type=END
#SBATCH --partition=batch
#SBATCH --job-name=l2l

. ~/.bashrc
module load Autotools

#module load ParaStationMPI/5.9.2-1-mt

#export OMP_DISPLAY_ENV=VERBOSE  
#export OMP_DISPLAY_AFFINITY=TRUE  
#export OMP_PROC_BIND=TRUE  
#export OMP_NUM_THREADS=47

python l2l-sbi-arbor.py

