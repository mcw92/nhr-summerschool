#!/bin/bash

#SBATCH --job-name=alex1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --time=4:00:00 # wall-clock time limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=hpc-prf-nhrgs
#SBATCH --output=/scratch/hpc-prf-nhrgs/mweiel/res/slurm-%j.out
#SBATCH --mail-user=marie.weiel@kit.edu  # adjust this to match your email address
#SBATCH --mail-type=ALL

module purge
module load vis/torchvision/0.13.1-foss-2022a-CUDA-11.7.0   # Load required modules.

#nvidia-smi
#echo $CUDA_VISIBLE_DEVICES

export PYDIR=/scratch/hpc-prf-nhrgs/mweiel/py
export RESDIR=/scratch/hpc-prf-nhrgs/mweiel/res/job_${SLURM_JOB_ID}
mkdir ${RESDIR}
cd ${RESDIR}

python -u ${PYDIR}/alex.py
mv ../slurm-${SLURM_JOBID}.out ${RESDIR}
