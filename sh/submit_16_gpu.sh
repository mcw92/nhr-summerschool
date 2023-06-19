#!/bin/bash

#SBATCH --job-name=alex16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=10:00
#SBATCH --nodes=4
#SBATCH --account=hpc-prf-nhrgs
#SBATCH --ntasks-per-node=4
#SBATCH --output=/scratch/hpc-prf-nhrgs/mweiel/res/slurm-%j.out
#SBATCH --mail-user=...  # adjust this to match your email address
#SBATCH --mail-type=ALL

module purge
module load vis/torchvision/0.13.1-foss-2022a-CUDA-11.7.0   # Load required modules.

# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12340
#export NCCL_DEBUG=INFO

# Get the first node name as master address 
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

#nvidia-smi
#echo $CUDA_VISIBLE_DEVICES

export PYDIR=/scratch/hpc-prf-nhrgs/mweiel/py
export RESDIR=/scratch/hpc-prf-nhrgs/mweiel/res/job_${SLURM_JOB_ID}
mkdir ${RESDIR}
cd ${RESDIR}

srun python -u ${PYDIR}/alex_parallel.py
mv ../slurm-${SLURM_JOBID}.out ${RESDIR}
