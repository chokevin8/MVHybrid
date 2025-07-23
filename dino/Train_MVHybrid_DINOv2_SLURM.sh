#!/bin/bash 

#SBATCH --job-name dinov2_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00
#SBATCH -p amd_a100nv_8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

# Load software list
module load cuda/12.1 htop/3.0.5 nvtop/1.1.0 singularity/3.9.7
source activate mambavision

export OMP_NUM_THREADS=8
export PYTHONPATH=/absolute/path/to/dino

python dinov2/run/train/train.py --config-file /configs/train/Train_MVHybrid.yaml \
  --nodes 3 \
  --ngpus=8 \
  --partition=slurm_partition \
  --output-dir /path/to/output \
  train.dataset_path=TCGADataset:root=/path/to/dataset