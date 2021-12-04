#!/bin/bash
#SBATCH --job-name=contrastive_training
#SBATCH --account=csci_ga_2271
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=barlow_%j.out
#SBATCH --error=barlow_%j.err

singularity exec --overlay /scratch/va2134/overlay-images/overlay-50G-10M.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c"
source /ext3/env.sh
conda activate base
echo "Here"
"


