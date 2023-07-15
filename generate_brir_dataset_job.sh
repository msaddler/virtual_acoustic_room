#!/bin/bash
#
#SBATCH --job-name=generate_brir
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=2000
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node[017-094,097,098]
#SBATCH --array=0-1999
#SBATCH --partition=normal

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

echo $(hostname) $job_idx

module add openmind8/anaconda/3-2022.10
source activate tf

python -u generate_brir_dataset.py ${job_idx}
