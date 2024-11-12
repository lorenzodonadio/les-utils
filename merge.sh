#!/bin/sh

#SBATCH --job-name=merge_netcdf
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3/4.12.0
conda activate base
pip install -r requirements.txt

INPUT_DIR="/scratch/ldonadio/dales-runs/utrecht_ideal_save/"
PROFILE_FILE=/scratch/ldonadio/dales-runs/utrecht_ideal_save/profiles_lite.001.nc
OUTPUT_FILE=/scratch/ldonadio/dales-runs/utrecht_ideal_save/completefielddump.nc

python fieldmerge.py merge --input_dir ${INPUT_DIR} --profile_file ${PROFILE_FILE} --output_file ${OUTPUT_FILE} 