#!/bin/sh

#SBATCH --job-name=merge_netcdf
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3/4.12.0
conda activate base
pip install -r requirements.txt

FIELD_DUMP_NC="/scratch/ldonadio/dales-runs/must_test2/fielddump_10s.nc"
OUTPUT_DIR="/scratch/ldonadio/adepostles-runs/must2/tracers/"

python must_tracergen.py ${FIELD_DUMP_NC} --output_dir ${OUTPUT_DIR}