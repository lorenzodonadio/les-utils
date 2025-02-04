#!/bin/sh

#SBATCH --job-name=merge_netcdf
#SBATCH --partition=compute-p2
#SBATCH --account=innovation
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3/4.12.0
conda activate base
pip install -r requirements.txt

NC_FILE="/scratch/ldonadio/dales-runs/must_test2/completefielddump.nc"
OUTPUT_DIR="/scratch/ldonadio/dales-runs/must_test2/"
SAMPLING_RATES="1,5,10,30,60"

python fieldsubsample.py ${NC_FILE} --output_dir ${OUTPUT_DIR} --sampling_rates ${SAMPLING_RATES} --skip_first 432 --nbatches 10