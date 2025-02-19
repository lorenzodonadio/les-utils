import argparse
import os
import re
from glob import glob
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from sourcegen import GenTracerProfile, step_source_from_mat


tracers_per_core = 8
nproc = 16

# TODO have this also in point sources, not needed for now all same height is fine
# not recomended z_idx_src == 1 because of BC
z_idx_src = 2

# source profile [ [start, end, value], ]
# the values are added together if the start/end overlap between entries
step_src_mat = [[0, 2, 0.3], [2, 5, 0.7], [5, 500, 1]]

def remove_gradually_near_center(arr, center=(128, 128), remove_count=17,stride= 2):
    # Compute Euclidean distances to the center
    distances = np.linalg.norm(arr - np.array(center), axis=1)

    sorted_indices = np.argsort(distances)
    candidate_indices = sorted_indices[:stride * remove_count]
    remove_indices = candidate_indices[::stride][:remove_count]
    
    return np.delete(arr, remove_indices, axis=0)
## domain specific variables
start = 7
stop = 250
space = 15 #stridekindof
sources = []
for i in range(start,stop,space):
    for j in range(start,stop,space):
        sources.append([i,j])

srcmat = np.array(sources)
# we just remove 17 values to get a total of 128 sources, nice power of 2
point_sources = remove_gradually_near_center(srcmat[::2,:],center=(127,127),remove_count=17,stride=2)


def main():
    """
    Main entry point for the CLI. Sampling rates are in seconds and must be integers.
    """

    assert tracers_per_core * nproc == len(point_sources)

    parser = argparse.ArgumentParser(description="CLI for subsampling fielddump files.")
    parser.add_argument(
        "filedump_path", type=str, help="Path to the fieldump NetCDF file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tracers/",
        help="Directory to save the subsampled NetCDF files. Default is './tracers/'.",
    )

    args = parser.parse_args()

    xarr = point_sources[:, 0].reshape(nproc, tracers_per_core)
    yarr = point_sources[:, 1].reshape(nproc, tracers_per_core)

    with Dataset(args.filedump_path, "r") as dataset:
        time_seconds = dataset.variables["time"][:]

    step_source = step_source_from_mat(time_seconds, step_src_mat)

    for n in range(nproc):
        fname = f"{args.output_dir}tracer_inp_{n+1:03}.nc"
        tp = GenTracerProfile(file_path=fname, fieldump_path=args.filedump_path)
        print('-----------new tracer file------------')
        print(srcname)
        print(xarr[n, :], yarr[n, :])
        for nsv in range(tracers_per_core):
            # print(xarr[r,n,nsv],yarr[r,n,nsv])
            x, y = xarr[n, nsv], yarr[ n, nsv]
            tracer_name = f"s{nsv}"
            srcname = f"s{nsv}_x_{x:3}_y_{y:3}".replace(' ','')
            print(srcname)
            tp.add_tracer(tracer_name, srcname, "kg/kg")
            tp.add_point_source(tracer_name, x, y, 2, step_source)
        tp.close()


if __name__ == "__main__":
    main()
