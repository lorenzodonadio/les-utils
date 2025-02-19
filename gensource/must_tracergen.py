import argparse
import os
import re
from glob import glob
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from sourcegen import GenTracerProfile, step_source_from_mat

tracers_per_core = 3
nproc = 2
nruns = 11

# TODO have this also in point sources, not needed for now all same height is fine
# not recomended z_idx_src == 1 because of BC
z_idx_src = 2

# source profile [ [start, end, value], ]
# the values are added together if the start/end overlap between entries
step_src_mat = [[0, 2, 0.3], [2, 5, 0.7], [5, 500, 1]]


# TODO find a better way to do import this, but it should be a n x 2 array
point_sources = np.array(
    [
        [8, 30],
        [8, 45],
        [8, 60],
        [8, 75],
        [8, 90],
        [8, 105],
        [8, 120],
        [8, 135],
        [8, 150],
        [8, 165],
        [8, 180],
        [8, 195],
        [8, 210],
        [8, 225],
        [16, 30],
        [16, 45],
        [16, 60],
        [16, 75],
        [16, 90],
        [16, 105],
        [16, 120],
        [16, 135],
        [16, 150],
        [16, 165],
        [16, 180],
        [16, 195],
        [16, 210],
        [16, 225],
        [24, 30],
        [24, 45],
        [24, 60],
        [24, 75],
        [24, 90],
        [24, 105],
        [24, 120],
        [24, 135],
        [24, 150],
        [24, 165],
        [24, 180],
        [24, 195],
        [24, 210],
        [24, 225],
        [32, 30],
        [32, 45],
        [32, 60],
        [32, 75],
        [32, 90],
        [32, 105],
        [32, 120],
        [32, 135],
        [32, 150],
        [32, 165],
        [32, 180],
        [32, 195],
        [32, 210],
        [32, 225],
    ]
)


def main():
    """
    Main entry point for the CLI. Sampling rates are in seconds and must be integers.
    """

    assert tracers_per_core * nruns * nproc == len(point_sources)

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

    xarr = point_sources[:, 0].reshape(nruns, nproc, tracers_per_core)
    yarr = point_sources[:, 1].reshape(nruns, nproc, tracers_per_core)

    with Dataset(args.filedump_path, "r") as dataset:
        time_seconds = dataset.variables["time"][:]

    step_source = step_source_from_mat(time_seconds, step_src_mat)

    for r in range(nruns):
        for n in range(nproc):
            fname = f"{args.output_dir}tracer_inp_{r+1:03}_{n+1:03}.nc"
            tp = GenTracerProfile(file_path=fname, fieldump_path=args.filedump_path)
            print(xarr[r, n, :], yarr[r, n, :])
            for nsv in range(tracers_per_core):
                # print(xarr[r,n,nsv],yarr[r,n,nsv])
                x, y = xarr[r, n, nsv], yarr[r, n, nsv]
                tracer_name = f"s{nsv}"
                tp.add_tracer(tracer_name, f"s{nsv}_x_{x:3}_y_{y:3}".replace(' ',''), "kg/kg")
                tp.add_point_source(tracer_name, x, y, 2, step_source)
            tp.close()


if __name__ == "__main__":
    main()
