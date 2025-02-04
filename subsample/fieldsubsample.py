#!/usr/bin/env python
# file: fieldsubsample.py

import numpy as np
from netCDF4 import Dataset
import os
import argparse
import math
from tqdm import tqdm


def subsample_netcdf(
    input_file,
    output_dir="./",
    skip_first=0,
    sampling_rates=[5, 10, 30, 60],
    nbatches=10,
):
    """
    Subsample a NetCDF file in time, skipping the initial non-physical observations.

    Parameters:
        input_file (str): Path to the input NetCDF file.
        output_dir (str): Directory to save the subsampled NetCDF files.
        skip_first (int): Number of initial time steps to skip.
        sampling_rates (list): List of sampling rates in seconds.
    """
    assert nbatches > 0
    # Open the input NetCDF file
    with Dataset(input_file, "r") as src:
        # Read time variable
        time = src.variables["time"][:]
        print(len(time))
        print(time[0], time[-1])
        print(skip_first)

        # Iterate over the desired sampling rates
        for rate in sampling_rates:
            print("Rate: ", rate)
            # skip = skip_first-skip_first%rate

            if skip_first >= len(time):
                print(
                    f"skip first: {skip_first} cant be more than the lenght of time dimension: {len(time)} "
                )
                continue

            idx = np.arange(skip_first, len(time), rate)

            if len(idx) <= 1:
                print(
                    f"too short subsampled time, reduce the rate or skip_first, rate:{rate} skip_first:{skip_first}"
                )
                continue

            while nbatches > len(idx):
                nbatches = nbatches // 2

            subsampled_time = time[idx] - time[skip_first]
            # pointperbatch = batch_size//rate
            ppb = math.ceil(len(idx) / (nbatches))  # points per batch
            timeidx = np.arange(0, len(idx))
            idxbatch, timebatch = [], []
            for i in range(nbatches):
                if len(idx[i * ppb : (1 + i) * ppb]) > 0:
                    idxbatch.append(idx[i * ppb : (1 + i) * ppb])
                    timebatch.append(timeidx[i * ppb : (1 + i) * ppb])

            # Create output NetCDF file
            output_file = os.path.join(output_dir, f"fielddump_{rate}s.nc")
            with Dataset(output_file, "w", format="NETCDF4") as dst:
                # Copy dimensions
                for name, dimension in src.dimensions.items():
                    if name == "time":
                        dst.createDimension(name, len(subsampled_time))
                    else:
                        dst.createDimension(
                            name,
                            len(dimension) if not dimension.isunlimited() else None,
                        )

                # Copy variables
                for name, variable in tqdm(src.variables.items()):
                    if name == "time":
                        # Handle time variable separately
                        new_var = dst.createVariable(name, variable.datatype, (name,))
                        new_var[:] = subsampled_time
                        new_var.setncatts(
                            {
                                attr: variable.getncattr(attr)
                                for attr in variable.ncattrs()
                            }
                        )
                    elif "time" in variable.dimensions:
                        # Subsample variables that depend on time
                        new_dims = tuple(
                            dim if dim != "time" else "time"
                            for dim in variable.dimensions
                        )
                        new_var = dst.createVariable(name, variable.datatype, new_dims)

                        for indices, tindices in zip(idxbatch, timebatch):
                            new_var[tindices] = variable[indices]
                        # new_var[:] = variable[indices]
                        new_var.setncatts(
                            {
                                attr: variable.getncattr(attr)
                                for attr in variable.ncattrs()
                            }
                        )
                    else:
                        # Copy non-time-dependent variables directly
                        new_var = dst.createVariable(
                            name, variable.datatype, variable.dimensions
                        )
                        new_var[:] = variable[:]
                        new_var.setncatts(
                            {
                                attr: variable.getncattr(attr)
                                for attr in variable.ncattrs()
                            }
                        )

                # Copy global attributes
                dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})

                print(f"Created subsampled file: {output_file}")


def main():
    """
    Main entry point for the CLI. Sampling rates are in seconds and must be integers.
    """
    parser = argparse.ArgumentParser(description="CLI for subsampling fielddump files.")
    parser.add_argument("input_file", type=str, help="Path to the input NetCDF file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the subsampled NetCDF files. Default is './'.",
    )
    parser.add_argument(
        "--skip_first",
        type=int,
        default=120,
        help="Number of initial time steps to skip. Default is 120.",
    )
    parser.add_argument(
        "--sampling_rates",
        type=str,
        default="5,10,30,60",
        help="Comma-separated list of sampling rates in seconds. Default is '5,10,30,60'.",
    )
    parser.add_argument(
        "--nbatches",
        type=int,
        default=10,
        help="Number of batches to process the data. Default is 10.",
    )
    args = parser.parse_args()

    # Parse sampling rates into a list of integers
    try:
        sampling_rates = list(map(int, args.sampling_rates.split(",")))
    except ValueError:
        print("Error: Sampling rates must be a comma-separated list of integers.")
        exit(1)

    # Call the subsampling function
    subsample_netcdf(
        input_file=args.input_file,
        output_dir=args.output_dir,
        skip_first=args.skip_first,
        sampling_rates=sampling_rates,
        nbatches=args.nbatches,
    )


if __name__ == "__main__":
    main()
