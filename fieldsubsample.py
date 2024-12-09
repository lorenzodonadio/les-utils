#!/usr/bin/env python
# file: fieldsubsample.py

import numpy as np
from netCDF4 import Dataset
import os
import argparse


def subsample_netcdf(
    input_file,
    output_dir="./",
    skip_first=120,
    sampling_rates=[5, 10, 30, 60],
    batch_size=200,
):
    """
    Subsample a NetCDF file in time, skipping the initial non-physical observations, using batch processing.

    Parameters:
        input_file (str): Path to the input NetCDF file.
        output_dir (str): Directory to save the subsampled NetCDF files.
        skip_first (int): Number of initial time steps to skip.
        sampling_rates (list): List of sampling rates in seconds.
        batch_size (int): Number of time steps to process in each batch.
    """
    # Open the input NetCDF file
    with Dataset(input_file, "r") as src:
        # Read time variable
        time = src.variables["time"][:]

        # Skip initial non-physical observations
        time = time[skip_first:] - time[skip_first]
        n_time_steps = len(time)

        # Iterate over the desired sampling rates
        for rate in sampling_rates:
            # Determine the indices to keep for this sampling rate
            indices = np.arange(0, n_time_steps, rate)
            subsampled_time = time[indices]

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
                for name, variable in src.variables.items():
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
                        # Subsample variables that depend on time using batch processing
                        new_dims = tuple(
                            dim if dim != "time" else "time"
                            for dim in variable.dimensions
                        )
                        new_var = dst.createVariable(name, variable.datatype, new_dims)

                        # Process data in batches
                        for start in range(0, len(indices), batch_size):
                            end = min(start + batch_size, len(indices))
                            batch_indices = indices[start:end]
                            new_var[start:end] = variable[batch_indices]
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
        "--batch_size",
        type=int,
        default=200,
        help="Number of time steps to process in each batch. Default is 200.",
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
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
