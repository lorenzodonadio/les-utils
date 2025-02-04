#!/usr/bin/env python3
# file: fieldmerge.py

import argparse
import os
import re
from glob import glob
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm


class PostProcessField:
    """
    Class to manage and process NetCDF files in a specified directory. It
    initializes a 2D matrix to organize files based on their processing tiles
    and provides methods to extract and merge data from these files.
    """

    def __init__(self, file_path="."):
        """
        Initializes the PostProcessField with a directory path to search for
        NetCDF files and organize them based on a naming pattern.

        Args:
            file_path (str): Directory path where NetCDF files are located.
        """
        p = os.path.join(file_path, "fielddump.*.*.*.nc")
        field_files = sorted(glob(p))
        extract_procs = lambda f: re.search(r"fielddump\.(\d+)\.(\d+)\.\d+\.nc$", f)
        last_file = field_files[-1]
        match_last = extract_procs(last_file)

        if not match_last:
            raise ValueError("Could not find correct nprocx and nprocy")

        self.nprocx_max = int(match_last.group(1))
        self.nprocy_max = int(match_last.group(2))

        self.file_tiles = [
            [None for _ in range(self.nprocy_max + 1)]
            for _ in range(self.nprocx_max + 1)
        ]

        for file_path in field_files:
            match = extract_procs(file_path)
            if match:
                nprocx = int(match.group(1))
                nprocy = int(match.group(2))
                self.file_tiles[nprocx][nprocy] = file_path

        self.dims = {}
        with Dataset(self.file_tiles[0][0], mode="r") as dataset:
            self.dims["time"] = dataset.variables["time"][:]
            self.dims["zt"] = dataset.variables["zt"][:]
            self.dims["zm"] = dataset.variables["zm"][:]

        xt, xm, yt, ym = [], [], [], []
        for i in range(self.nprocx_max + 1):
            with Dataset(self.file_tiles[i][0], mode="r") as dataset:
                xt.append(dataset.variables["xt"][:])
                xm.append(dataset.variables["xm"][:])
        for j in range(self.nprocy_max + 1):
            with Dataset(self.file_tiles[0][j], mode="r") as dataset:
                yt.append(dataset.variables["yt"][:])
                ym.append(dataset.variables["ym"][:])

        self.dims["xt"] = np.concatenate(xt)
        self.dims["xm"] = np.concatenate(xm)
        self.dims["yt"] = np.concatenate(yt)
        self.dims["ym"] = np.concatenate(ym)

    def extract_field(self, var_name: str, time_start: int, time_end: int):
        """
        Extracts a time range of a specified variable from all tiles.

        Args:
            var_name (str): Name of the variable to extract.
            time_start (int): Start index for the time dimension.
            time_end (int): End index for the time dimension.

        Returns:
            np.ndarray: Extracted data matrix.
        """
        data_mat = []
        for row in self.file_tiles:
            data_row = []
            for file in row:
                with Dataset(file, mode="r") as dataset:
                    field = dataset.variables[var_name][time_start:time_end, :, :, :]
                    data_row.append(field)
            data_mat.append(np.concatenate(data_row, axis=2))
        return np.concatenate(data_mat, axis=3)

    def save_to_single_netcdf(self, output_file: str, chunk_size: int = 10):
        """
        Saves content of NetCDF files into a single merged NetCDF file.

        Args:
            output_file (str): Path to the output file.
            chunk_size (int): Number of time steps processed at once.
        """
        with Dataset(output_file, mode="w", format="NETCDF4") as new_dataset:
            for dim_name, dim_values in self.dims.items():
                new_dataset.createDimension(dim_name, len(dim_values))
                new_dataset.createVariable(dim_name, dim_values.dtype, (dim_name,))
                new_dataset.variables[dim_name][:] = dim_values

            with Dataset(self.file_tiles[0][0], mode="r") as src_dataset:
                for var_name, var_obj in src_dataset.variables.items():
                    if var_name not in self.dims:
                        var_dims = var_obj.dimensions
                        var_dtype = var_obj.dtype
                        new_var = new_dataset.createVariable(
                            var_name, var_dtype, var_dims
                        )

                        print(f"Merging variable: {var_name}")
                        print(var_dims, var_dtype)
                        print("------------------")

                        total_time_steps = len(self.dims["time"])
                        for time_start in tqdm(range(0, total_time_steps, chunk_size)):
                            time_end = min(time_start + chunk_size, total_time_steps)
                            chunk_data = self.extract_field(
                                var_name, time_start, time_end
                            )
                            new_dataset.variables[var_name][
                                time_start:time_end, :, :, :
                            ] = chunk_data
            print("-------- Completed Merge ----------")
            print(new_dataset)
            print(f"Merged data saved to {output_file}")
            print("-----------------------------------")


def add_profiles_to_dataset(profile_path: str, dataset_path: str):
    """
    Adds profile variables from a profile file to an existing NetCDF dataset.

    Args:
        profile_path (str): Path to the profile file with additional variables.
        dataset_path (str): Path to the NetCDF file where profiles will be added.
    """
    with Dataset(profile_path, mode="r") as profiles:
        with Dataset(dataset_path, mode="r+") as dataset:
            if dataset.variables["time"].shape != profiles.variables["time"].shape:
                print("Time Dimension mismatch - Exiting without writing")
                return

            zmax = dataset.variables["zt"].size
            for var_name, var_obj in profiles.variables.items():
                if var_name in dataset.variables:
                    continue
                var_dims = var_obj.dimensions
                var_dtype = var_obj.dtype
                new_var = dataset.createVariable(var_name, var_dtype, var_dims)
                profile_data = profiles.variables[var_name][:, :zmax]
                dataset.variables[var_name][:, :] = profile_data

            print(f"Merged profiles added to {dataset_path}")


def main():
    """
    Main entry point for the CLI. Parses arguments and executes the appropriate
    function based on the chosen sub-command.
    """
    parser = argparse.ArgumentParser(description="CLI for processing NetCDF files.")
    subparsers = parser.add_subparsers(
        dest="command", help="Sub-commands for operations"
    )

    # Sub-command for merging NetCDF files
    merge_parser = subparsers.add_parser(
        "merge", help="Merge NetCDF files from a directory into a single file"
    )
    merge_parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing NetCDF files to merge, ex file: fielddump.001.001.001.nc",
    )
    merge_parser.add_argument(
        "--output_file",
        default="./completefielddump.nc",
        help="Path for the merged output NetCDF file",
    )
    merge_parser.add_argument(
        "--chunk_size", type=int, default=50, help="Chunk size for merging time steps"
    )
    merge_parser.add_argument(
        "--profile_file", help="Optional profile file to add to the merged NetCDF file"
    )

    # Sub-command for adding profiles
    add_profiles_parser = subparsers.add_parser(
        "add_profiles", help="Add profile variables to an existing NetCDF file"
    )
    add_profiles_parser.add_argument(
        "--profile_file", required=True, help="Path to the profile file with variables"
    )
    add_profiles_parser.add_argument(
        "--dataset_file", required=True, help="Path to the existing NetCDF dataset"
    )

    args = parser.parse_args()

    if args.command == "merge":
        pp_field = PostProcessField(args.input_dir)
        pp_field.save_to_single_netcdf(args.output_file, chunk_size=args.chunk_size)

        # If profile file is specified, add profiles after merging
        if args.profile_file:
            add_profiles_to_dataset(args.profile_file, args.output_file)

    elif args.command == "add_profiles":
        add_profiles_to_dataset(args.profile_file, args.dataset_file)


if __name__ == "__main__":
    main()
