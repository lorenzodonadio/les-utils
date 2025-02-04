import numpy as np
import pandas as pd
from os import path
from datetime import datetime
from copy import deepcopy


def df2lines(
    df: pd.DataFrame, low_sci=1e-3, high_sci=1e3, format_width=12, format_precision=5
):
    """
    Convert a DataFrame to lines of formatted strings.

    columns of only zeroes will not be converted to scientific notation

    Parameters:
    - df: pd.DataFrame - Input DataFrame
    - low_sci: tuple - Low treshold to write in scientific notation
    - high_sci: tuple - High treshold to write in scientific notation
    - format_width: int - Total width for each formatted value
    - format_precision: int - Precision of the formatted value

    Returns:
    - list of strings: Formatted rows as strings
    """
    dfabs = df.abs()
    colmax = dfabs.max()
    colmin = dfabs[dfabs > 0].min()  # Exclude zeros for minimum computation

    # Determine if each column requires scientific notation
    is_sci = {
        col: (colmax[col] > high_sci or (col in colmin and colmin[col] < low_sci))
        for col in df.columns
    }

    # Define formatting strings
    sci_format = f"{{:>{format_width}.{format_precision}e}}"
    float_format = f"{{:>{format_width}.{format_precision}f}}"

    # Generate rows
    lines = []
    for _, row in df.iterrows():
        formatted_row = ""
        for col, value in row.items():
            if is_sci[col]:
                formatted_row += sci_format.format(value)
            else:
                formatted_row += float_format.format(value)
        lines.append(formatted_row + "\n")

    return lines


def log_prof(z: np.array, z0=0.1, d=0, ustar=1):
    """
    Compute logarithmic profile for a given height array: f(z) = (ustar/k) * ln((z-d)/z0).

    Parameters:
        z (np.array): Array of heights.
        z0 (float): Roughness length. Default is 0.1.
        d (float): Displacement height. Default is 0.
        ustar (float): Friction velocity. Default is 1.

    Returns:
        np.array: Logarithmic profile values: .

    Raises:
        ValueError: If displacement height is greater than the first height in z.
    """
    if d >= z[0]:
        raise ValueError(
            "Negative or zero (z-d) not allowed in ln((z-d)/z0) - reduce d to less than z[0]"
        )
    if z0 <= 0:
        raise ValueError("z0 must be greater than 0")
    return (ustar / 0.41) * np.log((z - d) / z0)


def isonow():
    return datetime.isoformat(datetime.now(), sep="_", timespec="minutes")


def stretch_prof(z: np.array, y0=0, dy0=1, gf=1.08, zstretch=10e8):
    """
    Generate a linear-stretch profile based on height values with optional stretching.

    Parameters:
        z (np.array): Array of height values representing the vertical grid.
        y0 (float, optional): Base value for the profile. Default is 0.
        dy0 (float, optional): Linear increment per grid level. Default is 1.
        gf (float, optional): Growth factor for stretching the profile beyond `zstretch`. Default is 1.08.
        zstretch (float, optional): Threshold height. Profile values below this height are linearly spaced,
                                    while values beyond this are exponentially stretched. Default is 1e9.

    Returns:
        np.array: Array of profile values, linearly increasing until `zstretch` and stretched thereafter.

    Notes:
        - Linear values are computed as `prof = y0 + dy0 * z` for all `z`.
        - For `z` values below `zstretch`, the linear profile remains unchanged.
        - For `z` values above `zstretch`, the linear profile is raised to the power of `gf` to introduce stretching.
    """

    if zstretch < z[-1]:
        ssi = np.searchsorted(z, zstretch)
        dy = np.array(
            [dy0 if i < ssi else dy0 * (gf ** (i - ssi)) for i in range(len(z))]
        )
    else:
        dy = dy0

    return y0 + dy * z


class DALESInpGenerator:
    """
    Generates a vertical grid and associated profile datasets for DALES profile input generation.

    Attributes:
        expn (str): Experiment identifier, used as a suffix in output file names.
        kmax (int): Number of vertical grid cells (total grid points = `kmax + 1`).
        dz0 (float): Initial vertical grid spacing in meters.
        gf (float): Growth factor for stretching the vertical grid.
        stretch_start_index (int): Index at which grid stretching begins. Default is a high value to disable by default.
        zh (numpy.ndarray): Vertical grid edges with `kmax + 1` points (boundary points of each grid cell).
        z (numpy.ndarray): Vertical grid centers with `kmax` points (midpoints of each grid cell).

    Profile DataFrames:
        - `dfprof` (pd.DataFrame): Stores profile columns such as `thl`, `qt`, `u`, `v`, and `tke`.
        - `dflscale` (pd.DataFrame): Stores large-scale forcing data (`ug`, `vg`, `wfls`, `dqtdx`, `dqtdy`, `dqtdt`, `dthldt`).
        - `dfbaseprof` (pd.DataFrame): Stores base profile data (`rhobf` for base density).
        - `dfscalar` (pd.DataFrame): Stores scalar profile columns dynamically added via `add_scalars`.

    Time-Varying DataFrames:
        - `dflsfluxsurf` (pd.DataFrame): Surface flux data (`time`, `wtsurf`, `wqsurf`, `thlsurf`, `qtsurf`, `psurf`).
        - `dflsflux` (pd.DataFrame): Large-scale flux forcing terms (`time` and vertical profile columns).
        - `dfnudge` (pd.DataFrame): Nudge profiles with time-dependent adjustments (`time`, `z`, `factor`, `u`, `v`, `w`, `thl`, `qt`).

    Class Constants:
        - `_fillval` (float): Default value for uninitialized profile columns.
        - `_prof_cols` (list[str]): Default columns for profile datasets (`z`, `thl`, `qt`, `u`, `v`, `tke`).
        - `_lscale_cols` (list[str]): Default columns for large-scale forcing datasets.
        - `_baseprof_cols` (list[str]): Default columns for base profiles (`z`, `rhobf`).
        - `_nudge_cols` (list[str]): Default columns for nudge profiles.
        - `_ls_flux_surf_cols` (list[str]): Default columns for surface flux datasets.
    """

    _fillval = 0.0

    _prof_cols = ["z", "thl", "qt", "u", "v", "tke"]
    _lscale_cols = [
        "z",
        "ug",
        "vg",
        "wfls",
        "dqtdx",
        "dqtdy",
        "dqtdt",
        "dthldt",
    ]
    _baseprof_cols = ["z", "rhobf"]

    _nudge_cols = ["time", "z", "factor", "u", "v", "w", "thl", "qt"]
    _ls_flux_surf_cols = ["time", "wtsurf", "wqsurf", "thlsurf", "qtsurf", "psurf"]

    def __init__(
        self,
        kmax,
        dz0=1,
        gf=1.08,
        stretch_start_index=9999,
        expn="001",
        output_dir=".",
        write_created_at=True,
        format_opts=None,
        **kwargs,
    ):
        """
        Initialize the DALES input generator.

        This constructor creates and initializes the vertical grid and associated
        profile datasets for use in DALES. Profile columns can be pre-populated
        using `kwargs`, or left as zero-filled arrays by default.

        Parameters:
            kmax (int): Number of vertical grid cells (total points = kmax + 1).
            dz0 (float): Initial spacing between vertical levels. Default is 1.
            gf (float): Growth factor for grid stretching. Default is 1.08.
            stretch_start_index (int): Index where grid stretching starts. Default is 9999.
            expn (str): Experiment identifier used in output file naming. Default is "001".
            output_dir (str): Directory where generated files will be saved. Default is ".".
            write_created_at (bool): Whether to include a timestamp in file headers. Default is True.
            format_opts (dict | None): configure formating of output text files:
                default => format_opts = {
                    "low_sci ": 1e-3,   # low treshold to use scientific notation
                    "high_sci ": 1e3,   # high treshold to use scientific notation
                    "format_width": 12, # format width {format_width.format_precision}
                    "format_precision": 5, # format width {format_width.format_precision}
                }
            **kwargs: Supported keyword arguments to populate profile columns. If not provided,
                    these columns will be initialized with zeros.

                Supported Profile Columns:
                    - Scalar profiles: User-defined via `add_scalars`.
                    - Default columns (already included in datasets):
                        * Profiles (`dfprof`): `thl`, `qt`, `u`, `v`, `tke`.
                        * Large scale forcing (`dflscale`): `ug`, `vg`, `wfls`, `dqtdx`,
                        `dqtdy`, `dqtdt`, `dthldt`.
                        * Base profiles (`dfbaseprof`): `rhobf`.

                Values for `kwargs`:
                    - Scalar value (`float` or `int`): Assigns a constant value to the column.
                    - Array (`np.ndarray`): Must match the grid size `kmax`. Specifies custom
                    profile data for the column.
                    - Profile specification (`dict`): Allows dynamic generation of profiles
                    using one of the following types:
                        * `log`: Specify logarithmic profile parameters:
                            - `z0` (roughness length, default 0.1)
                            - `d` (displacement height, default 0)
                            - `ustar` (friction velocity, default 1)
                        * `linear`: linear profile parameters:
                            - `y0` (base value, default 0)
                            - `dy0` (slope: dy/dz, default 1)
                        * `stretch`: Specify linear-stretch profile parameters:
                            - `y0` (base value, default 0)
                            - `dy0` (initial slope: dy/dz, default 1)
                            - `gf` (growth factor, default 1.08)
                            - `zstretch` (threshold height, default 1e9)

                Example `kwargs` usage:
                    * `thl=300`: Sets `thl` (liquid water potential temperature) to 300 everywhere.
                    * `qt=np.linspace(0.01, 0.02, kmax)`: Defines `qt` (specific humidity) as a
                    linear gradient.
                    * `u={'profile': 'log', 'z0': 0.1, 'd': 10, 'ustar': 0.5}`: Generates a
                    logarithmic profile for `u` (horizontal velocity).

        Raises:
            AssertionError: If `stretch_start_index` is less than or equal to 0.
        """
        assert stretch_start_index > 0

        self.format_opts = {
            "low_sci": 1e-3,
            "high_sci": 1e3,
            "format_width": 12,
            "format_precision": 5,
        }

        # override format options
        if isinstance(format_opts, dict):
            for k in self.format_opts.keys():
                v = format_opts.get(k)
                if v is not None:
                    self.format_opts[k] = v

        self.expn = expn
        self.kmax = kmax
        self.dz0 = dz0
        self.gf = gf
        self.stretch_start_index = stretch_start_index
        self.write_created_at = write_created_at
        self.output_dir = output_dir
        # Generate grid on initialization
        self.zh = self._generate_vertical_grid()
        self.z = (self.zh[:-1] + self.zh[1:]) / 2

        self.dfscalar = (
            pd.DataFrame()
        )  # this is special since the user can add as many columns as they want with add_scalar func

        self.dfprof = self.create_prof_df(**kwargs)
        self.dflscale = self.create_lscale_df(**kwargs)
        self.dfbaseprof = self.create_baseprof_df(**kwargs)

        # these dataframes are used sometimes, and are timeseries ones so not initialized
        self.dflsfluxsurf = pd.DataFrame(columns=self._ls_flux_surf_cols, dtype=float)
        self.dflsflux = pd.DataFrame(columns=["time"] + self._lscale_cols, dtype=float)
        self.dfnudge = pd.DataFrame(columns=self._nudge_cols, dtype=float)
        # time col should be int - > seconds
        self.dflsfluxsurf = self.dflsfluxsurf.astype({"time": int})
        self.dflsflux = self.dflsflux.astype({"time": int})
        self.dfnudge = self.dfnudge.astype({"time": int})

    def _df2lines(self, df):
        return df2lines(df, **self.format_opts)

    def write_file(self, filename: str, header: str, lines: list[str]):
        filename = path.join(self.output_dir, filename)

        with open(filename, "w") as f:
            f.write(header)
            f.writelines(lines)

    def add_scalars(self, **kwargs):
        self.dfscalar = self.create_scalar_df(**kwargs)

    def add_ls_flux(self, dfprof: pd.DataFrame, time=0, **kwargs):
        assert (
            time not in self.dflsfluxsurf["time"].values
        ), f"Entries time {time} already exist"
        assert isinstance(time, int)

        dfprof = dfprof.copy()

        row = {k: self._fillval for k in self._ls_flux_surf_cols}
        row["time"] = time
        for k, v in kwargs.items():
            if k in self._ls_flux_surf_cols:
                row[k] = float(v)  # float to make sure we dont do crazy things

        self.dflsfluxsurf.loc[len(self.dflsfluxsurf)] = pd.Series(row)
        self.dflsfluxsurf = self.dflsfluxsurf.astype({"time": int})

        dfprof["time"] = time

        self.dflsflux = (
            pd.concat((self.dflsflux, dfprof))
            .sort_values(["time", "z"])
            .reset_index(drop=True)
        )

    def add_nudge(self, time=0, **kwargs):
        """kwargs: factor ,  u ,  v ,  w ,  thl ,  qt"""

        assert (
            time not in self.dfnudge["time"].values
        ), f"Entries time {time} already exist"

        df = pd.DataFrame(columns=self._nudge_cols, dtype=float)
        df = df.astype({"time": int})
        df["z"] = self.z
        df["time"] = time

        for c in self._nudge_cols[2:]:
            df[c] = self._validate_prof(kwargs.get(c))

        # make sure things are always sorted
        self.dfnudge = (
            pd.concat((self.dfnudge, df))
            .sort_values(["time", "z"])
            .reset_index(drop=True)
        )

    def create_scalar_df(self, **kwargs):
        df = pd.DataFrame({"z": self.z}, dtype=float)
        for k, v in kwargs.items():
            df[k] = self._validate_prof(v)
        return df

    def create_lscale_df(self, **kwargs):
        """kwargs:  ug  vg  wfls  dqtdx  dqtdy  dqtdt  dthldt"""
        df = pd.DataFrame(columns=self._lscale_cols, dtype=float)
        df["z"] = self.z

        for c in self._lscale_cols[1:]:
            df[c] = self._validate_prof(kwargs.get(c))
        return df

    def create_prof_df(self, **kwargs):
        """kwargs: thl, qt, u, v, tke"""

        df = pd.DataFrame(columns=self._prof_cols, dtype=float)
        df["z"] = self.z

        for c in self._prof_cols[1:]:
            df[c] = self._validate_prof(kwargs.get(c))
        return df

    def create_baseprof_df(self, **kwargs):
        rhobf = kwargs.get("rhobf")
        df = pd.DataFrame(columns=self._baseprof_cols, dtype=float)
        if rhobf is None:
            return df
        df["z"] = self.z
        df["rhobf"] = self._validate_prof(rhobf)
        return df

    def _generate_vertical_grid(self):
        """Returns: numpy.ndarray: An array representing the vertical grid edges (`kmax + 1`) points)."""
        grid = np.linspace(0, self.kmax, self.kmax + 1)

        ssi = self.stretch_start_index
        if ssi > len(grid):
            dz = self.dz0
        for i in range(ssi, self.kmax):
            dz = np.array(
                [
                    self.dz0 if i < ssi else self.dz0 * (self.gf ** (i - ssi))
                    for i in range(self.kmax + 1)
                ]
            )

        return grid * dz

    def _validate_prof(self, x):
        if x is None:
            return self._fillval

        if isinstance(x, dict):
            x = deepcopy(x)
            proftype = x.pop("profile")
            if proftype == "log":
                return self._log_prof(**x)

            if proftype == "linear":
                return self._linear_prof(**x)
            if proftype == "stretch":
                return self._stretch_prof(**x)

        elif isinstance(x, np.ndarray):
            assert len(x) == len(
                self.z
            ), f"profile lenght ({len(x)}) does not match length of z: {len(self.z)}"
            return x
        else:
            try:
                return float(x)
            except:
                raise ValueError("Could not parse profile into a float")

    # def _log_prof(self, z0=0.1, d=10, ustar=1):
    #     return log_prof(self.z,z0,d,ustar)

    # def _stretch_prof(self, y0=0, dy0=1, gf=1.08, zstretch=10e8):
    #     return stretch_prof(self.z,y0,dy0,gf,zstretch)

    def _log_prof(self, **kwargs):
        return log_prof(self.z, **kwargs)

    def _stretch_prof(self, **kwargs):
        return stretch_prof(self.z, **kwargs)

    def _linear_prof(self, **kwargs):
        y0 = kwargs.get("y0", 0)
        dy0 = kwargs.get("dy0", 1)
        return y0 + dy0 * self.z

    def write_simple_profile(self, filename, df, header):

        filename = f"{filename}.{self.expn}"
        # Construct header
        vnames = "     |     ".join(df.columns)
        if self.write_created_at:
            header = f"# {header}\n       {vnames}\n"
        else:
            header = f"# {header} - {isonow()}\n       {vnames}\n"

        # Write to file
        self.write_file(filename, header, self._df2lines(df))

    def write_prof(self, filename="prof.inp", header="Input Profiles"):
        self.write_simple_profile(filename, self.dfprof, header)

    def write_lscale(self, filename="lscale.inp", header="Large Scale Forcings"):
        self.write_simple_profile(filename, self.dflscale, header)

    def write_baseprof(self, filename="baseprof.inp", header="Base Density Profile"):
        self.write_simple_profile(filename, self.dfbaseprof, header)

    def write_scalar(self, filename="scalar.inp", header="Scalar Profile"):
        self.write_simple_profile(filename, self.dfscalar, header)

    def write_nudge(self, filename="nudge.inp", header="Nudge Profiles"):
        filename = f"{filename}.{self.expn}"
        if self.write_created_at:
            header = f"# {header}\n\n"
        else:
            header = f"# {header} - {isonow()}\n\n"

        # Format lines from the df
        lines = self._prepare_lines_for_timegroupped_df(self.dfnudge)

        # Write to file
        self.write_file(filename, header, lines)

    def write_ls_flux(self, filename="ls_flux.inp", header="Large scale fluxes"):
        filename = f"{filename}.{self.expn}"
        # Construct header
        vnames = "     |     ".join(self.dflsfluxsurf.columns)
        if self.write_created_at:
            header = f"# {header}\n      {vnames}\n\n"
        else:
            header = f"# {header} - {isonow()}\n      {vnames}\n\n"

        # Format lines from the df
        linessurf = self._df2lines(self.dflsfluxsurf)
        # Format lines from the df
        linestime = self._prepare_lines_for_timegroupped_df(self.dflsflux)
        lines = linessurf + ["\nLarge Scale Forcing Terms\n"] + linestime
        # Write to file
        self.write_file(filename, header, lines)

    def write_all_profiles(self):

        self.write_prof()
        self.write_lscale()

        if len(self.dfbaseprof) > 0:
            self.write_baseprof()
        if len(self.dfscalar.columns) > 1:
            self.write_scalar()
        if len(self.dfnudge) > 0:
            self.write_nudge()
        if len(self.dflsfluxsurf) > 0:
            self.write_ls_flux()

    def _prepare_lines_for_timegroupped_df(self, df: pd.DataFrame):
        lines = []
        vnames = "    |    ".join([c for c in df.columns if c != "time"])
        for time, group in df.groupby("time"):
            lines.append(f"       {vnames}\n")
            lines.append(f"#   {time}\n")
            lines.extend(self._df2lines(group.drop("time", axis=1)))
            # Add a blank line after each group
            lines.append("\n")
        return lines
