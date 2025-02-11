import numpy as np
from netCDF4 import Dataset


def src_val_from_mat(t: float, x: list[list[float, float, float]]):
    ret = 0
    for step in x:
        if step[0] <= t and t < step[1]:
            ret += step[2]
    return ret


def step_source_from_mat(t: np.ndarray, x: list[list[float, float, float]]):
    return np.array([src_val_from_mat(tt, x) for tt in t])


class GenTracerProfile:

    def __init__(
        self,
        file_path="./tracer_profile_001.nc",
        fieldump_path="completefielddump.nc",
        description="NetCDF file for tracer properties",
        history="Created with T_tracer structure equivalent",
    ):
        self.file_path = file_path
        with Dataset(fieldump_path) as dataset:
            self.x_size = dataset.dimensions["xt"].size
            self.y_size = dataset.dimensions["yt"].size
            self.z_size = dataset.dimensions["zt"].size
            self.t_size = dataset.dimensions["time"].size
            self.time_seconds = dataset.variables["time"][:]

        self._trac_dict = {}
        self._trac_num_src = {}

        self.ncfile = Dataset(self.file_path, "w", format="NETCDF4")
        # Optional: Add file metadata
        self.ncfile.description = description
        self.ncfile.history = history
        self.ncfile.numtracers = 0
        self.ncfile.createDimension("time", self.t_size)  # Unlimited time dimension
        self.ncfile.createDimension("x", self.x_size)
        self.ncfile.createDimension("y", self.y_size)
        self.ncfile.createDimension("z", self.z_size)
    
    def add_tracer(
        self, name: str, traclong="", unit="kg/kg", molar_mass=-999.0, lemis=True,initial_value = 0 
    ):
        assert (
            name not in self._trac_dict.keys()
        ), "Tracer with that name already exists"
        tracer = self.ncfile.createGroup(name)
        self._trac_dict[name] = tracer
        self._trac_num_src[name] = 0
        tracer.tracname = name  # Tracer name (max length 16)
        tracer.traclong = traclong  # Tracer long name (max length 64)
        tracer.unit = unit  # Tracer unit (max length 16)
        tracer.molar_mass = molar_mass  # Molar mass of tracer (float)
        tracer.lemis = "true" if lemis else "false"  # Emission flag (boolean)
        tracer.numsources = 10  # Emission flag (boolean)

        # init = tracer.createVariable("init", "f4", ("x", "y", "z"))
        init = tracer.createVariable("init", "f4", ("z", "y", "x"))
        init[:, :, :] = initial_value

    def add_point_source(self, tracer_name: str, x: int, y: int, z: int, values):
        assert (
            tracer_name in self._trac_dict
        ), f"Add a tracer with that name first, current tracers: {self._trac_dict.keys()}"
        assert x >= 0 and x <= self.x_size, f"x must be between 0 and {self.x_size}"
        assert y >= 0 and y <= self.y_size, f"y must be between 0 and {self.y_size}"
        assert z >= 0 and z <= self.z_size, f"z must be between 0 and {self.z_size}"

        tracer = self._trac_dict[tracer_name]
        self._trac_num_src[tracer_name] += 1
        source = tracer.createVariable(
            f"source_{self._trac_num_src[tracer_name]}", "f4", ("time")
        )
        source.x = x
        source.y = y
        source.z = z
        source[:] = values

    def close(self):

        self.ncfile.numtracers = len(self._trac_dict)
        totnumsources = 0

        for name, tracer in self._trac_dict.items():
            tracer.numsources = self._trac_num_src[name]
            totnumsources += self._trac_num_src[name]

        self.ncfile.totnumsources = totnumsources

        self.ncfile.close()
