import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import IntSlider, interactive
from netCDF4 import Dataset

class VizFieldDump:

    def __init__(self, file_name: str):
        self.file_name = file_name

        self.dims = {}
        with Dataset(self.file_name, mode="r") as dataset:
            self.dims["time"] = dataset.variables["time"][:]
            self.dims["zt"] = dataset.variables["zt"][:]
            self.dims["zm"] = dataset.variables["zm"][:]
            self.dims["xt"] = dataset.variables["xt"][:]
            self.dims["xm"] = dataset.variables["xm"][:]
            self.dims["yt"] = dataset.variables["yt"][:]
            self.dims["ym"] = dataset.variables["ym"][:]

    def extract_field(self, var_name: str, time_start: int, time_end: int):
        """Extracts a specific time window of the variable from all files."""
        try:
            with Dataset(self.file_name, mode="r") as dataset:
                # Extract the time window for the variable
                return dataset.variables[var_name][time_start:time_end, :, :, :]
        except Exception as e:
            print(e)
            return None

    def quiver_plot(
        self,
        u_field,
        v_field,
        axes,
        x3_idx=0,
        timestamp=0,
        stride = 2,
        background_field="thl",
        figsize=(8, 6),
    ):
        """
        Plot a quiver plot for velocity fields (u, v, w) along chosen axes (x, y, z) at a specific timestamp.

        Parameters:
            u_field (str): Name of the first velocity field ('u', 'v', or 'w').
            v_field (str): Name of the second velocity field ('u', 'v', or 'w').
            axes (str): The spatial axis for the x-dimension ('xy','xz','yz').
            x3_idx (int): the index for the slice of the dimension that is not shown.
            timestamp (int): The time index to plot (default is 0).
            stride (int): spacing for the arrows.
            background_field (str): Name of the background scalar field to plot.
            figsize (tuple): Figure size for the plot.

        Returns:
            fig (Figure): The matplotlib figure object.
            ax (Axes): The matplotlib axes object.
        """

        # Validate input
        if u_field not in ["u", "v", "w"] or v_field not in ["u", "v", "w"]:
            raise ValueError("Invalid field. Choose from 'u', 'v', or 'w'.")
        if axes not in ["xy", "xz", "yz"]:
            raise ValueError("Invalid axis. Choose valid spatial axes.")
        if timestamp >= len(self.dims["time"]):
            raise ValueError("Timestamp index out of range.")
        # Extract the specific time slice using the updated method
        u_data = self.extract_field(u_field, timestamp, timestamp + 1)
        v_data = self.extract_field(v_field, timestamp, timestamp + 1)
        xx = self.dims[axes[0] + "t"]
        yy = self.dims[axes[1] + "t"]

        # Reduce data to 2D slices at the specified timestamp
        # if axes == "xy":
        #     u_slice = u_data[0, x3_idx, :, :]
        #     v_slice = v_data[0, x3_idx, :, :]
        # elif axes == "xz":
        #     u_slice = u_data[0, :, x3_idx, :]
        #     v_slice = v_data[0, :, x3_idx, :]
        # elif axes == "yz":
        #     u_slice = u_data[0, :, :, x3_idx]
        #     v_slice = v_data[0, :, :, x3_idx]

        if axes == "xy":
            u_slice = u_data[0, x3_idx, ::stride, ::stride]
            v_slice = v_data[0, x3_idx, ::stride, ::stride]
        elif axes == "xz":
            u_slice = u_data[0, ::stride, x3_idx, ::stride]
            v_slice = v_data[0, ::stride, x3_idx, ::stride]
        elif axes == "yz":
            u_slice = u_data[0, ::stride, ::stride, x3_idx]
            v_slice = v_data[0, ::stride, ::stride, x3_idx]

        fig, ax = plt.subplots(figsize=figsize)
        quiver = ax.quiver(xx[::stride], yy[::stride], u_slice, v_slice)

        # Plot background field if provided
        if background_field:
            bg_data = self.extract_field(background_field, timestamp, timestamp + 1)
            if axes == "xy":
                bg_slice = bg_data[0, x3_idx, :, :]
            elif axes == "xz":
                bg_slice = bg_data[0, :, x3_idx, :]
            elif axes == "yz":
                bg_slice = bg_data[0, :, :, x3_idx]

            c = ax.imshow(
                bg_slice,
                extent=[xx[0], xx[-1], yy[0], yy[-1]],
                origin="lower",
                cmap="viridis",
                alpha=0.5,
            )
            fig.colorbar(c, ax=ax, label=background_field)

        ax.set_title(
            f"Quiver plot of {u_field} and {v_field} at time index {timestamp}"
        )
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_aspect("equal")
        return fig, ax

    def plot_xycontour(self, var, height, timestep):
        """Plot a 2D contour plot for a specific variable at a given height and timestep."""
        try:
            # Extract only the necessary time step for the plot
            if var == "speed":
                u = self.extract_field("u", timestep, timestep + 1)
                v = self.extract_field("v", timestep, timestep + 1)
                data = np.sqrt(u[0, height, :, :] ** 2 + v[0, height, :, :] ** 2)
            else:
                data = self.extract_field(var, timestep, timestep + 1)[0, height, :, :]
        except Exception as e:
            print(e)
            return

        fig, ax = plt.subplots(figsize=(8, 7))
        c = ax.contourf(data, cmap="viridis")
        fig.colorbar(c, ax=ax, label=var)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{var} at time {self.dims['time'][timestep]}s, height: {self.dims['zt'][height]}m"
        )
        plt.show()

    def plot_interactive_xycontour(self, var):
        """Create an interactive contour plot for a variable over height and time using sliders."""
        height_slider = IntSlider(
            min=0,
            max=self.dims["zt"].shape[0] - 1,
            step=1,
            value=10,
            description="Height:",
        )
        timestep_slider = IntSlider(
            min=0,
            max=self.dims["time"].shape[0] - 1,
            step=1,
            value=1,
            description="Timestep:",
        )

        interactive_plot = interactive(
            self.plot_xycontour,
            var=widgets.fixed(var),
            height=height_slider,
            timestep=timestep_slider,
        )

        display(interactive_plot)