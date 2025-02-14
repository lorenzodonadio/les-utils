import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interactive, fixed
from netCDF4 import Dataset
from matplotlib.colors import Normalize


class VizConcentration:
    def __init__(self, filepath, ibm_height):
        # Load the NetCDF file
        self.filepath = filepath
        self.dataset = Dataset(filepath, mode="r")
        self.ibm_height = ibm_height
        self.ibm_contour = np.zeros_like(ibm_height)
        # Extract dimensions
        self.x = self.dataset.variables["x"][:]
        self.y = self.dataset.variables["y"][:]
        self.z = self.dataset.variables["z"][:]
        self.time = self.dataset.variables["time"][:]
        self.nsv = self.dataset.dimensions["nsv"].size
        # Extract the concentration variable 'c'
        self.c = self.dataset.variables["c"]

    def get_concentration_at_time(self, time_index):
        """Get concentration data for a specific time index."""
        if time_index < 0 or time_index >= len(self.time):
            raise IndexError("Invalid time index")
        return self.c[time_index, :, :, :, :]

    def plot_slice(self, time_index, zidx, nsv=0, title="Concentration Slice"):
        """
        Plot a 2D slice of the concentration field at a specific time and z-index.

        Parameters:
        - time_index: Index of the time dimension to plot.
        - zidx: Index of the z-dimension to plot.
        """
        if zidx < 0 or zidx >= len(self.z):
            raise IndexError("Invalid z-index")

        # Get the 2D slice of concentration at the given time and z-index
        conc_slice = self.get_concentration_at_time(time_index)[nsv, zidx, :, :]
        z_curr = self.z[zidx]
        self.ibm_contour = np.zeros_like(self.ibm_height)
        self.ibm_contour[self.ibm_height > z_curr] = 1
        # Plot the 2D slice
        fig, ax = plt.subplots(figsize=(8, 6))
        # plt.figure(figsize=(8, 6))
        ax.contour(self.x, self.y, self.ibm_contour, cmap="Grays")
        cntplot = ax.contourf(self.x, self.y, conc_slice, alpha=0.8, cmap="Reds")
        # Add colorbar for the contourf plot
        fig.colorbar(cntplot, ax=ax, label="Concentration")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # plt.ylabel("Y")
        ax.set_title(
            f"{title} (Time: {self.time[time_index]:1.1f} s, Z: {self.z[zidx]}m"
        )
        # plt.title()
        # plt.show()
        return fig, ax

    def plot_interactive_slice(self):
        """Create an interactive contour plot for a variable over height and time using sliders."""
        height_slider = IntSlider(
            min=0,
            max=self.z.shape[0] - 1,
            step=1,
            value=2,
            description="Z:",
        )
        timestep_slider = IntSlider(
            min=0,
            max=self.time.shape[0] - 1,
            step=1,
            value=1,
            description="Timestep:",
        )
        nsv_slider = IntSlider(
            min=0,
            max=self.nsv - 1,
            step=1,
            value=0,
            description="Scalar Field Number",
        )

        interactive_plot = interactive(
            self.plot_slice,
            time_index=timestep_slider,
            zidx=height_slider,
            nsv=nsv_slider,
            title=fixed("Concentration Slice"),
        )

        display(interactive_plot)

    def plot_3d_slices(self, time_index, nsv=0, title="3D Concentration Slices"):
        """
        Plot multiple horizontal and vertical slices of the concentration field in a 3D space.

        Parameters:
        - time_index: Index of the time dimension to plot.
        - title: Title of the plot.
        """
        # Define slice indices
        z_indices = range(0, len(self.z), 2)  # Horizontal slices at various heights

        concentration_data = self.get_concentration_at_time(time_index)[nsv, :, :, :]
        global_max = np.max(concentration_data)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the IBM surface
        X, Y = np.meshgrid(self.x, self.y)
        ax.plot_surface(X, Y, self.ibm_height, cmap="Grays", alpha=0.5)

        # Plot horizontal slices (XY planes) at different heights (Z direction)
        for z_index in z_indices:
            if z_index < 0 or z_index >= len(self.z):
                continue
            z_curr = self.z[z_index]
            conc_slice = concentration_data[z_index, :, :]
            slicemax = np.max(conc_slice)
            alpha = slicemax / global_max  # Proportional alpha
            if alpha < 0.1:
                continue
            # Plot each slice as a 3D surface with contourf
            ax.contourf(
                X,
                Y,
                conc_slice,
                zdir="z",
                offset=z_curr,
                cmap="Reds",
                alpha=alpha,
                vmin=0,
                vmax=global_max,
            )

        # Y, Z = np.meshgrid(self.y, self.z)
        # for x_index in x_indices:
        #     x_curr = self.x[x_index]
        #     conc_slice = concentration_data[:, :, x_index]
        #     ax.plot_surface(x_curr * np.ones_like(Y), Y, Z, facecolors=plt.cm.Blues(conc_slice / global_max),
        #                     rstride=1, cstride=1, shade=False, alpha=0.5)

        # # Plot vertical slices in the XZ plane (constant Y)
        # X, Z = np.meshgrid(self.x, self.z)
        # for y_index in y_indices:
        #     y_curr = self.y[y_index]
        #     conc_slice = concentration_data[:, y_index, :]
        #     ax.plot_surface(X, y_curr * np.ones_like(X), Z, facecolors=plt.cm.Greens(conc_slice / global_max),
        #                     rstride=1, cstride=1, shade=False, alpha=0.5)

        # Set labels, title, and limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title} (Time: {self.time[time_index]} s)")
        ax.set_zlim([0, max(self.z)])
        plt.colorbar(ax.collections[1], ax=ax, label="Concentration")
        # plt.show()
        return fig, ax

    def plot_interactive_3d_slice(self):
        """Create an interactive contour plot for a variable over height and time using sliders."""

        timestep_slider = IntSlider(
            min=0,
            max=self.time.shape[0] - 1,
            step=1,
            value=1,
            description="Timestep:",
        )

        nsv_slider = IntSlider(
            min=0,
            max=self.nsv - 1,
            step=1,
            value=0,
            description="Scalar Field Number",
        )

        interactive_plot = interactive(
            self.plot_3d_slices,
            time_index=timestep_slider,
            nsv=nsv_slider,
            title=fixed("Concentration"),
        )

        display(interactive_plot)

    def plot_3d_scatter(
        self, time_index, nsv=0, show_treshold=0.01, cmap=plt.cm.Reds, title=""
    ):
        c = self.c[time_index, nsv, :, :, :]
        idx = np.where(c > show_treshold)
        vals = np.array(c[idx])
        z = np.array([self.x[i] for i in idx[0]])
        y = np.array([self.x[i] for i in idx[1]])
        x = np.array([self.x[i] for i in idx[2]])

        normed_values = vals / np.max(vals)
        colors = cmap(normed_values)

        # # Create alpha values for each self point based on its intensity and the specified decay factor
        # alphas = 0.4+ 0.6*(vals / np.max(vals)) ** decay

        # colors[:, -1] = alphas  # add alpha values to RGB values

        # Plot a 3D scatter with adjusted alphas

        fig = plt.figure(num=1, clear=True, figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([0, self.x[-1]])
        ax.set_ylim([0, self.y[-1]])
        ax.set_zlim([0, self.z[-1]])
        ax.set_aspect("equal")
        # Plot the IBM surface
        X, Y = np.meshgrid(self.x, self.y)
        ax.plot_surface(X, Y, self.ibm_height, cmap="Grays", alpha=0.6)
        # mydensityplot3d(ax,x,y,z,vals,decay=1,s=0.2)
        # ,linewidth=20
        ax.scatter(x, y, z, c=colors, s=0.2, marker="*")
        # ax.scatter(x, y, z)
        cbar = plt.cm.ScalarMappable(
            norm=Normalize(np.min(vals), np.max(vals)), cmap=cmap
        )
        # Add colorbar with labels and padding
        fig.colorbar(cbar, ax=ax, fraction=0.02, pad=0.1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title} (Time: {self.time[time_index]} s)")
        return fig, ax

    def plot_interactive_3d_scatter(self, show_treshold=0.01, cmap=plt.cm.Reds):
        timestep_slider = IntSlider(
            min=0,
            max=self.time.shape[0] - 1,
            step=1,
            value=1,
            description="Timestep:",
        )

        nsv_slider = IntSlider(
            min=0,
            max=self.nsv - 1,
            step=1,
            value=0,
            description="Scalar Field Number",
        )

        interactive_plot = interactive(
            self.plot_3d_scatter,
            nsv=nsv_slider,
            time_index=timestep_slider,
            title=fixed("Concentration"),
            cmap=fixed(cmap),
            show_treshold=fixed(show_treshold),
        )

        display(interactive_plot)

    # Example usage:
    # animation = plot_3d_scatter_animation(self)

    def plot_time_series(self, x_index, y_index, z_index, nsv=0):
        """
        Plot the time series of concentration at a specific (x, y, z) point.

        Parameters:
        - x_index: Index of the x-dimension to plot.
        - y_index: Index of the y-dimension to plot.
        - z_index: Index of the z-dimension to plot.
        """
        if (
            x_index < 0
            or x_index >= len(self.x)
            or y_index < 0
            or y_index >= len(self.y)
            or z_index < 0
            or z_index >= len(self.z)
        ):
            raise IndexError("Invalid index for x, y, or z")

        # Extract the concentration at the given (x, y, z) point over time
        concentration_time_series = self.c[:, nsv, z_index, y_index, x_index]

        # Plot the time series
        plt.figure(figsize=(8, 4))
        plt.plot(self.time, concentration_time_series, marker="o")
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.title(
            f"Concentration Time Series at (X index: {x_index}, Y index: {y_index}, Z index: {z_index})"
        )
        plt.grid(True)
        plt.show()

    def close(self):
        """Close the NetCDF dataset."""
        self.dataset.close()
