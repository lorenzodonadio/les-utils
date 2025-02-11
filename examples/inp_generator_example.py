import numpy as np
from dales_input_generator.dales_inp_generator import DALESInpGenerator

# make sure you have dales_inp_generator.py in your current working directory


# Example usage of the DALESInpGenerator class
def main():
    # Initialize grid parameters
    kmax = 256  # Number of vertical levels
    dz0 = 1.0  # Initial grid spacing
    gf = 1.01  # Growth factor for stretching
    stretch_start_index = 56  # Stretching starts at this index

    # Profiles examples:
    logwind = {
        "profile": "log",
        "z0": 0.1,
        "d": 0.45,
        "ustar": 5,
    }
    linear_then_streched = {
        "profile": "stretch",
        "y0": 0,  # intersect
        "dy0": 0.5,  # slope
        "gf": 1.2,  # growth factor
        "zstretch": 50,  # height form which growth factor takes effect
    }

    # if zstretch is not provided then this becomes just linear profile
    thl_linear = {
        "profile": "linear",
        "y0": 298,
        "dy0": 0.1,
    }
    wfls_linear = {
        "profile": "linear",
        "y0": 0,
        "dy0": 0.00022,
    }

    # Initialize DALESInpGenerator
    ig = DALESInpGenerator(
        kmax=kmax,
        dz0=dz0,
        gf=gf,
        stretch_start_index=stretch_start_index,
        expn="001",
        output_dir="./output",  # make sure the directory exists
        # Populate initial profiles using kwargs
        thl=thl_linear,  # Constant potential temperature
        qt=np.linspace(0.01, 0.02, kmax),  # Specific humidity gradient
        tke=0.1,
        u=logwind,  # easy to reuse profiles
        v=logwind,  # Linear-stretch wind profile
        wfls=wfls_linear,
    )

    # Add scalar profiles dynamically
    # one can also create profiles based on the generated height profile now
    pmprof = np.array([2 * np.random.random() * z**-0.2 for z in ig.z])
    ig.add_scalars(pm=pmprof, ch4=1, qr=0, nr=0)

    # Add nudging profiles
    ig.add_nudge(time=0, factor=0)
    ig.add_nudge(
        time=3600,
        factor=0.1,
        v={
            "profile": "log",
            "z0": 0.1,
            "ustar": 3,
        },
        thl=305,  # Nudging to a constant temperature
    )

    ig.add_nudge(
        7400,
        v={
            "profile": "log",
            "z0": 0.1,
            "ustar": 1,
        },
    )

    # Add Large Scale Forcing terms
    # for the first timestep it is recomended that the profiles in lscale match
    # the first profile in ls_flux, so just reuse the lsprof dataframe dflscale
    ig.add_ls_flux(ig.dflscale, time=0, wtsurf=0.1)
    # for the next times we can easily create more dataframes using create_lscale_df
    ig.add_ls_flux(
        ig.create_lscale_df(ug=1, dthldt={"profile": "linear", "dy0": 0.2}),
        time=3600,
        wtsurf=0.2,
        thlsurf=0.1,
    )

    ig.add_ls_flux(
        ig.create_lscale_df(ug=5, dthldt={"profile": "linear", "dy0": -0.2}),
        time=7200,
        wtsurf=0.2,
        thlsurf=0.1,
    )

    ig.write_all_profiles()

    # Output confirmation
    print(f"Profiles successfully written to: {ig.output_dir}")


if __name__ == "__main__":
    main()
