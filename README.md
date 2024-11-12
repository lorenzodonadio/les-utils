
# LES-Utils

LES-Utils is a set of utilities for processing and managing NetCDF data output from large eddy simulations (LES), particularly from DALES simulations.

**Installation Requirements**

You can install the dependencies via either pip or Lmod.
Option 1: Using pip

Install dependencies from requirements.txt:

`pip install -r requirements.txt`

Option 2: Using Lmod Modules

Load necessary modules with Lmod:

module load <module-name> 

**Example Usage**

Suppose you have a typical DALES output directory containing multiple NetCDF files, located here:

`ls ../utrecht_result_data/utrecht_ideal_save/`

Example directory structure:
```
fielddump.000.003.001.nc  fielddump.001.003.001.nc  fielddump.002.003.001.nc  fielddump.003.003.001.nc
fielddump.000.000.001.nc  fielddump.001.000.001.nc  fielddump.002.000.001.nc  fielddump.003.000.001.nc  ibm.inp.001
fielddump.000.001.001.nc  fielddump.001.001.001.nc  fielddump.002.001.001.nc  fielddump.003.001.001.nc  profiles.001.nc
fielddump.000.002.001.nc  fielddump.001.002.001.nc  fielddump.002.002.001.nc  fielddump.003.002.001.nc  profiles_lite.001.nc
```
To merge the files and include profile data, use the following command:

```python fieldmerge.py merge --input_dir ../utrecht_result_data/utrecht_ideal_save/ \
    --profile_file ../utrecht_result_data/utrecht_ideal_save/profiles_lite.001.nc
```

    --input_dir: The directory containing DALES output NetCDF files to be merged.
    --profile_file: An optional profile file to be added to the merged output make sure is profile_lite.001.nc profile does not contain all the information.
    --output_file: defaults to ./completefielddump.nc


Verify merging with: `ncdump -h completefielddump.nc`

Expected output: 
```
netcdf completefielddump {
dimensions:
        time = 1480 ;
        zt = 20 ;
        zm = 20 ;
        xt = 128 ;
        xm = 128 ;
        yt = 128 ;
        ym = 128 ;
variables:
        float time(time) ;
        float zt(zt) ;
        float zm(zm) ;
        float xt(xt) ;
        float xm(xm) ;
        float yt(yt) ;
        float ym(ym) ;
        float u(time, zt, yt, xm) ;
        float v(time, zt, ym, xt) ;
        float w(time, zm, yt, xt) ;
        float ekh(time, zt, yt, xt) ;
        float rhof(time, zt) ;
        float rhobf(time, zt) ;
        float rhobh(time, zm) ;
        float presh(time, zt) ;
}
```