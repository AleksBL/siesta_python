# siesta_python
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

Version 1.0

# Introduction
This is a code meant for easy handling of SIESTA, TranSIESTA and TBtrans calculations. This code allows to call these DFT programs directly in python, thereby allowing for better scripting using these programs. 

## Features 
- The SiP object allows for automatisation of many tasks, such as writing fdf files, rearranging geometries and finding particular indices to be passed to SIESTA. 
- wrapper functions around many features of SIESTA, TranSIESTA and sisl such as doping
- Handling of electrode and device objects and combining them inside the python script. 

## Installation
To install siesta_python, download the code as a zip file, unpack it and navigate to the siesta_python folder containing the setup.py file in a terminal. Now execute
```console
    python3 -m pip install -e .
```
and you will have an editable install of the code. 

## Contact 
Users can get in contact with the developer by submitting an issue on the Zandpack Github page. You also direct messages to aleksander.bl@proton.me.

## Examples
```python
import numpy as np
from siesta_python.siesta_python import SiP
import sisl

Sheet = sisl.geom.graphene(orthogonal = True).tile(2,0).tile(2,1)
pos = Sheet.xyz
cell = Sheet.cell
species = Sheet.to.ase().numbers
#make fourth atom nitrogen, (all atomic species up to Barium written into this program, else: go to funcs.py and add your element in Num2Sym)
species[3] = 7
Sheet = SiP(cell, pos, species, 
            
            #Not necessarily needed keywords below, take care with their default values though
            #Some labels for the folder name and calculation name
            directory_name = 'NitrogenDefectedDraphene', sl='NDG', sm = 'NDG',
            
            # siesta basis, k-point sampling, "mpi" defaults to "mpirun " but we set it to nothing
            # if you do mpirun, remember the space after -> "mpirun " <- because this is what is put into the os.system command
            basis = 'SZP', kp = [3,3,1], mpi = '', 
            
            # pseudopotential path to folder, defaults to "../pp" path relative to where "siesta_python.py" is put
            pp_path = '../pp',
            
            # DFT exchange-correlation functional, defaults to 'gga'
            xc = 'gga',
            
            #Overwrites any data in directory_name folder. Defaults to false, but can be True, False and 'reuse'
            overwrite = True, 
            )

# Write fdf file
Sheet.fdf()
Sheet.run_siesta_in_dir()

```
