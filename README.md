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
## Additional needs
You need to get a folder with the pseudo potentials (In the examples below it is "../pp" where this folder then has ../pp/C.gga.psf, ../pp/Au.gga.psf etc...). You can get these from e.g Virtual Vault (.psf files )or Pseudo dojo (.psml files).
## Examples
Simple DFT calculation of a nitrogen defect. 
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

Two crossed gold chains, giving four electrodes in total.
```python
import numpy as np
from siesta_python.siesta_python import SiP
import sisl
import matplotlib.pyplot as plt
# Initial Construction of the geometries
chain = sisl.Geometry([[0,0,0]], atoms=sisl.Atom[79], sc=[1.4, 1.4, 11])

elec_x = chain.tile(4, axis=0).add_vacuum(11 - 1.4, 1)
elec_y = chain.tile(4, axis=1).add_vacuum(11 - 1.4, 0)

chain_x = elec_x.tile(4, axis=0)
chain_y = elec_y.tile(4, axis=1)

chain_x = chain_x.translate(-chain_x.center(what='xyz'))
chain_y = chain_y.translate(-chain_y.center(what='xyz'))

device = chain_x.append(chain_y.translate([0, 0, -chain.cell[2, 2] + 2.1]), 2)
# Correct the y-direction vacuum
device = device.add_vacuum(chain_y.cell[1, 1] - chain_x.cell[1,1], 1)
device = device.translate(device.center(what='cell'))

emx = elec_x.move( device.xyz[np.where(device.xyz[:,0] == device.xyz[:,0].min()),:][0][0]  + 2 * chain.cell[0,:])
epx = elec_x.move( device.xyz[np.where(device.xyz[:,0] == device.xyz[:,0].max()),:][0][0]  - 5 * chain.cell[0,:])
emy = elec_y.move( device.xyz[np.where(device.xyz[:,1] == device.xyz[:,1].min()),:][0][0]  + 2 * chain.cell[1,:])
epy = elec_y.move( device.xyz[np.where(device.xyz[:,1] == device.xyz[:,1].max()),:][0][0]  - 5 * chain.cell[1,:])

# Make objects
EMX = SiP(emx.cell, emx.xyz, emx.to.ase().numbers,
          mpi = '', #<-- single process, delete this line / write 'mpirun ' if you want to use mpirun
          directory_name = 'EMX', sl = 'EMX',
          kp = [20,1,1], semi_inf = '-a1', mesh_cutoff=300.0,
          pp_path = '../pp'
          )
EPX = SiP(epx.cell, epx.xyz, epx.to.ase().numbers,
          mpi = '',
          directory_name = 'EPX', sl = 'EPX', 
          kp = [20,1,1], semi_inf = '+a1', mesh_cutoff=300.0,
          pp_path = '../pp'
          )

EMY = SiP(emy.cell, emy.xyz, emy.to.ase().numbers,
          mpi = '',
          directory_name = 'EMY', sl = 'EMY',
          kp = [1,20,1], semi_inf = '-a2',  mesh_cutoff=300.0,
          pp_path = '../pp'
          )

EPY = SiP(epy.cell, epy.xyz, epy.to.ase().numbers,
          mpi = '',
          directory_name = 'EPY', sl = 'EPY', 
          kp = [1,20,1], semi_inf = '+a2',  mesh_cutoff=300.0,
          pp_path = '../pp'
          )

elecs = [EMX, EPX, EMY, EPY]
for e in elecs: e.fdf(); e.run_siesta_in_dir()
# Define buffer atoms 
def buffer_atoms(x):
    if (x[0:2] <  2.5).any() or (x[0:2] > 20).any():
        return True
    return False
# Define contour parameters 
CS1 = {'V1'  :'-130.0', 'V2'  : '-15',
       'Np_1':  '65',   'Np_2':  '15' }

Dev = SiP(device.cell, device.xyz, device.to.ase().numbers,
          pp_path = '../pp', 
          mpi = '', mesh_cutoff=300.0,
          directory_name = 'Device', solution_method = 'transiesta',
          kp = [1,1,1], overwrite = True,
          kp_tbtrans = [1,1,1],print_console=True,
          elecs = elecs, contour_settings = [CS1,CS1,CS1,CS1],
          Chem_Pot = [0.0, 0.0, 0.0, 0.0]
          )

Dev.find_elec_inds(tol = 1e-2) # If you place your atoms 
Dev.set_buffer_atoms(buffer_atoms)

# Run
Dev.fdf()
Dev.write_more_fdf(['TS.Hartree.Fix +A'], name = 'TS_TBT')
Dev.run_analyze_in_dir()
Dev.run_siesta_in_dir()
Dev.run_tbtrans_in_dir(DOS_GF = True)
tbt = Dev.read_tbt()
# Plot
plt.plot(tbt.E, tbt.transmission(0,1))
plt.plot(tbt.E, tbt.transmission(0,1))
plt.plot(tbt.E, tbt.transmission(0,2))
plt.plot(tbt.E, tbt.transmission(0,3))
plt.plot(tbt.E, tbt.transmission(1,2))
plt.plot(tbt.E, tbt.transmission(2,3))
```
