# siesta_python
Siesta, tbtrans...... without touching the terminal

Use at your own resposibility and so on.... BUT:
## Write to me if you want something implemented: abalo@dtu.dk


### But how does it work?
Firstly, its not magic, you need to have siesta, transiesta and tbtrans compiled on your computer and you need to know their paths.
Favorably you can have them in your .bashrc file so that you can just call them as "siesta RUN.fdf > RUN.out", "tbtrans RUN.fdf > RUN.out", but it isnt strictly necessary, as you can also just give the code your paths to the various executables. Just "siesta" and "tbtrans" are however default values. 

Your folder should look like this:
Calc.py contains the code that you will see below, pp the pseudo-potentials, and siesta_python.py and funcs.py are the files you see on this webpage.

![image](https://user-images.githubusercontent.com/75378674/120886643-4925c600-c5ef-11eb-9cd6-d49e1b68314c.png)


## How to set up a siesta-calculation
Copy the "siesta_python.py" and "funcs.py" scripts to the folder of your choice, together with a folder with the pseudo-potentials you will use. The pseudo-potentials must be named like this: 'atom'.'type'.psf, eg. for the molybdenum gga pseudo-potential would read Mo.gga.psf or the carbon lda: C.lda.psf. We will later reference this folder in the initialisation of the class contained in 'siesta_python.py'.

Before starting, we should have a folder called "pp" with the pseudo-potentials in the same folder you are running the calculation in. You should also be able to run siesta from the terminal using just "siesta RUN.fdf > RUN.out", else you will need to additionally specify the "siesta_exec" keyword in the initialisation below.


We can make the atomic structure using [sisl](http://zerothi.github.io/sisl/docs/latest/index.html) (Which can also do a lot of other stuff, take a look). Lets make a simple graphene sheet with a nitrogen defect in the middle: 
```
import numpy as np
from siesta_python import SiP
import sisl

Sheet = sisl.geom.graphene(orthogonal = True).tile(2,0).tile(2,1)

pos = Sheet.xyz
cell = Sheet.cell
species = Sheet.toASE().numbers
#make fourth atom nitrogen, (all atomic species up to Barium written into this program, else: go to funcs.py and add your element in Num2Sym)
species[3] = 7

Sheet = SiP(cell, pos, species, 
            
            #Not necessarily needed keywords below, care with their default values though
            #Some labels for the folder name and calculation name
            directory_name = 'NitrogenDefectedDraphene', sl='NDG', sm = 'NDG',
            
            # siesta basis, k-point sampling, "mpi" defaults to "mpirun " but we set it to nothing
            # if you do mpirun, remember the space after -> "mpirun " <- because this is what is put into the os.system command
            basis = 'SZP', kp = [3,3,1], mpi = '', 
            
            # pseudopotential path to folder, defaults to "../pp" path relative to where "siesta_python.py" is put
            pp_path = 'pp',
            
            # DFT exchange-correlation functional, defaults to 'gga'
            xc = 'gga',
            
            #Overwrites any data in directory_name folder. Defaults to false, but can be True, False and 'reuse'
            overwrite = True, 
            )

# Write fdf file
Sheet.fdf()
Sheet.run_siesta_in_dir()
```
Now you have your electronic structure of the structure we made in the "NitrogenDefectedGraphene" folder, and you can e.g read the Hamiltonian into python using  [sisl](http://zerothi.github.io/sisl/docs/latest/index.html). See the [sisl](http://zerothi.github.io/sisl/docs/latest/index.html) documentation, it is really easy: 
```
#Hamiltonian and overlap matrices gets loaded:
HS = sisl.get_sile(Sheet.dir + '/' + Sheet.sl + '.TSHS').read_hamiltonian()
```
Which can be used as any other tight-binding model, but has been calculated with the Siesta-method!
# TranSiesta Transport calculation
Here we go through a sisl tutorial, but using just python calls. We do a four-terminal calculation on a system consisting of two crossing 1D chains of carbon atoms. The way to build two-electrode systems is completely analogous. Lets start:
### The electrodes, build simply with sisl
```
import numpy as np
from siesta_python import SiP
import sisl
import matplotlib.pyplot as plt

chain = sisl.Geometry([[0,0,0]], atoms=sisl.Atom[6], sc=[1.4, 1.4, 11])

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
```
Now we can take the structures from these and put it into the siesta_python code:
```

EMX = SiP(emx.cell, emx.xyz, emx.toASE().numbers,
          mpi = '', #<-- single process, delete this line / write 'mpirun ' if you want to use mpirun
          directory_name = 'EMX', sl = 'EMX', sm = 'EMX',
          kp = [20,3,1], semi_inf = '-a1', overwrite = True,
          pp_path = 'pp'
          )
EPX = SiP(epx.cell, epx.xyz, epx.toASE().numbers,
          mpi = '',
          directory_name = 'EPX', sl = 'EPX', sm = 'EPX',
          kp = [20,3,1], semi_inf = '+a1', overwrite = True,
          pp_path = 'pp'
          )

EMY = SiP(emy.cell, emy.xyz, emy.toASE().numbers,
          mpi = '',
          directory_name = 'EMY', sl = 'EMY', sm = 'EMX',
          kp = [3,20,1], semi_inf = '-a2',  overwrite = True,
          pp_path = 'pp'
          )

EPY = SiP(epy.cell, epy.xyz, epy.toASE().numbers,
          mpi = '',
          directory_name = 'EPY', sl = 'EPY', sm = 'EPY',
          kp = [3,20,1], semi_inf = '+a2',  overwrite = True,
          pp_path = 'pp'
          )

elecs = [EMX, EPX, EMY, EPY]
for e in elecs: e.fdf(); e.run_siesta_in_dir()
```
Now, we build the Scattering region calculation:
``` 
def buffer_atoms(x):
    if (x[0:2] <  2.5).any() or (x[0:2] > 20).any():
        return True
    return False

Dev = SiP(device.cell, device.xyz, device.toASE().numbers,
          pp_path = 'pp', 
          mpi = '',
          directory_name = 'Device', solution_method = 'transiesta',
          kp = [3,3,1], overwrite = True,
          kp_tbtrans = [1,50,1],
          elecs = elecs, 
          Voltage = 0.0,  Chem_Pot = [0.0, 0.0, 0.0, 0.0]
          )

Dev.find_elec_inds(tol = 1e-2)
Dev.set_buffer_atoms(buffer_atoms)
```



