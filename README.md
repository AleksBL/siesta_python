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

I you look carefully in the folders, there are five files: KP.fdf, STRUCT.fdf, DEFAULT.fdf, TS_TBT.fdf and RUN.fdf -  these are the files written by the self.fdf function. In the funcs.py script the templates for these files are, and you can add your own attributes it should write by adding them in the SiP class and making it write it in funcs.py


# TranSiesta Transport calculation
Here we go through a [sisl tutorial](https://github.com/zerothi/ts-tbt-sisl-tutorial), using just python. We do a four-terminal calculation on a system consisting of two crossing 1D chains of carbon atoms. The way to build two-electrode systems is completely analogous. Lets start:
### The electrodes, built simply with sisl
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
Now, we build the Scattering region calculation and use the methods "self.find_elec_inds", and "self.set_buffer_atoms(func)" to get the relevant indecies for the electrodes and buffer atoms
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
Next, we write the required fdf file, run the Analyze step, which writes the minimum memory pivotting scheme to the TS_TBT.fdf file, and run TranSiesta & tbtrans:

```
Dev.fdf()
Dev.write_more_fdf(['TS.Hartree.Fix +A'], name = 'TS_TBT')
Dev.run_analyze_in_dir()
Dev.run_siesta_in_dir()
Dev.run_tbtrans_dir(DOS_GF = True)
```
The above is a equillibrium calculation. We can use the density matrix from this calculation to make a bias calculation.  This is done by using the "self.copy_DM_from" method: 
```

Dev2 = SiP(device.cell, device.xyz, device.toASE().numbers,
          pp_path = 'pp', 
          mpi = '',
          directory_name = 'Device2', solution_method = 'transiesta',
          kp = [3,3,1], overwrite = True,
          kp_tbtrans = [1,50,1],
          elecs = elecs, 
          reuse_dm = True, # <<----- We give the code the flag to reuse 
                           #         a density matrix, this we will copy 
                           #         from the previous Dev-calculation
          Voltage = 2 * 0.26,  Chem_Pot = [0.26, 0.24, -0.26, -0.24],
          NEGF_calc = True)

Dev2.find_elec_inds(tol = 1e-2)
Dev2.set_buffer_atoms(buffer_atoms)
Dev2.fdf()
Dev2.write_more_fdf(['TS.Hartree.Fix +A'], name = 'TS_TBT')

Dev2.copy_DM_from(Dev) # <<------- Copying density matrix from the previous calculator object 

Dev2.run_analyze_in_dir()
Dev2.run_siesta_in_dir()
Dev2.run_tbtrans_dir(DOS_GF = True)
```
Yet again we can use sisl to read the transmission function:
```
t = sisl.get_sile(Dev2.dir + '/siesta.TBT.nc')
plt.plot(t.E,t.transmission(0,2))
```
## Getting the Green's function of the system from the TranSiesta Calculation:
Here we use the Transport_DCAC code, which also  relies in the Block_matrices code. Your Folder should look something like this if you have downloaded the various codes from the different repositories:
![image](https://user-images.githubusercontent.com/75378674/121400029-5c21f880-c957-11eb-8f51-101a1264ba9b.png)

We do the same steps as previously and do a four-terminal calculation:

```

import numpy as np
from siesta_python import SiP
from Transport_DCAC import System
import sisl
import matplotlib.pyplot as plt

chain = sisl.Geometry([[0,0,0]], atoms=sisl.Atom[6], sc=[1.4, 1.4, 11])

elec_x = chain.tile(4, axis=0).add_vacuum(11 - 1.4, 1)
elec_y = chain.tile(4, axis=1).add_vacuum(11 - 1.4, 0)

chain_x = elec_x.tile(12, axis=0)
chain_y = elec_y.tile(12, axis=1)

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

EMX = SiP(emx.cell, emx.xyz, emx.toASE().numbers,
          #mpi = '', #<-- single process, delete this line / write 'mpirun ' if you want to use mpirun
          directory_name = 'EMX', sl = 'EMX', sm = 'EMX', basis = 'SZ',
          kp = [50,1,1], semi_inf = '-a1', overwrite = True,
          pp_path = 'pp'
          )

EPX = SiP(epx.cell, epx.xyz, epx.toASE().numbers,
          #mpi = '',
          directory_name = 'EPX', sl = 'EPX', sm = 'EPX', basis = 'SZ',
          kp = [50,1,1], semi_inf = '+a1', overwrite = True,
          pp_path = 'pp'
          )

EMY = SiP(emy.cell, emy.xyz, emy.toASE().numbers,
          #mpi = '',
          directory_name = 'EMY', sl = 'EMY', sm = 'EMX',basis = 'SZ',
          kp = [1,50,1], semi_inf = '-a2',  overwrite = True,
          pp_path = 'pp'
          )

EPY = SiP(epy.cell, epy.xyz, epy.toASE().numbers,
          #mpi = '',
          directory_name = 'EPY', sl = 'EPY', sm = 'EPY',basis = 'SZ',
          kp = [1,50,1], semi_inf = '+a2',  overwrite = True,
          pp_path = 'pp'
          )

elecs = [EMX, EPX, EMY, EPY]
for e in elecs: e.fdf(); e.run_siesta_in_dir()
E = []
for e in elecs:
    h = sisl.get_sile(e.dir + '/' + e.sl + '.TSHS').read_hamiltonian()
    s = sisl.get_sile(e.dir + '/' + e.sl + '.TSHS').read_overlap()
    h.set_nsc((3,3,1))
    s.set_nsc((3,3,1))
    
    E += [{'H': h, 'S':s}]
    print(h.nsc, s.nsc)

def buffer_atoms(x):
    if (x[0:2] <  2.5).any() or (x[0:2] > 64).any():
        return True
    return False

Dev = SiP(device.cell, device.xyz, device.toASE().numbers,
          pp_path = 'pp', 
          #mpi = '',
          directory_name = 'Device', solution_method = 'transiesta',
          kp = [1,1,1], overwrite = True,
          kp_tbtrans = [1,50,1],
          basis = 'SZ',
          trans_emin = -0.5, trans_emax = 0.5, trans_delta = 0.05,
          elecs = elecs, 
          Voltage = 0.0,  Chem_Pot = [0.0, 0.0, 0.0, 0.0]
          )


Dev.find_elec_inds(tol = 1e-2)
Dev.set_buffer_atoms(buffer_atoms)
Dev.fdf()
Dev.write_more_fdf(['TS.Hartree.Fix -A'], name = 'TS_TBT')

Dev.run_analyze_in_dir()
Dev.run_siesta_in_dir()
Dev.run_tbtrans_dir()

H   = sisl.get_sile(Dev.dir + '/siesta.TSHS').read_hamiltonian()
S   = sisl.get_sile(Dev.dir + '/siesta.TSHS').read_overlap()
H.set_nsc((3,3,1))
S.set_nsc((3,3,1))

D = {'H': H, 'S':S}
```


Hopefully, the above code finishes. If you read through, we have taken out the device and electrode Hamiltonians and overlaps in the "D" and "E" dictionaries. TBtrans furthermore makes the bandwidth of the inverse Greens function smaller. We read this information:
```
tbt = sisl.get_sile(Dev.dir + '/siesta.TBT.nc')
btd = tbt.btd().copy()
pivot = tbt.pivot().copy()
```
Here we have the information on how to permute the columns and rows of the inverse Greens function to make the sparsity pattern favorable in "btd" and furthermore we have how to permute the columns and rows in "pivot". We now set up the inverse Greens function and the Gamma matrices:

```
Eg = np.linspace(-0.5, 0.5, 100)
Calc = System(D, E, Dev.elec_inds, Eg, 0.0, Eg[1]- Eg[0], buffer_inds = Dev.buffer_atoms, pivot = pivot, eta = 1e-2)

Calc.Set_kp([None]); nk = 1 # "1" k-point
Calc.Organise_and_Check()
Calc.Gen_SE_decimation(dirs = [(-1,0), (1,0), (0,-1), (0,1)])
P = [0]; 
for b in btd: P += [P[-1] + b]
Calc.Block_Setup_decimation( P = P )

iG  = Calc.iGreens
Gammas = Calc.Gammas
Gl = Gammas[0]
Gr = Gammas[2]
```
### Inspect Gammas and their "self.is_zero" block matrix, where their nonzero blocks are set to 1
```
G = iG.Invert(BW = '*\*')

M1 = Gl.BDot(G)
G.do_dag()
M2 = Gr.BDot(G)
G.do_dag()


##Tr(Gl * G * Gr * G^T*)
Transmission = M1.TrProd(M2).sum(axis=0)/nk

plt.plot(tbt.E, tbt.transmission(0,2))
plt.plot(Eg, Transmission)

```



