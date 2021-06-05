# siesta_python
Siesta, tbtrans...... without touching the terminal


### But how does it work?
Firstly, its not magic, you need to have siesta, transiesta and tbtrans compiled on your computer and you need to know their paths.
Favorably you can have them in your .bashrc file so that you can just call them as "siesta RUN.fdf > RUN.out", "tbtrans RUN.fdf > RUN.out", but it isnt strictly necessary, as you can also just give the code your paths to the various executables. Just "siesta" and "tbtrans" are however default values. 

## How to set up a siesta-calculation
Copy the "siesta_python.py" and "funcs.py" scripts to the folder of your choice, together with a folder with the pseudo-potentials you will use. The pseudo-potentials must be named like this: 'atom'.'type'.psf, eg. for the molybdenum gga pseudo-potential would read Mo.gga.psf or the carbon lda: C.lda.psf. We will later reference this folder in the initialisation of the class contained in 'siesta_python.py'. Now we are set up, as the siesta_python.py script will handle the creation of all the folders.

Before starting, we should have a folder called "pp" with the pseudo-potentials in the same folder you are running the calculation in. You should also be able to run siesta from the terminal using just "siesta RUN.fdf > RUN.out", else you will need to additionally specify the "siesta_exec" keyword in the 


We can make the atomic structure using [sisl](http://zerothi.github.io/sisl/docs/latest/index.html) (Which can also do a lot of other stuff). Lets make a simple graphene sheet with a nitrogen defect in the middle: 
```
import numpy as np
from siesta_python import SiP
import sisl

Sheet = sisl.geom.graphene(orthogonal = True).tile(2,0).tile(2,1)

pos = Sheet.xyz
cell = Sheet.cell
species = Sheet.toASE().numbers
#make fourth atom nitrogen
species[3] = 7

Sheet = SiP(cell, pos, species, 
            #Not necessarily needed keywords below, care with their default values through
            #Some labels for the folder name and calculation name
            directory_name = 'NitrogenDefectedDraphene', sl='NDG', sm = 'NDG',
            # siesta basis, k-point sampling, "mpi" defaults to "mpirun " but we set it to nothing
            # if you do mpirun, remember the space after -> "mpirun " <- because this is what is put into the os.system command
            basis = 'SZP', kp = [3,3,1], mpi = '', 
            # pseudopotential path to folder, defaults to "../pp" path relative to where "siesta_python.py" is put
            pp_path = 'pp',
            # DFT exchange-correlation functional
            xc = 'gga',
            #Overwrites any data in directory_name folder. Defaults to false, but can be True, False and 'reuse'
            overwrite = True, 
            )

# Write fdf file
Sheet.fdf()
Sheet.run_siesta_in_dir()


```


