# siesta_python
Siesta, tbtrans...... without touching the terminal


### But how does it work?
Firstly, its not magic, you need to have siesta, transiesta and tbtrans compiled on your computer and you need to know their paths.
Favorably you can have them in your .bashrc file so that you can just call them as "siesta RUN.fdf > RUN.out", "tbtrans RUN.fdf > RUN.out", but it isnt strictly necessary, as you can also just give the code your paths to the various executables. Just "siesta" and "tbtrans" are however default values. 

## How to set up a siesta-calculation
We can make the atomic structure using [sisl](http://zerothi.github.io/sisl/docs/latest/index.html) (Which can also do a lot of other stuff)

