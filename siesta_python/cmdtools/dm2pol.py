#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:39:05 2022

@author: aleksander
"""


from siesta_python.DFTOrbitals import DFTOrbitals
import sisl
import joblib as jl
import sys
import os
#from siesta_python.funcs import unique_list
import numpy as np
from numba import njit, prange
from time import time
import TimedependentTransport.HartreeFromDensity as HFD
import TimedependentTransport.mpi_tools as MT


args = sys.argv[1:]
print('dm2pol Program Start')
print('sys.argv:', sys.argv)
print('')

options = {
           'DirName': 'noname',
           'UT_file':'none',
           'nprocs': '4',
           'verbose': '1',
           'backend': 'multiprocessing',
           'OutFile': 'polarisation',
           'Nrad': '10',
           'dt': '0.1',
           'dx': '0.0',
           }

for a in args:
    split = a.replace(' ', '').split('=')
    options.update({split[0]:split[1]})


DirName = options['DirName']
    



path       = args[0].replace(' ', '').split('=')[1]
os.chdir(path)
new_folder = os.getcwd()
sys.path.append(new_folder)
hsfile = None

for f in os.listdir(DirName):
    if '.TSHS' in f:
        hsfile = f
        break

if hsfile is None:
    print('No HS-file found. Run a siesta/transiesta calculation.')
    exit()

HS  = sisl.get_sile(DirName+'/'+hsfile).read_hamiltonian()
structfile = sisl.get_sile(DirName+'/'+'STRUCT.fdf').read_geometry()


Orbitals = DFTOrbitals(DirName, np.array(structfile.atoms.Z), N = int(options['Nrad']))
Orbitals.set_pos(structfile.xyz.copy())

t, DM = MT.combine_dm([options['DirName']])
if options['UT_file']=='none': U = np.eye(DM.shape[-1])
else: U = np.load(options['UT_file'])
DM = U@DM@U

tidx = []
ti = t.min()
dt = float(options['dt'])
while ti < t.max():
    diff = np.abs(t - ti)
    tidx += [np.where(diff == diff.min())[0][0]]
    ti += dt






