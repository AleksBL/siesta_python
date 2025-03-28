#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


fake classes to make projections possible
and possibly circumvent TBTrans


Created on Tue Sep 20 13:11:58 2022

@author: aleksander
"""
import numpy as np
import scipy.sparse as sp


class fakeTBT:
    # Fake class for emulating/parsing TBtrans / sisl output
    # not in any way exhausting the different things you
    # can find in the actual TBTsile class
    # the info is just stored in the npz file you initialise it with
    # 
    # meant for use in conjunction with the siesta_python.eigenstate_projection
    # 
    
    def __init__(self, fakeTBT):
        assert '.fakeTBT.' in fakeTBT
        file = np.load(fakeTBT)
        self.npz = file
        self.Imposter = True
    
    def transmission(self, kavg = None):
        if kavg is None: return self.npz['transmission']
        else:          return self.npz['_tbtTk_full'][kavg]
    @property
    def wkpt(self): return self.npz['wkpt']
    @property
    def E(self):    return self.npz['E']
    def pivot(self):return self.npz['pivot']
    def btd(self):  return self.npz['btd']
    def real_pivot(self): return self.npz['real_pivot']
    def real_btd(self):   return self.npz['real_btd']
    @property
    def k(self):    return self.npz['kv']
    def read_fermi_level(self): return self.npz['E_F']


class fakeHS:
    # Meant to be able to emulate a sisl H.Hk(k=k)/S.Sk(k=k) call
    # Probably slow as salty snail, its meant for intermediate linking, not performance
    # No checks on actual matrix elements, when ever or not its actually is a valid overlap
    # matrix / Hamiltonian
    # 
    def __init__(self, Mat, k):
        # Mat: (nk, no, no) np.ndarray
        # k  : (nk ,3)      np.ndarray
        self.M = Mat
        self.k = k
        self.no= Mat.shape[-1]
        self.spin = dummyspin()
        self.Imposter = True

    def Hk(self, k, spin = 0):
        dk = np.linalg.norm(self.k - k,axis=1)
        idx = np.where(dk == dk.min())[0][0]
        return sp.csr_matrix(self.M[idx])
    def Sk(self, k, spin = 0):
        dk = np.linalg.norm(self.k - k,axis=1)
        idx = np.where(dk == dk.min())[0][0]
        return sp.csr_matrix(self.M[idx])
    @property
    def shape(self):
        return self.M.shape
    

class dummyspin:
    def __init__(self):
        self.name = 'I AM DUMMYSPIN'
        self.Imposter = True
    def __str__(self):
        return 'unpolarized'

def read_fakeSE_from_tbtrans(file):
    assert 'fakeTBT.SE' in file
    # Pendent to read_SE_from_tbtrans found in Gf_Module.Gf file
    # The fakeTBT.SE files are just dense numpy arrays of n_eig dimension,
    # no matrix elements are left out
    
    npz = np.load(file)
    if 'Mode2' not in npz.files:
        SE = npz['SE']
        _inds = npz['inds']
        SE = [se for se in SE]
        inds = [_inds.copy() for _ in SE]
        return SE, inds
    else:
        cond = True
        it   = 0
        SE = []
        inds = []
        while cond:
            try:
                SE   += [npz['SE_' + str(it)] ]
                inds += [npz['inds_' + str(it)] ]
                it+=1
            except:
                cond = False
            
        return SE,inds

        
        

