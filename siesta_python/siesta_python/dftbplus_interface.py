#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:20:17 2023

@author: aleksander
"""

import numpy as np
import matplotlib.pyplot as plt
import sisl
from scipy.linalg import eigh
from numba import njit,jit
from numba.typed import List
#from time import time, sleep
import scipy.sparse as sp
from time import time
from io import StringIO
import re

H2eV     = 27.211396641308
eV2H     = 1/H2eV
BOHR__AA = 0.529177249
AA__BOHR = 1 / BOHR__AA
#@jit
def loadtxt(file):
    lines = open(file,'r').readlines()
    NS    = 0
    cond  = True
    while cond:
        if "IATOM1" in lines[NS]  and "INEIGH" in lines[NS] and "IATOM2F" in lines[NS]:
            break
        NS    += 1
    header     = lines[0:NS]
    lines      = lines[NS:]
    lines2     = []
    inf        = np.genfromtxt(header[2:],dtype=int)
    H_ele      = []
    readH      = False
    hele       = []
    for i in range(len(lines)):
        if 'IATOM1' in lines[i] and 'INEIGH' in lines[i]:
            lines2  += [''+lines[i+1]]
            readH    = False
            if i>0:
                # arr     = np.genfromtxt(hele)
                # arr     = np.loadtxt(hele)
                arr = np.array([line.strip().split() for line in hele],np.float64)
                arrs    = arr.shape
                if arrs==():
                    arr = arr[None,None]
                if len(arrs)==1 and len(hele)>1:
                    arr = arr[:, None]
                elif len(arrs)==1 and len(hele)==1:
                    arr = arr[None,:]
                H_ele.append(arr)
        if readH == True:
            hele.append(lines[i])#re.sub(r'.......E', 'E', lines[i]))
        if '# MATRIX' in lines[i]:
            readH = True
            hele  = []
    
    arr = np.genfromtxt(hele)
    if arr.shape==():
        arr = arr[None,None]
    if len(arr.shape)==1 and len(hele)>1:
        arr = arr[:, None]
    elif len(arr.shape)==1 and len(hele)==1:
        arr = arr[None,:]
    
    H_ele .append(arr)
    R      = np.array([line.strip().split() for line in lines2],int)
    R      = R[:,[0,2,3,4,5]].copy()
    H_ele  = List(H_ele)
    return H_ele, R, inf

@njit
def isequiv(ci, cj):
    if ci[0] == cj[1] and ci[1] == cj[0] and np.abs(ci[2:]+cj[2:]).sum()<1e-10:
        return True
    return False
@njit
def has_equiv_in_l(L):
    equiv = np.zeros(len(L))
    for i,li in enumerate(L):
        for j,lj in enumerate(L):
            if isequiv(li,lj):
                equiv[i] = 1
                break
    return equiv

def readband(file):
    lines = open(file,'r').readlines()
    for i in range(len(lines)):
        if 'KPT' in lines[i]:
            lines[i] = "#" + lines[i]
        if len(lines[i])==0:
            lines[i] = "#" + lines[i]
    return np.genfromtxt(lines)

@njit(fastmath=True)
def add_to_slice(m,val,row,col):
    for i,I in enumerate(range(row[0],row[1])):
        for j,J in enumerate(range(col[0],col[1])):
            m[I,J]+= val[i,j]

@njit(parallel = False, fastmath = True)
def constructHk(k,HR, R, i1, equiv, gamma_only = False):
    mj2pi = -1j*2*np.pi
    slices = [np.array([-1,-1])]
    na     = len(i1)
    orbc   = 0
    for i in range(na):
        slices += [np.array([orbc, orbc + i1[i,-1]])]
        orbc   += i1[i,-1]
    slices = slices[1:]
    Hk        = np.zeros((orbc, orbc), dtype=np.complex128)
    nHR = len(HR)
    zrs = np.zeros(3)
    for i in range(nHR):
        m          = HR[i]
        inf        = R[i]
        s1,s2      = inf[0]-1, inf[1]-1
        ruc        = inf[2:]
        phase      = np.exp(mj2pi*(k*ruc).sum())
        if not (np.abs(ruc- zrs)<1e-10).all() and gamma_only:
            continue
        
        hsub       = m*phase
        add_to_slice(Hk, hsub, slices[s1],slices[s2])
        if equiv[i]==0:
            add_to_slice(Hk, hsub.conj().T, slices[s2],slices[s1])
    return Hk

def construct_sparse(HR, R, i1, equiv, geom_sc, tol=1e-8,gamma_only = False,Roff = None):
    slices = []
    na     = len(i1)
    orbc   = 0
    for i in range(na):
        slices += [slice(orbc, orbc + i1[i,-1])]
        orbc   += i1[i,-1]
    nuc = np.prod(geom_sc.nsc)
    if gamma_only:
        ele = sp.lil_matrix((orbc, orbc))
    else:
        ele = sp.lil_matrix((orbc, nuc*orbc))
    #Iv,Jv,Vv = np.zeros((0,)), np.zeros((0,)),np.zeros((0,))
    zrs = np.zeros(3)
    for i,m in enumerate(HR):
        inf        = R[i]
        s1,s2      = inf[0]-1, inf[1]-1
        ruc        = inf[2:]
        
        if not np.allclose(ruc, zrs) and gamma_only:
            continue
        I,J        = np.where(np.abs(m)>tol)
        v          = m[I,J]
        I         += slices[s1].start
        J         += slices[s2].start
        iuc        = geom_sc.sc_index(ruc)
        J         += iuc*orbc
        #Iv         = np.hstack((Iv,I))
        #Jv         = np.hstack((Jv,J))
        #Vv         = np.hstack((Vv,v))
        ele[I,J] = v
        if equiv[i]==0:
            mT  = m.T
            I,J = np.where(np.abs(mT)>tol)
            v   = mT[I,J]
            I  += slices[s2].start
            J  += slices[s1].start
            iuc = geom_sc.sc_index(-ruc)
            J  += iuc*orbc
            ele[I,J] = v
            #Iv  = np.hstack((Iv,I))
            #Jv  = np.hstack((Jv,J))
            #Vv  = np.hstack((Vv,v))
            
    #ele[Iv,Jv]  = Vv
    return ele.tocsr()

def read_mulliken_from_detailedout(file):
    lines = open(file,'r').readlines()
    s   = []
    Sh  = []
    L   = []
    m   = []
    Pop = []
    label=[]
    read = False
    for i,l in enumerate(lines):
        if "Orbital populations" in lines[i-2] and "Atom" in lines[i-1] and "Sh." in lines[i-1]:
            read = True
        if len(l)==1:
            read = False
        if read:
            spl = l.replace('\n','').split()
            s. append(int(spl[0]))
            Sh.append(int(spl[1]))
            L. append(int(spl[2]))
            m. append(int(spl[3]))
            Pop.append(float(spl[4]))
            label.append(spl[5])
    s = np.array(s).astype(int)
    Sh= np.array(Sh).astype(int)
    L = np.array(L).astype(int)
    m = np.array(m).astype(int)
    Pop = np.array(Pop)
    res = {'s':s,
           'Shell':Sh,
           'l':L,
           'm':m,
           'pop':Pop,
           'label':label}
    return res

# for i in range(10):
#     H_ele, R, inf = loadtxt('/home/aleksander/Desktop/Calculations/CLEAN_TD/dftb_rib/Dev/hamreal1.dat')
# #     #eq =  has_equiv_in_l(R)
