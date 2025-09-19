#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:34:03 2022

@author: aleksander
"""
import numpy as np

def Alt_real_space_coupling(self, ret_indices=False):
    r""" Real-space coupling surafec where the outside fold into the surface real-space unit cell

    The resulting parent object only contains the inner-cell couplings for the elements that couple
    out of the real-space matrix.

    Parameters
    ----------
    ret_indices : bool, optional
       if true, also return the atomic indices (corresponding to `real_space_parent`) that encompass the coupling matrix

    Returns
    -------
    parent : object
        parent object only retaining the elements of the atoms that couple out of the primary unit cell
    atom_index : numpy.ndarray
        indices for the atoms that couple out of the geometry, only if `ret_indices` is true
    """
    k_ax = self._k_axes
    unfold   = self._unfold
    PC_semi = self.semi.spgeom1.copy()
    PC = self.surface.copy()
    for ax in range(3):
        PC      = PC.tile(self._unfold[ax], ax)
        PC_semi = PC_semi.tile(self._unfold[ax], ax)
    
    atom_idx = []
    for i in range(PC.na):
        if (np.linalg.norm(PC.xyz[i] - PC_semi.xyz, axis = 1)<0.001).any():
            atom_idx+=[i]
    atom_idx = np.array(atom_idx)
    no = PC.no
    assert (PC.nsc>1).sum() ==1
    assert (PC.nsc[PC.nsc>1]==3).all()
    
    Vkax1    = PC._csr.tocsr(0)[:,no:2*no].toarray()
    Vkax2    = PC._csr.tocsr(0)[:,2*no:3*no].toarray()
    idx1     = PC.o2a(np.unique(np.where(Vkax1!=0.)[0]))
    idx2     = PC.o2a(np.unique(np.where(Vkax2!=0.)[0]))
    U        = np.union1d
    atom_idx = U(atom_idx,U(idx1, idx2))
    
    PC = PC.sub(atom_idx)
    # Remove all out-of-cell couplings such that we only have inner-cell couplings.
    PC.set_nsc((1,1,1))
    
    if ret_indices:
        return PC, atom_idx
    return PC
