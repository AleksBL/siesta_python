""""
Code for finding the coulomb integrals that need evaluation.
using the symmetries of the 2-electron integrals

"""

from numba import njit
import numpy as np




@njit
def op1(t):
    t1,t2,t3,t4 = t
    return (t2,t1,t3,t4)
@njit
def op2(t):
    t1,t2,t3,t4 = t
    return (t1,t2,t4,t3)

@njit
def op3(t):
    t1,t2,t3,t4 = t
    return (t3,t4,t1,t2)

@njit
def unique_list(l):
    o=[]
    for i in l:
        if i not in o:
            o.append(i)
    return o

@njit
def composed(t,n):
    if n==0:
        return op1(t)
    elif n==1:
        return op2(t)
    elif n==2:
        return op3(t)


@njit
def equiv(t):
    eq_l = [(-1,-1,-1,-1)]
    nops = 3
    for i in range(nops):
        for j in range(nops):
            for k in range(nops):
                eq_l += [composed(composed(composed(t,k),j),i)]

    return unique_list(eq_l)[1:] + [t]


# def reducewithtranslationsymmetry(inds, types, Pos,eps = 1e-3):
#     """
#     inds: output from the find algo in FourCenterIntegrals, i.e list of [(i,j,k,l)]
#     types:  list of tuples of (atom type, orbital) specifying the atomic orbital 
#            type, length large as number of different i.
#     Pos  : position of each orbital. 
    
#     Coulomb integrals are invariant to rigid shift of all the orbitals.
#     we use this fact to reduce computational load
    
#     Here we find which indices in the inds list are actually equal
#     """
#     ni       = len(inds)
#     isunique = np.zeros(ni, dtype = int)
#     equivto  = np.zeros(ni, dtype = int)
#     Info_ij  = np.zeros((ni,7))
#     Info_kl  = np.zeros((ni,7))
#     R_ij_kl  = np.zeros((ni,3))
#     for i in range(ni):
#         ri,rj          = Pos[inds[i][0]], Pos[inds[i][1]]
#         Info_ij[i,0:3] = ri - rj
#         rk,rl          = Pos[inds[i][2]], Pos[inds[i][3]]
#         Info_kl[i,0:3] = rk - rl
        
#         Info_ij[i,3:5] = types[inds[i][0]]
#         Info_ij[i,5:7] = types[inds[i][1]]
        
#         Info_kl[i,3:5] = types[inds[i][2]]
#         Info_kl[i,5:7] = types[inds[i][3]]
        
#         R_ij_kl[i,0:3] = (ri+rj)/2 - (rk+rl)/2
    
    
    
    
    
    



