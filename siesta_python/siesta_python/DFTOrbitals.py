#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:45:04 2022

@author: aleksander
"""
from sisl.io.siesta.basis import ionncSileSiesta
import os
import numpy as np
import becke
from time import time
from scipy.integrate import tplquad
#import quadpy
from Zandpack.HartreeFromDensity import make_density,matrixelementsoffield#, make_density_jit



ld = os.listdir


class DFTOrbitals:
    # you can probably toggle offf the force_uneven keyword.
    # its meant for when grids needs to be matched in the make_density,
    # and matrixelementsoffield Routines
    # It'll often throw an error and you need to tune
    # the vacuum and N keywords. Good luck.....
    
    
    def __init__(self, Dir, s, N = 30, vacuum = 1.0, force_uneven = True):
        
        
        Nr = 2 * N+1
        
        files = ld(Dir)
        files = [file for file in files if '.ion.nc' in file]
        self.files = files
        B = [ionncSileSiesta(Dir + '/'+f).read_basis() for f in files]
        #print(B)
        self.ionncfiles = B
        self.dx = max([B[i].maxR()/Nr for i in range(len(B))])
        self.vacuum = vacuum
        
        orbs   = []
        labels = []
        funcs  = []
        for a in range(len(B)):
            Aorbs  = []
            Alabels= []
            Afuncs = []
            R = B[a].maxR() + self.vacuum
            Rs = np.arange(-R,R,self.dx)
            if force_uneven:
                assert len(Rs)%2 == 1
            x,y,z    = np.meshgrid(Rs,Rs,Rs,indexing = 'ij')
            Rs       = np.hstack([x.ravel()[:,None],y.ravel()[:,None],z.ravel()[:,None]])
            
            for io,o in enumerate(B[a].orbitals):
                Aorbs  += [o.psi(Rs).reshape(x.shape)]
                Alabels+= [(B[a].Z,io)]
                Afuncs += [o.psi]
            orbs   += [Aorbs  ]
            labels += [Alabels]
            funcs  += [Afuncs ]
        self.s        = s
        self.orbitals = orbs
        self.labels   = labels
        self.funcs    = funcs
        #print(labels)
        self.i2o()
        self.pos = None
        
    def set_pos(self, Ri):
        self.pos = Ri
    
    def get_pos(self,i):
        return self.pos[self.i2a(i)]
    
    
    
    def get_orbitals(self, Z):
        for ia in range(len(self.orbitals)):
            if self.labels[ia][0][0] == Z:
                return self.orbitals[ia], ia
            
    def get_no(self):
        no = 0
        for i in range(len(self.s)):
            orb = self.get_orbitals(self.s[i])[0]
            no += len(orb)
        return no
    
    def i2a(self,j):
        no = 0
        for i in range(len(self.s)):
            no += len(self.get_orbitals(self.s[i])[0])
            if no>j:
                return i
    
    def i2o(self):
        no = self.get_no()
        pairs = []
        counter = 0
        for i, n in enumerate(self.s):
            orbs,ia = self.get_orbitals(n)
            for io in range(len(orbs)):
                pairs += [(ia, io)]
        self.pairs = pairs
    
    def orbital_on_grid(self,i):
        return self.orbitals[self.pairs[i][0]][self.pairs[i][1]]
    
    def get_func(self, i):
        return self.funcs[self.pairs[i][0]][self.pairs[i][1]]
    
    def ravelled_orbs_and_inds(self):
        o_list = []
        label_list = []
        
        for ia,a in enumerate(self.orbitals):
            for io,o in enumerate(a):
                o_list += [o]
                label_list += [(ia,io)]
        i_list = []
        for i in range(self.get_no()):
            i_list += [label_list.index(self.pairs[i])]
            
        return o_list, i_list
    
    def becke_electron_repulsion(self, i,j,k,l,eps = 1e-5, Time = False):
        t1 = time()
        
        v  = eps*np.ones(3)
        Ri = self.get_pos(i)-2*v
        Rj = self.get_pos(j)-  v
        Rk = self.get_pos(k)+  v
        Rl = self.get_pos(l)+2*v
        
        def fi(x,y,z):
            s = x.shape#.copy()
            r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
            return self.get_func(i)(r).reshape(s)
        def fj(x,y,z):
            s = x.shape#.copy()
            r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
            return self.get_func(j)(r).reshape(s)
        def fk(x,y,z):
            s = x.shape#.copy()
            r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rk
            return self.get_func(k)(r).reshape(s)
        def fl(x,y,z):
            s = x.shape#.copy()
            r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rl
            return self.get_func(l)(r).reshape(s)
        atoms = [(1, Ri),
                 (1, Rj),
                 (1, Rk),
                 (1, Rl)]
        val = becke.electron_repulsion(atoms, fi, fj, fk, fl)
        t2 = time()
        if Time:
            print('Fourcenter integral evaulated in : ',t2-t1, 's')
            
        return val
    
    def becke_overlap(self,i,j, eps =1e-5 ):
        v  = eps*np.ones(3)
        Ri = self.get_pos(i)-0.5*v
        Rj = self.get_pos(j)+0.5*v
        
        def fi(x,y,z):
            s = x.shape#.copy()
            r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
            return self.get_func(i)(r).reshape(s)
        def fj(x,y,z):
            s = x.shape#.copy()
            r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
            return self.get_func(j)(r).reshape(s)
        atoms = [(1, Ri),
                 (1, Rj),
                ]
        
        return becke.overlap(atoms, fi,fj)
    
    def becke_electronic_dipole(self, i,j, eps = 1e-5, U = None):
        v  = eps*np.ones(3)
        Ri = self.get_pos(i)-0.5*v
        Rj = self.get_pos(j)+0.5*v
        if U is None:
            
            def fi(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
                return self.get_func(i)(r).reshape(s)
            def fj(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
                return self.get_func(j)(r).reshape(s)
            atoms = [(1, Ri),
                     (1, Rj),
                     ]
        else:
            Ud = U.conj().T
            def fi(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
                res = np.zeros(x.ravel().shape)
                for uij in Ud[i]:
                    res += uij*self.get_func(i)(r)
                return res.reshape(s)
            def fj(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
                res = np.zeros(x.ravel().shape)
                for uij in U[j]:
                    res += uij*self.get_func(j)(r)
                return res.reshape(s)
            atoms = [(1, Ri),
                     (1, Rj),
                     ]
        
        return becke.electronic_dipole(atoms, fi, fj)
    
    def becke_nuclear(self, i,j, eps = 1e-5, U = None):
        v  = eps*np.ones(3)
        Ri = self.get_pos(i)-0.5*v
        Rj = self.get_pos(j)+0.5*v
        if U is None:
            
            def fi(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
                return self.get_func(i)(r).reshape(s)
            def fj(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
                return self.get_func(j)(r).reshape(s)
            atoms = [(1, Ri),
                     (1, Rj),
                     ]
        else:
            Ud = U.conj().T
            def fi(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
                res = np.zeros(x.ravel().shape)
                for uij in Ud[i]:
                    res += uij*self.get_func(i)(r)
                return res.reshape(s)
            def fj(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
                res = np.zeros(x.ravel().shape)
                for uij in U[j]:
                    res += uij*self.get_func(j)(r)
                return res.reshape(s)
            atoms = [(1, Ri),
                     (1, Rj),
                     ]
        
        return becke.nuclear(atoms, fi, fj)
    
    
    def becke_new_overlap(self, i,j, eps = 1e-5, U = None, vac = 10.0, quadpy_N = None):
        v  = eps*np.ones(3)
        Ri = self.get_pos(i)-0.5*v
        Rj = self.get_pos(j)+0.5*v
        if U is None:
            
            def fi(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
                return self.get_func(i)(r).reshape(s)
            def fj(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
                return self.get_func(j)(r).reshape(s)
            atoms = [(1, Ri),
                     (1, Rj),
                     ]
        else:
            Ud = U.conj().T
            def fi(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Ri
                res = np.zeros(x.ravel().shape)
                for c,uij in enumerate(Ud[i]):
                    res += uij*self.get_func(c)(r)
                return res.reshape(s)
            def fj(x,y,z):
                s = x.shape
                r = np.hstack([x.ravel()[:,None], y.ravel()[:,None], z.ravel()[:,None]])-Rj
                res = np.zeros(x.ravel().shape)
                for c,uij in enumerate(U[j]):
                    res += uij*self.get_func(c)(r)
                return res.reshape(s)
            atoms = [(1, Ri),
                     (1, Rj),
                     ]
        xmin, xmax = min(Ri[0], Rj[0])-vac, max(Ri[0], Rj[0])+vac
        ymin, ymax = min(Ri[1], Rj[1])-vac, max(Ri[1], Rj[1])+vac
        zmin, zmax = min(Ri[2], Rj[2])-vac, max(Ri[2], Rj[2])+vac
        
        def F(z,y,x):
            _x = np.array([x])
            _y = np.array([y])
            _z = np.array([z])
            return fi(_x,_y,_z).conj() * fj(_x,_y,_z)
        def _FF(xyz):
            x,y,z = xyz
            #print(x.shape)
            return fi(x,y,z).conj() * fj(x,y,z)
        if quadpy_N is None:
            return tplquad(F, xmin, xmax, ymin, ymax, zmin, zmax,epsabs = 1e-5, epsrel = 1e-3 )
        else:
            pass
            # if isinstance(quadpy_N,int):
            #     Nx=Ny=Nz=quadpy_N
            # if isinstance(quadpy_N, list):
            #     Nx,Ny,Nz = quadpy_N
            # #scheme = quadpy.t3.get_good_scheme(5)
            
            # scheme = quadpy.c3.product(quadpy.c1.newton_cotes_closed(3))
            # # scheme.show()
            # DX = (xmax - xmin)/Nx; Rx = DX/2
            # DY = (ymax - ymin)/Ny; Ry = DY/2
            # DZ = (zmax - zmin)/Nz; Rz = DZ/2
            # res = 0.0
            # for I in range(Nx):
            #     xC = xmin + I * DX
            #     for J in range(Ny):
            #         yC = ymin + J * DY
            #         for K in range(Nz):
            #             zC = zmin + K * DZ
            #             val = scheme.integrate(
            #                 _FF,#lambda x: np.exp(x[0]),
            #                 quadpy.c3.cube_points([xC - Rx, xC + Rx], [yC - Ry, yC + Ry], [zC - Rz, zC + Rz]),
            #                 )
            #             #val = scheme.integrate(
            #             #    _FF,
            #                 #[[xC-Rx, yC-Ry, zC-Rz], [xC+Rx, yC-Ry, zC-Rz], [xC-Rx, yC+Ry, zC-Rz], [xC-Rx, yC-Ry, zC+Rz]],
            #             #    [[xC - Rx, xC + Rx], [yC - Ry, yC + Ry], [zC - Rz, zC + Rz]]
            #             #    )
            #             res += val
            # return res
    
    def InitDensity(self, StaticDens = None, vac =[.5, .5, .5], 
                    dtype = np.float32, didx = None,):
        if didx is None:
            self.didx = np.arange(self.get_no())
        else:
            self.didx = didx
        step = self.dx/4
        _vac = np.array(vac)
        Nxyz = np.array([2,2,2])
        def cond():
            return np.mod(Nxyz,2)
        it = 0
        while (cond()==0).any():
            _vac[cond() == 0]+= step
            tpos = np.array([self.get_pos(i) for i in self.didx])
            tpos_min = tpos.min(axis=0)
            #print(tpos_min.shape)
            
            #_Lx,_Ly,_Lz = np.max(tpos,axis=0) - np.min(tpos,axis=0) + vac
            tpos[:,0]-=tpos[:,0].min()
            tpos[:,1]-=tpos[:,1].min()
            tpos[:,2]-=tpos[:,2].min()
            tpos += _vac
            Lx = tpos[:,0].max() + _vac[0] 
            Ly = tpos[:,1].max() + _vac[1] 
            Lz = tpos[:,2].max() + _vac[2] 
            Nx, Ny, Nz = int(Lx/self.dx), int(Ly/self.dx),int(Lz/self.dx)
            Nxyz       = np.array([Nx,Ny,Nz])
            #print(it)
            it+=1
        #print(_vac)
            
        self.Dens    = np.zeros((Nx, Ny, Nz), dtype  = dtype)
        self.fpos    = tpos / self.dx
        self.StaticDens = StaticDens
        self.OrbList = [self.orbital_on_grid(i).astype(dtype) for i in self.didx]
        self.Flist   = None
    
    def evaluate_density(self, DM, UT = None, tol = 1e-7, use_numba = False, Sij = None):
        if UT is not None:
            _DM = UT@DM@UT.conj().T
        else:
            _DM = DM
        if self.StaticDens is not None: self.Dens[:,:,:] = self.StaticDens
        else:                           self.Dens[:,:,:] = .0
        
        orb_kind = [i for i in range(_DM.shape[-1])]
        rpos     = self.fpos * self.dx
        AS       = False
        
        if self.Flist is None:
            self.Flist = make_density(self.OrbList, _DM, orb_kind, rpos, self.dx,
                                      self.Dens, self.StaticDens, tol = tol, add_static = AS, return_Flist = True,
                                      Sij = Sij)
        
        make_density(self.OrbList, _DM, orb_kind, rpos, self.dx,
                     self.Dens, self.StaticDens, tol = tol, add_static = AS, 
                     Flist = self.Flist, Sij = Sij )
        
        return self.Dens
    
    def MatrixElements(self,Field, out, Sij = None, tol = 1e-5):
        rpos     = self.fpos * self.dx
        orb_kind = [i for i in range(out.shape[-1])]
        return matrixelementsoffield(self.OrbList, orb_kind, rpos, self.dx,
                                     Field, out, Flist = self.Flist, Sij = Sij, 
                                     tol = tol)
    
    
        
    
    
        
    
