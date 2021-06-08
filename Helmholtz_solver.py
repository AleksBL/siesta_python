#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:43:06 2021

@author: aleks
"""
import numpy as np
from numba import jit
from scipy.interpolate import griddata
from Conj_grad import fftZP,Pil_ud
from joblib import Parallel, delayed
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as Sparse_Solve
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from mayavi import mlab



norm = np.linalg.norm
exp  = np.exp
pi   = np.pi
FT   = np.fft.fftn
IFT  = np.fft.ifftn
IFS  = np.fft.ifftshift
FS   = np.fft.fftshift
def Repeat_vec(v,n,m):
    return np.repeat(np.repeat(v[:,np.newaxis],n,axis=1)[:,:,np.newaxis],m,axis=2)

def gd(dr,k):
    if dr.shape==(3,):
        n=norm(dr)
    else:
        n=np.sqrt(np.sum(dr**2,axis=-1))
    return exp(1j*k*n)/(4*pi*n)

def Interpolate_Parallel(r,f,r_out,n_cores = 4):
    mp='loky'
    B = np.array_split(r_out,n_cores)
    Res=Parallel(n_jobs=n_cores,backend = mp, mmap_mode = 'w+')(delayed(griddata)(r, f, i,method='linear') for i in B )
    Arr =  np.hstack(Res)
    Arr[np.where(np.isnan(Arr))] = 0
    return Arr
    

def get_rectangular_grid(r,dx):
    # input r skal være (n,3) shape
    x0=r[:,0].min();x1=r[:,0].max()
    y0=r[:,1].min();y1=r[:,1].max()
    z0=r[:,2].min();z1=r[:,2].max()
    x=np.arange(x0,x1+dx,dx); y=np.arange(y0,y1+dx,dx); z=np.arange(z0,z1+dx,dx)
    nx=len(x)
    ny=len(y)
    nz=len(z)
    rx,ry,rz = np.meshgrid(x,y,z,indexing='ij')
    ri = np.vstack([rx.ravel(),ry.ravel(),rz.ravel()]).T
    ri = ri.reshape(nx,ny,nz,3)
    return ri


@jit
def Insert(A,a,r,k,Vol):
    nx=a.shape[0]
    ny=a.shape[1]
    nz=a.shape[2]
    for i0 in range(2*nx-1):
        for i1 in range(2*ny-1):
            for i2 in range(2*nz-1):
                A[i0,i1,i2]=a[abs(i0-(nx-1)),abs(i1-(ny-1)),abs(i2-(nz-1))]*Vol
    A[nx-1,ny-1,nz-1] = ( -1 + (1-1j*r*k)*exp(1j*r*k))/k**2


class Helmholtz_Solver:
    def __init__(self, f, r):
        self.f = f
        self.r = r
        if isinstance(self.f,type(None)):
            pass
        else:
            assert r.shape[1]==3
            assert len(f) == r.shape[0]
        
        
    def interpolate_f_samples(self,dx,ncores=1):
        x0=self.r[:,0].min();x1=self.r[:,0].max()
        y0=self.r[:,1].min();y1=self.r[:,1].max()
        z0=self.r[:,2].min();z1=self.r[:,2].max()
        Vol = (x1-x0)*(y1-y0)*(z1-z0)
        x=np.arange(x0,x1+dx,dx); y=np.arange(y0,y1+dx,dx); z=np.arange(z0,z1+dx,dx)
        rx,ry,rz = np.meshgrid(x,y,z,indexing='ij')
        ri = np.vstack([rx.ravel(),ry.ravel(),rz.ravel()]).T
        
        del rx,ry,rz
        print('Interpolating...\n')
        #Interpoleringen virker for vilkårlig sampling, men er vanvittig langsom, så regner hellere punkter på et 
        # Rektangulært grid i stedet og brug manual input hvis det er muligt!
        
        if ncores==1:
            self.fi     = griddata(self.r,self.f,ri,method='linear').reshape(len(x),len(y),len(z))
        else:
            self.fi     = Interpolate_Parallel(self.r,self.f,ri).reshape(len(x),len(y),len(z))
        
        self.fi[np.where(np.isnan(self.fi))] = 0
        self.ri     = ri.reshape(len(x),len(y),len(z),3)
        self.dx     = dx
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)
        self.dV = Vol/((len(x)+1)*(len(x)+1)*(len(x)+1))
        self.equiv_r = ((3/(4*pi))*self.dV)**(1/3)
    
    def Manual_input(self,fi,ri):
        #Til hvis man vil interpolere parallelt, det går galt af en eller anden grund 
        #når man putter "Parallel" ind i en class
        self.fi  = fi
        self.ri  = ri
        dx = ri[1,0,0,0]-ri[0,0,0,0]
        self.dx     = dx
        self.nx = len(ri[:,0,0,0])
        self.ny = len(ri[0,:,0,0])
        self.nz = len(ri[0,0,:,0])
        assert np.isclose(ri[1,0,0,0]-ri[0,0,0,0],ri[0,1,0,1]-ri[0,0,0,1])
        assert np.isclose(ri[0,1,0,1]-ri[0,0,0,1],ri[0,0,1,2]-ri[0,0,0,2])
        
        self.dV = dx**3
        self.equiv_r = ((3/(4*pi))*self.dV)**(1/3)
    
    def Matrix_System(self,k,pbc = [0,0,0]):
        print('Setting up sparse system...\n')
        Ix = sp.identity(self.nx)
        Iy = sp.identity(self.ny)
        Iz = sp.identity(self.nz)
        ox = np.ones(self.nx)
        oy = np.ones(self.ny)
        oz = np.ones(self.nz)
        
        Lx = sp.csr_matrix(sp.spdiags([ox,-2*ox,ox],[-1,0,1],self.nx,self.nx))/self.dx**2
        if pbc[0]==1:
            Lx[self.nx-1,0] = 1/self.dx**2; 
            Lx[0,self.nx-1] = 1/self.dx**2
        Ly = sp.csr_matrix(sp.spdiags([oy,-2*oy,oy],[-1,0,1],self.ny,self.ny))/self.dx**2
        if pbc[1]==1:
            Ly[self.ny-1,0] = 1/self.dx**2; 
            Ly[0,self.ny-1] = 1/self.dx**2
        Lz = sp.csr_matrix(sp.spdiags([oz,-2*oz,oz],[-1,0,1],self.nz,self.nz))/self.dx**2
        if pbc[2]==1:
            Lz[self.nz-1,0] = 1/self.dx**2; 
            Lz[0,self.nz-1] = 1/self.dx**2
        
        #Nabla² er nem at bygge med et kronecker produkt....
        Laplacian = sp.kron(sp.kron(Lx,Iy),Iz) + sp.kron(sp.kron(Ix,Ly),Iz) + sp.kron(sp.kron(Ix,Iy),Lz)
        #(w/c)**2 led i helmholtz-lign
        woc =sp.csr_matrix(sp.spdiags([ k**2*np.ones(self.nx*self.ny*self.nz)],[0],
                                                     self.nx*self.ny*self.nz,
                                                     self.nx*self.ny*self.nz)     )
        
        self.LHS = Laplacian + woc
        self.b   = self.fi.reshape(self.nx*self.ny*self.nz)
    
    def Add_PML(self,fracx=1/10,fracy=1/10,fracz=1/10, P = 3, R = 1e-3, tol = 1e-5,where= [0,0,0]):
        from scipy.constants import epsilon_0, mu_0
        n0 = np.sqrt(mu_0/epsilon_0)
        bc = self.b.copy().reshape(self.nx,self.ny,self.nz).astype(np.complex128)
        
        ix = int(np.floor(fracx*self.nx))
        iy = int(np.floor(fracy*self.ny))
        iz = int(np.floor(fracz*self.nz))
        
        sl0_x = np.arange(0,ix)
        sl0_y = np.arange(0,iy)
        sl0_z = np.arange(0,iz)
        sl1_x = np.arange(self.nx-ix,self.nx)
        sl1_y = np.arange(self.ny-iy,self.ny)
        sl1_z = np.arange(self.nz-iz,self.nz)
        #En masse checks om det inhomogene led er nul i PML.....
        def pml_error():
            print('Scattering term not zero inside PML!?!??!?!')
            assert 1==2
            
        if not (np.abs(bc[sl0_x,:,:]) < tol).all():
            pml_error()
        if not (np.abs(bc[sl1_x,:,:]) < tol).all():
            pml_error()
        if not (np.abs(bc[:,sl0_y,:]) < tol).all():
            pml_error()
        if not (np.abs(bc[:,sl1_y,:]) < tol).all():
            pml_error()
        if not (np.abs(bc[:,:,sl0_z]) < tol).all():
            pml_error()
        if not (np.abs(bc[:,:,sl1_z]) < tol).all():
            pml_error()
        t0_x = self.ri[sl0_x,:,:,0].max()-self.ri[sl0_x,:,:,0].min()
        t0_y = self.ri[:,sl0_y,:,1].max()-self.ri[:,sl0_y,:,1].min()
        t0_z = self.ri[:,:,sl0_z,2].max()-self.ri[:,:,sl0_z,2].min()
        t1_x = self.ri[sl1_x,:,:,0].max()-self.ri[sl1_x,:,:,0].min()
        t1_y = self.ri[:,sl1_y,:,1].max()-self.ri[:,sl1_y,:,1].min()
        t1_z = self.ri[:,:,sl1_z,2].max()-self.ri[:,:,sl1_z,2].min()
        
        sig0_x = -(P+1)*np.log(R)/(2*n0*t0_x**(P+1))
        sig1_x = -(P+1)*np.log(R)/(2*n0*t1_x**(P+1))
        sig0_y = -(P+1)*np.log(R)/(2*n0*t0_y**(P+1))
        sig1_y = -(P+1)*np.log(R)/(2*n0*t1_y**(P+1))
        sig0_z = -(P+1)*np.log(R)/(2*n0*t0_z**(P+1))
        sig1_z = -(P+1)*np.log(R)/(2*n0*t1_z**(P+1))
        
        PML0_x = sig0_x * np.abs(sl0_x-           ix +1  )**P 
        PML1_x = sig1_x *       (sl1_x-(self.nx - ix )   )**P
        PML0_y = sig0_y * np.abs(sl0_y-           iy +1  )**P
        PML1_y = sig1_y *       (sl1_y-(self.ny - iy )   )**P
        PML0_z = sig0_z * np.abs(sl0_z-           iy +1  )**P
        PML1_z = sig1_z *       (sl1_z-(self.nz - iz )   )**P
        if where[0]==1:
            bc[sl0_x,:,:] += - 1j * Repeat_vec(PML0_x,self.ny,self.nz)
            bc[sl1_x,:,:] += - 1j * Repeat_vec(PML1_x,self.ny,self.nz)
        if where[1]==1:
            bc[:,sl0_y,:] += - 1j * Repeat_vec(PML0_y,self.nx,self.nz).transpose(1,0,2)
            bc[:,sl1_y,:] += - 1j * Repeat_vec(PML1_y,self.nx,self.nz).transpose(1,0,2)
        if where[2]==1:
            bc[:,:,sl0_z] += - 1j * Repeat_vec(PML0_z,self.nx,self.ny).transpose(1,2,0)
            bc[:,:,sl1_z] += - 1j * Repeat_vec(PML1_z,self.nx,self.ny).transpose(1,2,0)
        
        self.b_nopml = self.b.copy()
        self.b       = bc.reshape(self.nx*self.ny*self.nz)
    
    def Solve(self,solver='cg'):
        if solver=='cg':
            sol,ex = cg (self.LHS , self.b)
            if ex==0:
                return sol.reshape(self.nx,self.ny,self.nz)
            else:
                print('CG algorithm not converged')
        if solver=='Normal':
            return Sparse_Solve(self.LHS,self.b).reshape(self.nx,self.ny,self.nz)
    
    def get_g(self,k,d_t = np.complex128,BC='radiating'):
        offset = np.array([self.ri[:,:,:,0].min(),self.ri[:,:,:,1].min(),self.ri[:,:,:,2].min()])
        print(offset)
        if BC=='radiating':
            G_VEC=gd(self.ri - offset , k)
        g = np.zeros((2*self.nx-1,2*self.ny-1,2*self.nz-1),dtype=d_t)
        Insert(g,G_VEC,self.equiv_r,k,self.dV)
        self.fg = FT(g)
    
    def Fold(self):
        # Minus foran
        # Foldning af g(r-r') med spredningsleddet på højre side af Poisson
        return  - Pil_ud(IFS(IFT(self.fg*FT(fftZP(self.fi)))),self.nx,self.ny,self.nz)




# N=20
# l1 = np.linspace(-10,10,N)
# l2 = np.linspace(-10,10,N)
# l3 = np.linspace(-10,10,N)

# r_sphere = 2
# x,y,z=np.meshgrid(l1,l2,l3,indexing='ij')
# R=np.vstack([x.ravel(),y.ravel(),z.ravel()]).T
# d = np.ones(len(R))
# d[np.where(np.sum(R**2,axis=1)>r_sphere**2)]=0

# wn = 0.01
# A  = Helmholtz_Solver( d  , R )
# B  = Helmholtz_Solver( d  , R )

# A.interpolate_f_samples( dx=0.3, ncores = 4)

# B.interpolate_f_samples( dx=0.3 )

# ri = get_rectangular_grid(R,0.3)
# fi = Interpolate_Parallel(R,d,ri,n_cores = 4)

# A.Manual_input(fi,ri)
# B.Manual_input(fi,ri)


# A.Matrix_System(wn,pbc = [0,0,0])
# A.Add_PML(where=[1,1,1])
# V1 = A.Solve()

# B.get_g(wn)
# V2  = B.Fold()







