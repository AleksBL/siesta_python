#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:08:36 2021

@author: aleks
"""    

## J.P. Dahl Introduction to the quantum world of atoms and molecules
## Kapitel 8, appendix C og D

from scipy.special import lpmv as Lg
import numpy as np
import math
import os
import matplotlib.pyplot as plt
ld = os.listdir

fact    = math.factorial
cos     = np.cos
sin     = np.sin
exp     = np.exp
sqrt    = np.sqrt
pi      = np.pi
arctan2 = np.arctan2
matmul  = np.matmul

def d_Lg_dx(m, v, x):
    #  https://www.wolframalpha.com/input/?i=d%2Fdx+LegendreP%5Bm%2Cn%2Cx%5D
    return ( (v-m+1) * Lg(m,v+1,x) - ( v + 1 ) * x * Lg(m,v,x) ) / (x**2 - 1)

def Y_ml(theta,phi,l = 0, m = 0,diff = 'none'):
    #  Condon-Shortley fase-konvention (JP Dahl)
    N = (-1) ** ( (m + abs(m))/2 ) *  sqrt(  (( 2*l+1 )/( 4*pi ))*( (fact( l - abs(m) )/fact( l + abs(m) )) )  )
    if diff == 'none':
        return N *      Lg( abs(m), l, cos(theta) )                    * exp( 1j * m * phi )
    elif diff == 'theta':
        return N * d_Lg_dx( abs(m), l, cos(theta) ) * ( - sin(theta) ) * exp( 1j * m * phi )
    elif diff == 'phi':
        return N *      Lg( abs(m), l, cos(theta) ) *     1j * m       * exp( 1j * m * phi )  

def cart_to_sph(r):
    # Wiki spherical coordinates 
    Res = np.zeros(r.shape)
    if len(r.shape) == 4: 
        R     = np.sqrt(np.sum(r**2,axis=3))
        theta = arctan2( sqrt(r[:,:,:,0]**2 + r[:,:,:,1]**2 ) , r[:,:,:,2] )
        phi   = arctan2( r[:,:,:,1]                           , r[:,:,:,0] )
        
        Res[:,:,:,0] = R
        Res[:,:,:,1] = theta
        Res[:,:,:,2] = phi
        return Res
    
    if len(r.shape) == 2:
        R     = np.sqrt(np.sum(r**2,axis=1))
        theta = arctan2( sqrt(r[:,0]**2 + r[:,1]**2 ) , r[:,2] )
        phi   = arctan2( r[:,1]                       , r[:,0] )
        Res[:,0] = R
        Res[:,1] = theta
        Res[:,2] = phi
        return Res

def sph_to_cart(c):
    # Wiki spherical coordinates
    Res = np.zeros(c.shape)
    if len(c.shape) == 4: 
        x = c[:,:,:,0] * sin( c[:,:,:,1] ) * cos( c[:,:,:,2] )
        y = c[:,:,:,0] * sin( c[:,:,:,1] ) * sin( c[:,:,:,2] )
        z = c[:,:,:,0] * cos( c[:,:,:,1] )
        Res[:,:,:,0] = x
        Res[:,:,:,1] = y
        Res[:,:,:,2] = z
        return Res
    
    if len(c.shape) == 2:
        x = c[:,0] * sin( c[:,1] ) * cos( c[:,2] )
        y = c[:,0] * sin( c[:,1] ) * sin( c[:,2] )
        z = c[:,0] * cos( c[:,1] )
        Res[:,0] = x
        Res[:,1] = y
        Res[:,2] = z
        return Res

def read_gen_basis_output(file):
    x   = []
    values = []
    with open(file,'r') as f:
        it=0
        for lines in f:
            if it == 0:
                At    =  lines.split()[1]
                l     =  lines.split()[2]
                n     =  lines.split()[3]
                z     =  lines.split()[4]
                pol   =  lines.split()[5]
                pop   =  lines.split()[6]
            elif it==1:
                pass
            else:
                vals=lines.split()
                x      += [float(vals[0])]
                values += [float(vals[1])]
            it+=1
    f.close()
    x = np.array(x)
    f = np.array(values)
    dx = (np.roll(x,-1)-x)[0:len(x)-1]
    df = (np.roll(f,-1)-f)[0:len(x)-1]
    dfdx = df/dx
    x_2 = (np.roll(x,-1)+x)[0:len(x)-1]/2
    Dict = {'Atom':At,
            'l': l,
            'n': n,
            'z': z,
            'is_polarised':pol,
            'popul': pop,
            'Radial_grid':x,
            'Radial_func': f,
            'Radial_derivative_func': dfdx,
            'Radial_derivative_grid'  : x_2}
    return Dict
          
def vector_transform(vector_field,coords,way):
    # Den kan enten koordiaterne og komponenterne af vektorfeltet i  (n,3)-format
    # Eller(nx,ny,nz,3)-format
    
    if way == 'sph_to_cart':
        if len(coords.shape)==2:
            r     = coords[:,0]
            theta = coords[:,1]
            phi   = coords[:,2]
        elif len(coords.shape)==4:
            r     = coords[:,:,:,0]
            theta = coords[:,:,:,1]
            phi   = coords[:,:,:,2]
        Zrs = np.zeros(r.shape)
        
        Transform = np.array([[sin(theta)*cos(phi) ,    cos(theta)*cos(phi) ,    -sin(phi)],
                              [sin(theta)*sin(phi) ,    cos(theta)*sin(phi) ,     cos(phi)],
                              [cos(theta)          ,   -sin(theta)          ,     Zrs     ] ])
        
        if len(coords.shape)==2:
            Transform = Transform.transpose(2,0,1)
        elif len(coords.shape)==4:
            Transform = Transform.transpose(2,3,4,0,1)
        coords_new = sph_to_cart(coords)
    elif way == 'cart_to_sph':
        if len(coords.shape)==2:
            x     = coords[:,0]
            y     = coords[:,1]
            z     = coords[:,2]
        elif len(coords.shape)==4:
            x     = coords[:,:,:,0]
            y     = coords[:,:,:,1]
            z     = coords[:,:,:,2]
        N     = sqrt(x**2+y**2+z**2)
        rho   = sqrt(x**2+y**2)
        Zrs   = np.zeros(x.shape)
        
        Transform = np.array([[ x/N         ,  y/N          ,   z/N               ],
                              [ x*z/(rho*N) ,  y*z/(rho*N)  ,  -rho/N             ],
                              [-y/rho       ,  x/rho        ,   Zrs               ] ])
        
        if len(coords.shape)==2:
            Transform = Transform.transpose(2,0,1)
        elif len(coords.shape)==4:
            Transform = Transform.transpose(2,3,4,0,1)
        coords_new = cart_to_sph(coords)
    
    if len(coords.shape)==2:
        Vf = np.repeat(vector_field[:,:,np.newaxis],1,axis=2)
        return matmul(Transform,Vf)[:,:,0], coords_new

    if len(coords.shape)==4:
        Vf = np.repeat(vector_field[:,:,:,:,np.newaxis],1,axis=4)
        return matmul(Transform,Vf)[:,:,:,:,0], coords_new

class siesta_basis_set:
    def __init__(self,basis_dir):
        self.basis_dir = basis_dir
        self.read_basis()
    
    def read_basis(self):
        it=0
        load = []
        for file_name in ld(self.basis_dir):
            if 'ORB' in file_name:
                name = self.basis_dir+'/'+file_name
                load+=[name]
                it+=1
        Orbs = np.array(([1 for i in range(it)]),dtype=object)
        
        it=0
        for i in load:
            Orbs[it] = read_gen_basis_output(i)
            it+=1
        
        self.basis = Orbs
        self.n_radial = it
    def orbital(self,r,N,r_cartesian = True):
        if r_cartesian == True:
            rs = cart_to_sph(r)
        else:
            rs = r.copy()
        
        All = []
        n      = int(self.basis[N]['n'])
        l      = int(self.basis[N]['l'])
        print('angular momentum : ' + str(l) +'\n')
        print('principal quantum number : ' + str(n) +'\n')
        
        rcords = self.basis[N]['Radial_grid']
        Rf     = self.basis[N]['Radial_func']
        ml = np.arange(-l,l+1)
        if len(rs.shape) == 2:
            R_part = np.interp(rs[:,0],rcords,Rf)
        elif len(rs.shape) == 4:
            R_part = np.interp(rs[:,:,:,0],rcords,Rf)
        
        for val in ml: 
            if len(rs.shape) == 2:
                Y = Y_ml(rs[:,1]    ,rs[:,2]    ,l,val)
            if len(rs.shape) == 4:
                Y = Y_ml(rs[:,:,:,1],rs[:,:,:,2],l,val)
            All += [R_part * Y]
        return All
    
    def grad_orbital(self,r,N,r_cartesian = True, return_cartesian = True):
        if r_cartesian == True:
            rs = cart_to_sph(r)
        else:
            rs = r.copy()
        n      = int(self.basis[N]['n'])
        l      = int(self.basis[N]['l'])
        ml     =     np.arange(-l,l+1)
        
        print('angular momentum : ' + str(l) +'\n')
        print('principal quantum number : ' + str(n) +'\n')
        rcords =     self.basis[N]['Radial_grid']
        Rf     =     self.basis[N]['Radial_func']
        if len(rs.shape) == 2:
            R_part = np.interp(rs[:,0],rcords,Rf)
            theta     = rs[:,1]
            rad_coord = rs[:,0]
        elif len(rs.shape) == 4:
            R_part = np.interp(rs[:,:,:,0],rcords,Rf)
            theta     = rs[:,:,:,1]
            rad_coord = rs[:,:,:,0]
            
        der_rcords =     self.basis[N]['Radial_derivative_grid']
        der_Rf     =     self.basis[N]['Radial_derivative_func']
        
        if len(rs.shape) == 2:
            dRf_dr = np.interp(rs[:,0],    der_rcords,der_Rf)
        elif len(rs.shape) == 4:
            dRf_dr = np.interp(rs[:,:,:,0],der_rcords,der_Rf)
        Gradients = []
        for val in ml: 
            if len(rs.shape) == 2:
                Y    = Y_ml(rs[:,1]    ,    rs[:,2]    ,l,val)
                dYdtheta = Y_ml(rs[:,1]    ,rs[:,2]    ,l,val,diff='theta')
                dYdphi   = Y_ml(rs[:,1]    ,rs[:,2]    ,l,val,diff='phi'  )
            
            if len(rs.shape) == 4:
                Y        = Y_ml(rs[:,:,:,1],rs[:,:,:,2] ,l,val)
                dYdtheta = Y_ml(rs[:,:,:,1],rs[:,:,:,2] ,l,val,diff='theta')
                dYdphi   = Y_ml(rs[:,:,:,1],rs[:,:,:,2] ,l,val,diff='phi'  )
            grad_r     = dRf_dr *    Y 
            grad_theta = R_part *   dYdtheta /rad_coord 
            grad_phi   = R_part *   dYdphi   /(rad_coord*sin(theta))
            
            if len(rs.shape) == 2:
                Gradient_spherical = np.array([grad_r,grad_theta,grad_phi]).transpose(1,0)
            elif len(rs.shape) == 4:
                Gradient_spherical = np.array([grad_r,grad_theta,grad_phi]).transpose(1,2,3,0)
            
            if return_cartesian==False:
                Gradients += [Gradient_spherical]
            elif return_cartesian==True:
                Gradient_cartesian,pos_cart = vector_transform(Gradient_spherical,rs,way = 'sph_to_cart')
                Gradients += [Gradient_cartesian]
        
        return Gradients


# import sympy as sp
# from sympy.abc import r, phi, theta 
#
# Mu = sp.Matrix([[sp.sin(theta)*sp.cos(phi), sp.cos(theta)*sp.cos(phi),-sp.sin(theta)*sp.sin(phi)],
#                 [sp.sin(theta)*sp.sin(phi), sp.cos(theta)*sp.sin(phi), sp.sin(theta)*sp.cos(phi)],
#                 [sp.cos(theta),          -sp.sin(theta),                          0   ] ])
# Mu_inv = sp.simplify(Mu.inv())

# M = sp.Matrix([[sp.sin(theta)*sp.cos(phi), r*sp.cos(theta)*sp.cos(phi),-r*sp.sin(theta)*sp.sin(phi)],
#                 [sp.sin(theta)*sp.sin(phi), r*sp.cos(theta)*sp.sin(phi), r*sp.sin(theta)*sp.cos(phi)],
#                 [sp.cos(theta)            ,-r*sp.sin(theta)          ,               0]])
# M_inv = sp.simplify(M.inv())

# D = 5
# N = 81

# x,y,z = np.meshgrid(np.linspace(-D,D,N),np.linspace(-D,D,N),np.linspace(-D,D,N),indexing = 'ij')
# u = -y/np.sqrt(x**2 + y**2)
# v =  x/np.sqrt(x**2 + y**2)

# r = np.array([x,y,z]).transpose(1,2,3,0)
# vfc =  np.array([u,v,np.zeros(u.shape)]).transpose(1,2,3,0)

# vfs,  rs  = vector_transform(vfc,r, way = 'cart_to_sph')

# vfc2, rc  = vector_transform(vfs,rs,way = 'sph_to_cart')


