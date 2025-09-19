#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 08:46:07 2022

@author: aleksander
"""

from sisl import SelfEnergy
import numpy as np
from numpy import conjugate, matmul
from numpy import subtract
from numpy import empty, zeros, eye, delete
from numpy import zeros_like, empty_like
from numpy import complex128
from numpy import abs as _abs

from sisl._internal import set_module
from sisl.messages import warn, info
from sisl.utils.mathematics import fnorm
from sisl.utils.ranges import array_arange
from sisl._help import array_replace
import sisl._array as _a
from sisl.linalg import linalg_info, solve, inv
from sisl.linalg.base import _compute_lwork
from sisl.sparse_geometry import _SparseGeometry
from sisl.physics.brillouinzone import MonkhorstPack
from sisl.physics.bloch import Bloch


def CZ(s):
    return np.zeros(s,dtype=complex)
Inv = np.linalg.inv
MM = np.matmul


def SE_vectorised(E_in,H,V,S00=None,S01=None,eta=1e-3,eps=1e-15,DT=np.complex128):
    # numpy broadcaster til de sidste to indexer i arrayet, dvs så længe vi holder de egentlige
    # to indekser vi vi gerne vil invertere, matrix-multiplicere osv i de sidste to kan man vectorisere
    # indexet over energi. H og V afhænger af k-indekset, ud den går ikke her
    n_e = len(E_in)
    n=len(H)
    alpha = CZ((n_e,n,n))
    #alpha[:,:,:] += V
    beta = CZ((n_e,n,n))
    #beta [:,:,:] +=  V.T.conj()
    if S00 is None and S01 is None:
        I=np.diag(np.ones(n))
        igb = CZ((n_e,n,n))
        for i in range(n_e):
            igb[i,:,:] =I*(E_in[i]+1j*eta) - H
            alpha[i,:,:] = V
            beta [i,:,:] = V.conj().T
    else:
        igb = CZ((n_e,n,n))
        for i in range(n_e):
            z = (E_in[i]+1j*eta)
            igb[i,:,:] =S00*z - H
            alpha[i,:,:] = V - S01 * z
            beta [i,:,:] = V.conj().T - S01.conj().T * z
    sse=np.zeros((n_e,n,n),dtype=DT)
    while True:
        gb       = Inv(igb)
        gb_beta  = MM(gb,beta)
        gb_alpha = MM(gb,alpha)
        sse     += MM(alpha,gb_beta)
        igb     -= MM(alpha,gb_beta) + MM(beta,gb_alpha)
        alpha    = MM(alpha,gb_alpha)
        beta     = MM(beta,gb_beta)
        if ((np.sum(np.abs(alpha),axis=(1,2))+np.sum(np.abs(beta),axis=(1,2)))<eps).all():
            return sse


@set_module("sisl.physics")
class RealSpaceSE2E(SelfEnergy):
    r""" Surface real-space self-energy (or Green function) for a given physical object with limited periodicity

    The surface real-space self-energy is calculated via the k-averaged Green function:

    .. math::
        \boldsymbol\Sigma^\mathcal{R}(E) = \mathbf S^\mathcal{R} (E+i\eta) - \mathbf H^\mathcal{R}
             - \Big[\sum_{\mathbf k} \mathbf G_{\mathbf k}(E)\Big]^{-1}

    The method actually used is relying on `RecursiveSI` and `~sisl.physics.Bloch` objects.

    Parameters
    ----------
    semi : SemiInfinite
        physical object which contains the semi-infinite direction, it is from
        this object we calculate the self-energy to be put into the surface.
        a physical object from which to calculate the real-space self-energy.
        `semi` and `surface` must have parallel lattice vectors.
    surface : SparseOrbitalBZ
        parent object containing the surface of system. `semi` is attached into this
        object via the overlapping regions, the atoms that overlap `semi` and `surface`
        are determined in the `initialize` routine.
        `semi` and `surface` must have parallel lattice vectors.
    k_axes : array_like of int
        axes where k-points are desired. 1 or 2 values are required. The axis cannot be a direction
        along the `semi` semi-infinite direction.
    unfold : (3,) of int
        number of times the `surface` structure is tiled along each direction
        Since this is a surface there will maximally be 2 unfolds being non-unity.
    eta : float, optional
        imaginary part in the self-energy calculations (default 1e-4 eV)
    dk : float, optional
        fineness of the default integration grid, specified in units of Ang, default to 1000 which
        translates to 1000 k-points along reciprocal cells of length 1. Ang^-1.
    bz : BrillouinZone, optional
        integration k-points, if not passed the number of k-points will be determined using
        `dk` and time-reversal symmetry will be determined by `trs`, the number of points refers
        to the unfolded system.
    trs : bool, optional
        whether time-reversal symmetry is used in the BrillouinZone integration, default
        to true.

    Examples
    --------
    >>> graphene = geom.graphene()
    >>> H = Hamiltonian(graphene)
    >>> H.construct([(0.1, 1.44), (0, -2.7)])
    >>> se = RecursiveSI(H, "-A")
    >>> Hsurf = H.tile(3, 0)
    >>> Hsurf.set_nsc(a=1)
    >>> rsi = RealSpaceSI(se, Hsurf, 1, (1, 4, 1))
    >>> rsi.green(0.1)

    The Brillouin zone integration is determined naturally.

    >>> graphene = geom.graphene()
    >>> H = Hamiltonian(graphene)
    >>> H.construct([(0.1, 1.44), (0, -2.7)])
    >>> se = RecursiveSI(H, "-A")
    >>> Hsurf = H.tile(3, 0)
    >>> Hsurf.set_nsc(a=1)
    >>> rsi = RealSpaceSI(se, Hsurf, 1, (1, 4, 1))
    >>> rsi.set_options(eta=1e-3, bz=MonkhorstPack(H, [1, 1000, 1]))
    >>> rsi.initialize()
    >>> rsi.green(0.1) # eta = 1e-3
    >>> rsi.green(0.1 + 1j * 1e-4) # eta = 1e-4

    Manually specify Brillouin zone integration and default :math:`\eta` value.
    """

    def __init__(self, semis, surface, k_axes, unfold=(1, 1, 1), **options):
        """ Initialize real-space self-energy calculator """
        self.semi = semis
        self.surface = surface

        if not self.semi.sc.parallel(surface.sc):
            raise ValueError(f"{self.__class__.__name__} requires semi and surface to have parallel "
                             "lattice vectors.")

        self._k_axes = np.sort(_a.arrayi(k_axes).ravel())
        k_ax = self._k_axes

        if self.semi.semi_inf in k_ax:
            raise ValueError(f"{self.__class__.__name__} found the self-energy direction to be "
                             "the same as one of the k-axes, this is not allowed.")

        # Local variables for the completion of the details
        self._unfold = _a.arrayi([max(1, un) for un in unfold])

        if self.surface.nsc[semi.semi_inf] > 1:
            raise ValueError(f"{self.__class__.__name__} surface has periodicity along the semi-infinite "
                             "direction. This is not allowed.")
        if np.any(self.surface.nsc[k_ax] < 3):
            raise ValueError(f"{self.__class__.__name__} found k-axes without periodicity. "
                             "Correct `k_axes` via `.set_option`.")

        if self._unfold[semi.semi_inf] > 1:
            raise ValueError(f"{self.__class__.__name__} cannot unfold along the semi-infinite direction. "
                             "This is a surface real-space self-energy.")

        # Now we need to figure out the atoms in the surface that corresponds to the
        # semi-infinite direction.
        # Now figure out which atoms in `semi` intersects those in `surface`
        semi_inf = self.semi.semi_inf
        semi_na = self.semi.geometry.na
        semi_min = self.semi.geometry.xyz.min(0)

        surf_na = self.surface.geometry.na

        # Check the coordinates...
        if self.semi.semi_inf_dir == 1:
            # "right", last atoms
            atoms = np.arange(surf_na - semi_na, surf_na)
        else:
            # "left", first atoms
            atoms = np.arange(semi_na)

        # Semi-infinite atoms in surface
        surf_min = self.surface.geometry.xyz[atoms, :].min(0)

        g_surf = self.surface.geometry.xyz[atoms, :] - (surf_min - semi_min)

        # Check atomic coordinates are the same
        # Precision is 0.001 Ang
        if not np.allclose(self.semi.geometry.xyz, g_surf, rtol=0, atol=1e-3):
            print("Coordinate difference:")
            print(self.semi.geometry.xyz - g_surf)
            raise ValueError(f"{self.__class__.__name__} overlapping semi-infinite "
                             "and surface atoms does not coincide!")

        # Surface orbitals to put in the semi-infinite self-energy into.
        self._surface_orbs = self.surface.geometry.a2o(atoms, True).reshape(-1, 1)

        self._options = {
            # For true, the semi-infinite direction will use the bulk values for the
            # elements that overlap with the semi-infinito
            "semi_bulk": True,
            # fineness of the integration k-grid [Ang]
            "dk": 1000,
            # whether TRS is used (G + G.T) * 0.5
            "trs": True,
            # imaginary part used in the Green function calculation (unless an imaginary energy is passed)
            "eta": 1e-4,
            # The BrillouinZone used for integration
            "bz": None,
        }
        self.set_options(**options)
        self.initialize()

    def __str__(self):
        """ String representation of RealSpaceSI """
        d = {"class": self.__class__.__name__}
        for i in range(3):
            d[f"u{i}"] = self._unfold[i]
        d["k"] = str(list(self._k_axes))
        d["semi"] = str(self.semi).replace("\n", "\n  ")
        d["surface"] = str(self.surface).replace("\n", "\n  ")
        d["bz"] = str(self._options["bz"]).replace("\n", "\n ")
        d["trs"] = str(self._options["trs"])
        return  ("{class}{{unfold: [{u0}, {u1}, {u2}],\n "
                 "k-axes: {k}, trs: {trs},\n "
                 "bz: {bz},\n "
                 "semi-infinite:\n  {semi},\n "
                 "surface:\n  {surface}\n}}").format(**d)

    def set_options(self, **options):
        r""" Update options in the real-space self-energy

        After updating options one should re-call `initialize` for consistency.

        Parameters
        ----------
        semi_bulk : bool, optional
            whether the semi-infinite matrix elements are used for in the surface. Default to true.
        eta : float, optional
            imaginary part in the self-energy calculations (default 1e-4 eV)
        dk : float, optional
            fineness of the default integration grid, specified in units of Ang, default to 1000 which
            translates to 1000 k-points along reciprocal cells of length 1. Ang^-1.
        bz : BrillouinZone, optional
            integration k-points, if not passed the number of k-points will be determined using
            `dk` and time-reversal symmetry will be determined by `trs`, the number of points refers
            to the unfolded system.
        trs : bool, optional
            whether time-reversal symmetry is used in the BrillouinZone integration, default
            to true.
        """
        self._options.update(options)

    def real_space_parent(self):
        r""" Fully expanded real-space surface parent

        Notes
        -----
        The returned object does *not* obey the ``semi_bulk`` option. I.e. the matrix elements
        correspond to the `self.surface` object, always!
        """
        if np.allclose(self._unfold, 1):
            P0 = self.surface.copy()
        else:
            P0 = self.surface
        for ax in range(3):
            if self._unfold[ax] == 1:
                continue
            P0 = P0.tile(self._unfold[ax], ax)
        nsc = array_replace(P0.nsc, (self._k_axes, 1))
        P0.set_nsc(nsc)
        return P0

    def real_space_coupling(self, ret_indices=False):
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
        n_unfold = np.prod(self._unfold)

        # There are 2 things to check:
        #  1. The semi-infinite system
        #  2. The full surface
        PC_k = self.semi.spgeom0
        PC_semi = self.semi.spgeom1
        if np.allclose(self._unfold, 1):
            PC = self.surface.copy()
        else:
            PC = self.surface
        for ax in range(3):
            if self._unfold[ax] == 1:
                continue
            PC_k = PC_k.tile(self._unfold[ax], ax)
            PC_semi = PC_semi.tile(self._unfold[ax], ax)
            PC = PC.tile(self._unfold[ax], ax)

        # If there are any axes that still has k-point sampling (for e.g. circles)
        # we should remove that periodicity before figuring out which atoms that connect out.
        # This is because the self-energy should *only* remain on the sites connecting
        # out of the self-energy used. The k-axis retains all atoms, per see.
        nsc = array_replace(PC_k.nsc, (k_ax, None), (self.semi.semi_inf, None), other=1)
        PC_k.set_nsc(nsc)
        nsc = array_replace(PC_semi.nsc, (k_ax, None), (self.semi.semi_inf, None), other=1)
        PC_semi.set_nsc(nsc)
        nsc = array_replace(PC.nsc, (k_ax, None), other=1)
        PC.set_nsc(nsc)

        # Now we need to figure out the coupled elements
        # In all cases we remove the inner cell components
        def get_connections(PC, nrep=1, na=0, na_off=0):
            # Geometry short-hand
            g = PC.geometry
            # Remove all inner-cell couplings (0, 0, 0) to figure out the
            # elements that couple out of the real-space region
            n = PC.shape[0]
            idx = g.sc.sc_index([0, 0, 0])
            cols = _a.arangei(idx * n, (idx + 1) * n)
            csr = PC._csr.copy([0]) # we just want the sparse pattern, so forget about the other elements
            csr.delete_columns(cols, keep_shape=True)
            # Now PC only contains couplings along the k and semi-inf directions
            # Extract the connecting orbitals and reduce them to unique atomic indices
            orbs = g.osc2uc(csr.col[array_arange(csr.ptr[:-1], n=csr.ncol)], True)
            atom = g.o2a(orbs, True)
            expand(atom, nrep, na, na_off)
            return atom

        def expand(atom, nrep, na, na_off):
            if nrep > 1:
                la = np.logical_and
                off = na_off - na
                for rep in range(nrep - 1, 0, -1):
                    r_na = rep * na
                    atom[la(r_na + na > atom, atom >= r_na)] += rep * off

        # The semi-infinite direction is a bit different since now we want what couples out along the
        # semi-infinite direction
        atom_semi = []
        for atom in PC_semi.geometry:
            if len(PC_semi.edges(atom)) > 0:
                atom_semi.append(atom)
        atom_semi = _a.arrayi(atom_semi)
        expand(atom_semi, n_unfold, self.semi.spgeom1.geometry.na, self.surface.geometry.na)
        atom_k = get_connections(PC_k, n_unfold, self.semi.spgeom0.geometry.na, self.surface.geometry.na)
        atom = get_connections(PC)
        del PC_k, PC_semi, PC

        # Now join the lists and find the unique set of atoms
        atom_idx = np.unique(np.concatenate([atom_k, atom_semi, atom]))

        # Only retain coupling atoms
        # Remove all out-of-cell couplings such that we only have inner-cell couplings
        # Or, if we retain periodicity along a given direction, we will retain those
        # as well.
        PC = self.surface
        for ax in range(3):
            if self._unfold[ax] == 1:
                continue
            PC = PC.tile(self._unfold[ax], ax)
        PC = PC.sub(atom_idx)

        # Remove all out-of-cell couplings such that we only have inner-cell couplings.
        nsc = array_replace(PC.nsc, (k_ax, 1))
        PC.set_nsc(nsc)

        if ret_indices:
            return PC, atom_idx
        return PC

    def initialize(self):
        r""" Initialize the internal data-arrays used for efficient calculation of the real-space quantities

        This method should first be called *after* all options has been specified.

        If the user hasn't specified the ``bz`` value as an option this method will update the internal
        integration Brillouin zone based on the ``dk`` option. The :math:`\mathbf k` point sampling corresponds
        to the number of points in the non-folded system and thus the final sampling is equivalent to the
        sampling times the unfolding (per :math:`\mathbf k` direction).
        """
        P0 = self.real_space_parent()
        V_atoms = self.real_space_coupling(True)[1]
        self._calc = {
            # Used to calculate the real-space self-energy
            "P0": P0.Pk,
            "S0": P0.Sk,
            # Orbitals in the coupling atoms
            "orbs": P0.a2o(V_atoms, True).reshape(-1, 1),
        }

        # Update the BrillouinZone integration grid in case it isn't specified
        if self._options["bz"] is None:
            # Update the integration grid
            # Note this integration grid is based on the big system.
            sc = self.surface.sc * self._unfold
            rcell = fnorm(sc.rcell)[self._k_axes]
            nk = _a.onesi(3)
            nk[self._k_axes] = np.ceil(self._options["dk"] * rcell).astype(np.int32)
            self._options["bz"] = MonkhorstPack(sc, nk, trs=self._options["trs"])

    def self_energy(self, E, k=(0, 0, 0), bulk=False, coupling=False, dtype=None, **kwargs):
        r""" Calculate real-space surface self-energy

        The real space self-energy is calculated via:

        .. math::
            \boldsymbol\Sigma^{\mathcal{R}}(E) = \mathbf S^{\mathcal{R}} E - \mathbf H^{\mathcal{R}}
               - \Big[\sum_{\mathbf k} \mathbf G_{\mathbf k}(E)\Big]^{-1}

        Parameters
        ----------
        E : float/complex
           energy to evaluate the real-space self-energy at
        k : array_like, optional
           only viable for 3D bulk systems with real-space self-energies along 2 directions.
           I.e. this would correspond to circular self-energies.
        bulk : bool, optional
           if true, :math:`\mathbf S^{\mathcal{R}} E - \mathbf H^{\mathcal{R}} - \boldsymbol\Sigma^\mathcal{R}`
           is returned, otherwise :math:`\boldsymbol\Sigma^\mathcal{R}` is returned
        coupling : bool, optional
           if True, only the self-energy terms located on the coupling geometry (`coupling_geometry`)
           are returned
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        **kwargs : dict, optional
           arguments passed directly to the ``self.surface.Pk`` method (not ``self.surface.Sk``), for instance ``spin``
        """
        if dtype is None:
            dtype = complex128
        if E.imag == 0:
            E = E.real + 1j * self._options["eta"]

        # Calculate the real-space Green function
        G = self.green(E, k, dtype=dtype)

        if coupling:
            orbs = self._calc["orbs"]
            iorbs = delete(_a.arangei(len(G)), orbs).reshape(-1, 1)
            SeH = self._calc["S0"](k, dtype=dtype) * E - self._calc["P0"](k, dtype=dtype, **kwargs)
            if bulk:
                return solve(G[orbs, orbs.T], eye(orbs.size, dtype=dtype) - matmul(G[orbs, iorbs.T], SeH[iorbs, orbs.T].toarray()), True, True)
            return SeH[orbs, orbs.T].toarray() - solve(G[orbs, orbs.T], eye(orbs.size, dtype=dtype) - matmul(G[orbs, iorbs.T], SeH[iorbs, orbs.T].toarray()), True, True)

            # Another way to do the coupling calculation would be the *full* thing
            # which should always be slower.
            # However, I am not sure which is the most numerically accurate
            # since comparing the two yields numerical differences on the order 1e-8 eV depending
            # on the size of the full matrix G.

            #orbs = self._calc["orbs"]
            #iorbs = _a.arangei(orbs.size).reshape(1, -1)
            #I = zeros([G.shape[0], orbs.size], dtype)
            # Set diagonal
            #I[orbs.ravel(), iorbs.ravel()] = 1.
            #if bulk:
            #    return solve(G, I, True, True)[orbs, iorbs]
            #return (self._calc["S0"](k, dtype=dtype) * E - self._calc["P0"](k, dtype=dtype, **kwargs))[orbs, orbs.T].toarray() \
            #    - solve(G, I, True, True)[orbs, iorbs]

        if bulk:
            return inv(G, True)
        return (self._calc["S0"](k, dtype=dtype) * E - self._calc["P0"](k, dtype=dtype, **kwargs)).toarray() - inv(G, True)

    def green(self, E, k=(0, 0, 0), dtype=None, **kwargs):
        r""" Calculate the real-space Green function

        The real space Green function is calculated via:

        .. math::
            \mathbf G^\mathcal{R}(E) = \sum_{\mathbf k} \mathbf G_{\mathbf k}(E)

        Parameters
        ----------
        E : float/complex
           energy to evaluate the real-space Green function at
        k : array_like, optional
           only viable for 3D bulk systems with real-space Green functions along 2 directions.
           I.e. this would correspond to a circular real-space Green function
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        **kwargs : dict, optional
           arguments passed directly to the ``self.surface.Pk`` method (not ``self.surface.Sk``), for instance ``spin``
        """
        opt = self._options

        # Retrieve integration k-grid
        bz = opt["bz"]
        try:
            # If the BZ implements TRS (MonkhorstPack) then force it
            trs = bz._trs >= 0
        except:
            trs = opt["trs"]

        if dtype is None:
            dtype = complex128

        # Now we are to calculate the real-space self-energy
        if E.imag == 0:
            E = E.real + 1j * opt["eta"]

        # Used k-axes
        k_ax = self._k_axes

        k = _a.asarrayd(k)
        is_k = np.any(k != 0.)
        if is_k:
            axes = [self.semi.semi_inf] + k_ax.tolist()
            if np.any(k[axes] != 0.):
                raise ValueError(f"{self.__class__.__name__}.green requires k-point to be zero along the integrated axes.")
            if trs:
                raise ValueError(f"{self.__class__.__name__}.green requires a k-point sampled Green function to not use time reversal symmetry.")
            # Shift k-points to get the correct k-point in the larger one.
            bz._k += k.reshape(1, 3)

        # Self-energy function
        #SE = self.semi.self_energy
        def SE(E, k, bulk):
            
        M0 = self.surface
        M0Pk = M0.Pk

        #getrf = linalg_info("getrf", dtype)
        #getri = linalg_info("getri", dtype)
        #getri_lwork = linalg_info("getri_lwork", dtype)
        #lwork = int(1.01 * _compute_lwork(getri_lwork, M0.shape[0]))
        #def inv(A):
        #    lu, piv, info = getrf(A, overwrite_a=True)
        #    if info == 0:
         #       x, info = getri(lu, piv, lwork=lwork, overwrite_lu=True)
         #   if info != 0:
         #       raise ValueError(f"{self.__class__.__name__}.green could not compute the inverse.")
         #   return x
        inv = np.linalg.inv
        if M0.orthogonal:
            # Orthogonal *always* identity
            S0E = eye(len(M0), dtype=dtype) * E
            def _calc_green(k, dtype, surf_orbs, semi_bulk):
                invG = S0E - M0Pk(k, dtype=dtype, format="array", **kwargs)
                if semi_bulk:
                    invG[surf_orbs, surf_orbs.T] = SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                else:
                    invG[surf_orbs, surf_orbs.T] -= SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                return inv(invG)
        else:
            M0Sk = M0.Sk
            def _calc_green(k, dtype, surf_orbs, semi_bulk):
                invG = M0Sk(k, dtype=dtype, format="array") * E - M0Pk(k, dtype=dtype, format="array", **kwargs)
                if semi_bulk:
                    invG[surf_orbs, surf_orbs.T] = SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                else:
                    invG[surf_orbs, surf_orbs.T] -= SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                return inv(invG)

        # Create functions used to calculate the real-space Green function
        # For TRS we only-calculate +k and average by using G(k) = G(-k)^T
        # The extra arguments is because the internal decorator is actually pretty slow
        # to filter out unused arguments.

        # Define Bloch unfolding routine and number of tiles along the semi-inf direction
        bloch = Bloch(self._unfold)

        # If using Bloch's theorem we need to wrap the Green function calculation
        # as the method call.
        if len(bloch) > 1:
            def _func_bloch(k, dtype, surf_orbs, semi_bulk):
                return bloch(_calc_green, k, dtype=dtype, surf_orbs=surf_orbs, semi_bulk=semi_bulk)
        else:
            _func_bloch = _calc_green

        # calculate the Green function
        G = bz.apply.average(_func_bloch)(dtype=dtype,
                                          surf_orbs=self._surface_orbs,
                                          semi_bulk=opt["semi_bulk"])

        if is_k:
            # Restore Brillouin zone k-points
            bz._k -= k.reshape(1, 3)

        if trs:
            # Faster to do it once, than per G
            return (G + G.T) * 0.5
        return G

    def clear(self):
        """ Clears the internal arrays created in `initialize` """
        del self._calc

