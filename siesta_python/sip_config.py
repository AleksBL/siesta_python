"""Dataclass holding the SiP.__init__ options.

Single source of truth for the constructor defaults. `SiP.__init__` accepts the
same keyword names as before, builds a SiPConfig from them, rejects unknown
keywords, and materialises each field as a flat attribute.
"""
from dataclasses import dataclass, field

from sip_shared import (wd as _c_wd, PP_path as _c_PP_path, sp as _c_sp,
                        tb_trans_exec as _c_tb_trans_exec, gen_basis as _c_gen_basis,
                        M as _c_M, pdos_exec as _c_pdos_exec, gulp_path as _c_gulp_path,
                        B as _c_B, mix_ps as _c_mix_ps)


@dataclass
class SiPConfig:
    sm: str = 'siesta'
    sl: str = 'siesta'
    spin_pol: str = 'non-polarized'
    energy_shift: object = None
    dm_mixing_weight: object = None
    number_pulay: object = None
    dm_tol: str = '1.d-4'
    xc: str = 'GGA'
    xc_authors: str = 'PBE'
    basis: str = 'SZP'
    mesh_cutoff: int = 150
    max_scf_it: int = 300
    kgridcutoff: object = None
    solution_method: object = None
    electronic_temperature_mev: int = 25
    calc_bands: bool = False
    kp: list = field(default_factory=lambda: [1, 1, 1])
    k_shift: float = 0.0
    TwoDim: bool = True
    write_matrices: bool = True
    pp_path: str = _c_PP_path
    wd: str = _c_wd
    siesta_exec: str = _c_sp
    tb_trans_exec: str = _c_tb_trans_exec
    gen_basis: str = _c_gen_basis
    mpi: str = _c_M
    PDOS_EXEC: str = _c_pdos_exec
    Gulp_exec: str = _c_gulp_path
    Bands_plot: str = _c_B
    mix_ps: str = _c_mix_ps
    Standardize_cell: bool = False
    elec_inds: object = None
    print_console: bool = False
    Chem_Pot: object = None
    semi_inf: object = None
    elecs: list = field(default_factory=list)
    buffer_atoms: list = field(default_factory=list)
    Device_atoms: list = field(default_factory=list)
    kp_tbtrans: list = field(default_factory=lambda: [1, 1, 1])
    save_potentials: bool = False
    trans_emin: int = -1
    trans_emax: int = 1
    trans_delta: float = 0.02
    custom_tbtrans_contour: object = None
    NEGF_calc: bool = False
    reuse_dm: bool = False
    set_pdos: object = None
    overwrite: bool = True
    mixer: str = 'Pulay'
    n_hist: int = 8
    elec_bloch: object = None
    mix_what: str = 'Hamiltonian'
    elec_RSSE: bool = False
    elec_SurRSSE: bool = False
    elec_2E_RSSE: bool = False
    elec_3D_RSSE: bool = False
    save_SE: bool = False
    save_SE_only: bool = False
    lua_script: object = None
    dictionary: object = None
    skip_distance_check: bool = False
    skip_spg_sp: bool = False
    print_mulliken: bool = False
    denchar_files: bool = False
    contour_settings: object = None
