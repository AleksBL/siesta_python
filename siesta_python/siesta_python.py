from scipy.sparse import csr_matrix
from datetime import datetime
from subprocess import Popen
import numpy as np
import os
import sys
import sisl
import seekpath
import spglib as spg
from time import time
sys.path.append(__file__[:-16])
from funcs import unique_list, Num2Sym, read_analyze, write_fdf, write_gin, read_geom_from_fdf
from funcs import read_gulp_results, read_total_energy, read_fermi_level, write_relax_fdf, read_siesta_relax, Mulliken, Identity
from funcs import get_btd_partition_of_matrix as _BTDP
from funcs import write_mpr, write_dftb_hsd
from funcs import read_contour_from_failed_RSSE_calc, recreate_old_calculation, unique_list, barebones_RUN
from dftbplus_interface import construct_sparse, read_mulliken_from_detailedout, constructHk
from dftbplus_interface import has_equiv_in_l as equiv_dftb
from dftbplus_interface import loadtxt as loadtxt_dftb
from config import GfModule_extension

if GfModule_extension:
    from Gf_Module.Gf import read_SE_from_tbtrans

terminal = os.system; rem = os.remove; ld = os.listdir
wd = os.getcwd()

PP_path = '../pp'
#put siesta and tbtrans in path....#
#M             = 'mpirun '

sp            = 'siesta '
tb_trans_exec = 'tbtrans '
B             = 'gnubands '
pdos_exec     = 'fmpdos'
gulp_path     = 'gulp '
gen_basis     = 'gen-basis '
mix_ps        = 'mixps '
joblib_backend= 'multiprocessing'

M             = ''
try:
    M = os.environ['mpi_exec']
except:
    pass

def error():
    """  
        Throw an error.
    """
    
    print('Error in siesta_python')
    assert 1==0

no_rem = ['siesta_python.py', 
          'fdf_read.py', 
          'divacancy585', 
          '__pycache__',
          'funcs.py',
          'GB.xsf',
          'Structs',
          'Save_DM',
          'Save_HS',
          '.npy',
          'Save_ES',
          'Save_TBT_nc',
          'Save_TP'
          ]

# def clean_dir():
#     for fname in ld(wd):
#         if fname in no_rem or '.py' in fname or 'Struct.fdf' in fname:
#             pass
#         else:
#             if os.path.isdir(fname)==True: 
#                 pass
#             else:
#                 rem(wd+'/'+fname)

class SiP:
    ######"Siesta in Python"#####
    # ASE-like interface for managing transport calculations with Transiesta and  TBtrans + more
    def __init__(self,lat,pos,s, # lat: (3,3) np array, pos:(n_at,3) np.array, s (n_at,) np array of integers
                 directory_name='noname',
                 sm = 'siesta',
                 sl = 'siesta',
                 spin_pol = 'non-polarized',
                 energy_shift = None,      
                 dm_mixing_weight = None, # Mixer Opts
                 number_pulay = None,     # Mixer Opts
                 dm_tol = '1.d-4',
                 xc     = 'GGA',
                 xc_authors='PBE',
                 basis  = 'SZP',
                 mesh_cutoff = 150,
                 max_scf_it  = 300, 
                 kgridcutoff = None,
                 solution_method = None,#'diagon',
                 electronic_temperature_mev = 25,
                 calc_bands = False,              # Calculate band structure in sisl instead
                 kp         = [1,1,1],
                 k_shift    = 0.0,
                 TwoDim     = True,
                 write_matrices = True,
                 pp_path    = PP_path,
                 wd         = wd,
                 siesta_exec= sp,
                 tb_trans_exec = tb_trans_exec,
                 gen_basis  = gen_basis,
                 mpi        = M,
                 PDOS_EXEC=pdos_exec,
                 Gulp_exec=gulp_path,
                 Bands_plot = B,
                 mix_ps = mix_ps,
                 Standardize_cell=False,        # standardizes cell to seekpath convention
                 elec_inds  = None,             # set here or use method find_elec_inds
                 print_console = False,         # you want to all the printing?
                 Chem_Pot   = None,             # list of floats (2 elecs: [.0, .0])
                 semi_inf   = None,             # '+a1', '-a1' etc
                 elecs      = [],               # If solution_method is transiesta, you'll need to supply some electrodes
                 buffer_atoms = [],             # set here or use the method set_buffer_atoms
                 Device_atoms = [],             # ???
                 kp_tbtrans = [1,1,1],          # 
                 #save_EP    = False,           # ???
                 save_potentials=False,         # VT and VH
                 trans_emin = -1,               # Values for tbtrans transport calculation
                 trans_emax =  1,               #
                 trans_delta= 0.02,             # 
                 custom_tbtrans_contour = None, # supply a custom tbtrans contour (n_E,) np array
                 NEGF_calc  = False,            # toggle non-equilibrium flags in fdf files. Autoset when Chem_Pot!=0
                 reuse_dm   = False,            # Reuse siesta.DM / siesta.TSDE file
                 set_pdos   = None,             # ???
                 #save_es    = False,            # ???
                 overwrite  = True,             # True, False, 'reuse', overwrite directory folder?
                 mixer      = 'Pulay',          # SCF mixer, see siesta manual
                 n_hist     = 8,                # history for mixer, see siesta manual
                 elec_bloch = None,             # bloch tiling of electrode, see siesta/tbtrans manual
                 mix_what   = 'Hamiltonian',    # 'Hamiltonian', 'Density', 'Charge', see transiesta manual
                 elec_RSSE  = False,            # Electrode is real space electrode? 
                 elec_SurRSSE = False,          # Electrode is semi-infinite real space electrode?
                 elec_2E_RSSE = False,
                 elec_3D_RSSE = False,
                 save_SE    = False,            # write *TBT_SE.nc files?
                 save_SE_only = False,          # do nothing else but calculate SE?
                 lua_script = None,             # does nothing yet
                 dictionary = None,             # ???
                 skip_distance_check = False,   # Code checks distances between atoms, can be lengthy for big systems
                 skip_spg_sp = False,           # spacegroup of system, --||--
                 print_mulliken = False,        # mulliken in out file?
                 denchar_files = False,         # 
                 contour_settings=None,         # Custom transiesta contour (dictionary). See siesta_python.funcs for implementation.
                 
                 ):
        
        """
            Very long but fairly easy to understand initialisation of the
            instance of the siesta_python class. You will find most of the
            keywords refer to a specfic siesta parameter. (Read the siesta 
            documentation)
            Meant as an ASE-like interface for managing transport calculations with Transiesta and  TBtrans + more
            Often modifed keywords are the 
                directory_name: the folder name of the calculation
                basis: (SZ, SZP, DZ etc. see siesta docs.)
                kp  and kp_tbtrans : [nk1, nk2, nk3] list of how many kpoints to
                                     use in each direction
                elecs : None if the instance is not a device region 
                        if the instance is the central device region,
                        specify the elecs as such: [EM,EP,...] where EM and 
                        EP are also siesta_python instances, that have/will be
                        run before any transiesta calculations are done
                        using this instance.
                        solution_method: diagon/transiesta
                Chem_Pot: list of floats describing the chemical potential of the
                          electrodes
                Read the comments in the code for more info.
                semi_inf: string (+a1, -a1 ,-a2, a3 etc., ab for RSSE ) designates the direction
                        of periodicity of the electrode. 
                pp_path: location of a folder with pseudopotentials 
                         They should be named like Au.gga.psf/ Au.lda.psf
                                                    C.gga.psf
        The interested reader is furthermore refered to many of the tutorials
        in the Zandpack code which uses this code extensively. 
        
        """
        
        
        assert pos.shape[0] == len(s)
        Reuse=False
        for i in ld(wd):
            if directory_name==i:
                #print('Directory called '+i+' already exists! Remove it? (y/reuse/n)')
                if overwrite==True:
                    inp='y'
                elif overwrite == 'reuse':
                    inp = 'reuse'
                    print('reusing')
                else:
                    inp=input()
                if inp=='y':
                    for j in ld(wd+'/'+i):
                        name=wd+'/'+i+'/'+j
                        #print('file '+name+' removed!\n')
                        if '.py' in name or j in no_rem:
                            pass
                        else:
                            if j == 'orbital_directory' or j == 'lua_comm':
                                for k in ld(name):
                                    #print('file '+name+k+' removed!\n')
                                    rem(name+'/'+k)
                                os.rmdir(name)
                            else:
                                rem(name)
                    os.rmdir(i)
                elif inp=='reuse':
                    Reuse=True
                else:
                    assert 1==0
        
        if Reuse == False:
            os.mkdir(directory_name)
            os.mkdir(directory_name+'/orbital_directory')
            os.mkdir(directory_name+'/lua_comm')
        
        if Reuse == True:
            try:
                p,lat = read_geom_from_fdf(directory_name + '/' + 'STRUCT.fdf')
                pos = p[:,0:3]
                s   = p[:,  3]
            except:
                print('Couldnt read STRUCT.fdf, stopping')
                assert 1 == 0
            if 'siesta.TBT.nc' in ld(directory_name):
                os.system('mv ' + directory_name + '/siesta.TBT.nc '  + directory_name + '/old_siesta.TBT.nc ')
            if 'siesta.TBT_UP.nc' in ld(directory_name):
                os.system('mv ' + directory_name + '/siesta.TBT_UP.nc '  + directory_name + '/old_siesta.TBT_UP.nc ')
            if 'siesta.TBT_DN.nc' in ld(directory_name):
                os.system('mv ' + directory_name + '/siesta.TBT_DN.nc '  + directory_name + '/old_siesta.TBT_DN.nc ')
            
        ###### Initialize all the stuff neeeded to write a siesta .fdf file. 
        self.dir=directory_name
        self.Standardize_cell = Standardize_cell
        self.xc_authors = xc_authors
        self.xc               = xc
        self.skip_spg_sp = skip_spg_sp
        
        self.set_struct(lat, pos, s)
        
        self.sm               = sm
        self.sl               = sl
        self.spin_pol         = spin_pol
        self.energy_shift     = energy_shift
        self.dm_mixing_weight = dm_mixing_weight
        self.number_pulay     = number_pulay
        self.dm_tol           = dm_tol
        self.basis            = basis
        self.mesh_cutoff      = mesh_cutoff
        self.max_scf_it       = max_scf_it
        self.min_scf_it       = 3
        self.kgridcutoff      = kgridcutoff
        if solution_method is None:
            if len(elecs)==0:
                self.solution_method = 'diagon'
            else:
                self.solution_method = 'transiesta'
        else:
            self.solution_method  = solution_method
        self.electronic_temperature_mev = electronic_temperature_mev
        self.calc_bands       = calc_bands
        self.kp               = kp
        self.k_shift          = k_shift
        self.TwoDim           = TwoDim
        
        self.write_matrices   = write_matrices
        self.elecs            = elecs
        self.pp_path          = pp_path
        self.siesta_exec      = siesta_exec
        self.pdos_exec        = PDOS_EXEC
        self.gulp_exec        = Gulp_exec
        self.bands_plot       = Bands_plot
        self.wd               = wd
        self.elec_inds        = elec_inds
        self.mpi              = mpi + ' '
        self.mix_ps           = mix_ps
        if print_console:
            self.print_no_print = '|tee'
        else:
            self.print_no_print = '>'
        self.Chem_Pot         = Chem_Pot
        if Chem_Pot is None and len(self.elecs)>0:
            self.Chem_Pot = [0.0 for i in range(len(self.elecs)) ]
        if self.Chem_Pot is not None:
            _CP = self.Chem_Pot
            self.Voltage          = max(max(_CP) - min(_CP), max([abs(cp) for cp in _CP]))
            #self.Voltage          = max(Chem_Pot) - min(Chem_Pot)
        else:
            self.Voltage = None
        self.semi_inf         = semi_inf
        self.tb_trans_exec    = tb_trans_exec
        self.gen_basis_exec   = gen_basis
        self.kp_tbtrans       = kp_tbtrans
        #self.save_EP          = save_EP
        self.trans_emin       = trans_emin
        self.trans_emax       = trans_emax
        self.trans_delta      = trans_delta
        self.custom_tbtrans_contour = custom_tbtrans_contour
        if (np.array(Chem_Pot)!=0.0).any(): NEGF_calc = True
        self.NEGF_calc        = NEGF_calc
        self.reuse_dm         = reuse_dm
        self.dQ               = None
        self.set_pdos         = set_pdos
        #self.save_electrostatic_potential = save_es
        self.save_potential_grids=save_potentials
        self.buffer_atoms     = buffer_atoms
        self.Device_atoms     = Device_atoms
        self.mixer            = mixer
        self.n_hist           = n_hist
        self.elec_bloch       = elec_bloch
        self.mix_what         = mix_what
        self.elec_RSSE        = elec_RSSE
        self.elec_SurRSSE     = elec_SurRSSE
        self.elec_2E_RSSE     = elec_2E_RSSE
        self.elec_3D_RSSE     = elec_3D_RSSE
        if elec_SurRSSE: self.elec_RSSE = True
        self.save_SE          = save_SE
        self.save_SE_only     = save_SE_only
        self.lua_script       = lua_script
        self.dic              = dictionary
        self.print_mulliken   = print_mulliken
        self.contour_settings = contour_settings
        
        
        if skip_distance_check == True:
            pass
        else:
            self.Check_Distances()
        self.denchar_files          = denchar_files
        self.initialised_attributes =  [i for i in self.__dict__.keys()]
        
        non_ortho=False
        for i in range(3):
            for j in range(3):
                dot=self.lat[i,:].dot(self.lat[j,:])
                if i!=j and np.abs(dot)>1e-10:
                    non_ortho = True
        self.non_ortho=non_ortho
        
        if TwoDim==True and kp is not None and print_console and self.sym is not None:
            print('2-D crystal with space group ' + self.sym)
        if self.elec_bloch == [1,1,1]:
            self.elec_bloch = None            
    
    def reararange(self,idx):
        self.set_struct(self.lat, self.pos_real_space[idx], self.s[idx])
        
    def set_struct(self, lat, pos, s):
        self.lat   = np.array(lat)
        self.s     = s
        self.sname = [Num2Sym[i] for i in s]
        pos        = np.array(pos)
        A_mat      = self.lat.T
        ### lattice vectors now in columns
        A_mat_inv=np.linalg.inv(A_mat)
        positions=[]
        for i in range(len(pos[:,0])):
            fpos_i=np.dot(A_mat_inv,pos[i])
            positions += [fpos_i]
        
        self.pos   = np.array(positions)
        
        #Shitty name, yes, but now it is there. 
        self.pos_real_space = pos
        
        self.cell_spg = (self.lat,self.pos,np.abs(s))
        try:
            if self.skip_spg_sp:
                assert 1 == 0
            self.get_path =  seekpath.get_path_orig_cell(self.cell_spg,True)
        except:
            self.get_path = None
        try:
            if self.skip_spg_sp:
                assert 1 == 0
            self.sym_dic  = self.get_path['point_coords']
        except:
            self.sym_dic  = None
        try:
            if self.skip_spg_sp:
                assert 1 == 0
            self.k_path   = seekpath.get_path_orig_cell(self.cell_spg,True)['path']
        except:
            self.k_path   = None 
        try:
            if self.skip_spg_sp:
                assert 1 == 0
            self.sym      = spg.get_spacegroup(self.cell_spg,symprec=1e-3)
        except:
            self.sym      = None
        try:
            if self.skip_spg_sp:
                assert 1 == 0
            self.standard_cell =  spg.standardize_cell(self.cell_spg)
        except:
            self.standard_cell = None
        if self.Standardize_cell==True:
            self.standardise_cell()
        self.UC_idx = self.cell_offset()
    
    def standardise_cell(self):
        if self.standard_cell==None:
            pass
        else:    
            (self.lat,self.pos, self.s) = (self.standard_cell[0],  self.standard_cell[1], self.standard_cell[2])
            self.pos_real_space=np.array([self.lat.T.dot(self.pos[i,:]) for i in range(len(self.pos))])
    
    def set_buffer_atoms(self,condition):
        inds_buffer=[]
        for i in range(len(self.pos_real_space)):
            pi=self.pos_real_space[i,:]
            if condition(pi)==True:
                inds_buffer+=[i]
        if len(self.buffer_atoms) == 0:
            self.buffer_atoms = inds_buffer
        else:
            print('You have already set buffer atoms. Dont give these if you want to use this')
    
    def set_device_atoms(self,condition):
        inds_dev=[]
        for i in range(len(self.pos_real_space)):
            pi=self.pos_real_space[i,:]
            if condition(pi)==True:
                inds_dev+=[i]
        if len(self.Device_atoms) == 0:
            self.Device_atoms = inds_dev
        else:
            print('You have already set device atoms. Dont give these if you want to use this')
    
    def barebone_fdf(self, eta = 0.0):
        barebones_RUN(self, eta = eta)
    def make_bandpath(self,zdir=2):
        v = [self.sym_dic[i] for i in self.sym_dic.keys()]
        Names  = [i for i in self.sym_dic.keys()]
        bz_path_names=[]
        bz_path = []
        #bz_path_num=[1]
        
        for i,kp in enumerate(v):
            name = Names[i]
            if self.TwoDim:
                if abs(kp[zdir])<1e-10:
                    bz_path_names+=[name]
                    bz_path+=[kp]
                    #print(kp)
                    
            if self.TwoDim == False:
                bz_path_names+=[name]
                bz_path+=[kp]
        bz_path += [bz_path[0]]
        bz_path_names += [bz_path_names[0]]
        return bz_path, bz_path_names
    
    
    
    def fdf(self, eta = 0.0, manual_pp = [], fix_hartree = False, parallel_k=False):
        """
            Standardised fdf files for running siesta
        
        """
        
        write_fdf(self, eta = eta)
        if self.Voltage is None:
            pass
        elif self.Voltage!=0 and fix_hartree:
            self.write_more_fdf(['TS.Hartree.Fix -A'])
        self.get_pseudo_paths(manual = manual_pp)
        if hasattr(self, '_mixing_dic'):
            for k in self._mixing_dic.keys():
                atoms, frac = self._mixing_dic[k]
                self.ps_mixer(atoms, frac, k)
            self.dump_pseudo_list_to_struct()
        
        if self.dic != None:
            keys = self.dic.keys()
            keys = [k for k in keys if 'dftb' not in k]
            for k in keys:
                v = self.dic[k]
                if len(k.split())==1:
                    which = 'DEFAULT'
                else:
                    which = k.split(' ')[0]
                    k = k.split(' ')[1]
                self.write_more_fdf([k+ ' ' + v], name = which)
        if parallel_k:
            self.set_parallel_k()
        
    
    def fdf_relax(self, Constraints, force_tol = 0.01, max_it = 1000, 
                  parallel_k=False, variable_cell = False):
        if self.solution_method=='transiesta':
            self.fdf()
        write_relax_fdf(self, Constraints = Constraints, 
                        force_tol = force_tol, max_it = max_it,
                        variable_cell = variable_cell)
        if self.solution_method=='transiesta':
            self.write_more_fdf(['%include TS_TBT.fdf'], name = 'RUN')
        self.get_pseudo_paths()
        if parallel_k:
            self.set_parallel_k()
    
    def read_relax(self):
        p,l = read_siesta_relax(self.dir + '/'+self.sl+'.XV')
        self.relaxed_pos = p
        self.relaxed_lat = l
    
    def gin(self,cell_opti=[],f=None, var_cell = None, relax_only = False, phonons=False):
        fix=[]
        if f is None and var_cell is None:
            pass
        elif f is not None:
            for i in range(len(self.pos_real_space)):
                pi=self.pos_real_space[i,:]
                if f(pi)==True:
                    fix+=[[0,0,0]]
                else:
                    fix+=[[1,1,1]]
        elif var_cell is not None:
            for i in range(len(self.pos_real_space)):
                if i in var_cell:
                    fix+=[[1,1,1]]
                else:
                    fix+=[[0,0,0]]
        
        write_gin(self,cell_opti=cell_opti,fix=fix, relax_only = relax_only, phonons=phonons)
    
    def dftb_hsd(self, Scc = 'Yes', write_cell = True, angmom_dic = {},
                       ReadInititialCharges = 'No', WriteChargesAsText = 'Yes', ReadChargesAsText = 'No',
                       WriteRealHS='No', CalculateForces='No', SkipChargeTest='No',
                       ThirdOrderFull='No', HubDeriv=None,DampCor=None):
        
        write_dftb_hsd(self, Scc = Scc, write_cell = write_cell, angmom_dic = angmom_dic,
                             ReadInitialCharges = ReadInititialCharges, WriteChargesAsText=WriteChargesAsText,
                             ReadChargesAsText=ReadChargesAsText, WriteRealHS    = WriteRealHS, 
                             CalculateForces=CalculateForces,     SkipChargeTest = SkipChargeTest,
                             ThirdOrderFull = ThirdOrderFull, HubDeriv=HubDeriv,DampCor=DampCor)
        
        
        if self.dic is None:
            self.dic = {'dftb_angmom': angmom_dic}
        else:
            self.dic.update({'dftb_angmom': angmom_dic})
    
    def run_dftb_in_dir(self,silent = False, subprocess=False,wait=False):
        if silent==False:
            command = 'dftb+ '
        else:
            command = 'dftb+ > /dev/null'
        
        if subprocess:
            with open(os.devnull, 'wb') as devnull:
                if wait:
                    Popen('cd '+self.dir + ' && ' + 
                          command + ' && cd ..', shell = True,stdout = devnull,stderr =devnull).wait()
                else:
                    Popen('cd '+self.dir + ' && ' + 
                          command, shell = True   )
            return
        
        ###SCC RUN
        os.chdir(self.dir)
        os.system(command)
        os.chdir('..')
        
    
    def write_and_run_dftb_in_dir(self, angmom, skip_HS_write = False,
                                  skip_write_charges=False, ThirdOrderFull='No',
                                  HubDeriv = None, DampCor=None):
        ###SCC RUN
        self.dftb_hsd(angmom_dic = angmom, 
                      ThirdOrderFull=ThirdOrderFull, 
                      HubDeriv=HubDeriv,
                      DampCor=DampCor
                      )
        self.run_dftb_in_dir(angmom)
        #if skip_write_charges:
        #    return
        detailed_out = read_mulliken_from_detailedout(self.dir+'/detailed.out')
        self.dftb_mullikencharge_interface()
        Q0 = detailed_out['pop']
        self.dic['dftb_charge'].set_Q(Q0)
        self.dic['dftb_charge'].read_Q(label = 'init_Q')
        
        if skip_HS_write:
            return
        os.chdir(self.dir)
        os.system('mv band.out Band.out')
        os.system('mv detailed.out Detailed.out')
        os.chdir('..')
        ### Write HS run
        self.dftb_hsd(angmom_dic           = angmom,
                      WriteRealHS          = 'Yes', 
                      ReadChargesAsText    = 'Yes',
                      ReadInititialCharges = 'Yes',
                      SkipChargeTest       = 'Yes',
                      ThirdOrderFull=ThirdOrderFull, 
                      HubDeriv=HubDeriv,
                      DampCor=DampCor
                      )
        
        self.run_dftb_in_dir(angmom)
    
    
    def read_dftb_hamiltonian(self):
        H_ele, RH, iH = loadtxt_dftb(self.dir+'/hamreal1.dat')
        S_ele, RS, iS = loadtxt_dftb(self.dir+'/overreal.dat')
        Roff          = self.cell_deviation()
        DR            = Roff[RH[:,0]-1] - Roff[RH[:,1]-1]
        RH[:,2:]     += DR
        RS[:,2:]     += DR
        
        for h in H_ele:
            h*=27.211396641308 # Rydberg to eV
        if self.dic is None or 'dftb_equivH' not in self.dic.keys():
            self.dftb_find_equiv(RH,'H')
        if self.dic is None or 'dftb_equivS' not in self.dic.keys():
            self.dftb_find_equiv(RS,'S')
        return H_ele, RH, iH, S_ele, RS, iS
    
    def dftb2sisl(self, tol = 1e-5, gamma_only = False):
        # Read
        H_ele, RH, iH, S_ele, RS, iS = self.read_dftb_hamiltonian()
        # Make dummy orbitals
        if len(iH.shape)==1:
            iH = iH[None,:]
            iS = iS[None,:]
        sislAtoms = [sisl.Atom(self.s[i], orbitals = [sisl.Orbital(R = 5.0) 
                                                      for j in range(iH[i,2])]) 
                     for i in range(len(iH))]
        # SISL
        geom = sisl.Geometry(self.pos_real_space, atoms = sislAtoms, lattice = self.lat, )
        
        H    = sisl.Hamiltonian(geom, orthogonal = False)
        maxx, maxy, maxz = np.abs(RH[:,2]).max(), np.abs(RH[:,3]).max(), np.abs(RH[:,4]).max()
        if gamma_only:
            H.set_nsc((1,1,1))
        else:
            H.set_nsc((maxx*2+1,maxy*2+1, maxz*2+1))
        # Construct sparse matrices
        Roff = None#+(self.center_cell() - self.UC_idx) 
        Hsp  = construct_sparse(H_ele, RH, iH, self.dic['dftb_equivH'], H.sc, 
                                tol  = tol,  gamma_only = gamma_only, 
                                Roff = Roff
                                )
        
        Ssp  = construct_sparse(S_ele, RS, iS, self.dic['dftb_equivS'], H.sc, 
                                tol  = tol, gamma_only = gamma_only, 
                                Roff = Roff
                                )
        
        H    = H.fromsp(H.geometry, Hsp, Ssp)
        return H
    
    def fast_dftb_hk(self, 
                     k     = np.array((.0, .0, .0)), 
                     label = 'hamreal1', 
                     gamma_only = False):
        H_ele, RH, iH  = loadtxt_dftb(self.dir+'/'+label+'.dat')
        if 'hamreal' in label:
            eq = self.dic['dftb_equivH']
        else:
            eq = self.dic['dftb_equivS']
        hk = constructHk(k, H_ele, RH, iH, eq)
        if 'hamreal' in label:
            hk*=27.21139664130
        return hk
    
    def dftb_find_equiv(self, R,label):
        equiv = equiv_dftb(R)
        if self.dic is None:
            self.dic = {'dftb_equiv'+label: equiv}
        else:
            self.dic.update({'dftb_equiv'+label: equiv})
        return equiv
    
    def dftb_mullikencharge_interface(self):
        angmom2norb = {'s': 1, 'p': 4, 'd': 9, 'f': 16}
        angmom = self.dic['dftb_angmom']
        norb   = np.array([angmom2norb[angmom[Num2Sym[si]]] for si in self.s])
        Charge = dftb_charge(self.dir+'/charges.dat', self.s, norb)
        if self.dic is None:
            self.dic = {'dftb_charge': Charge}
        else:
            self.dic.update({'dftb_charge': Charge})
    
    def get_pseudo_paths(self, manual = [], labs = None):
        paths=[]
        
        atoms = unique_list(self.s) + manual
        atoms = [a for a in atoms if a <= 200]
        
        Dir   = self.pp_path
        PPs   = ld(Dir)
        for atom in atoms:
            for pp in PPs:
                name=''
                for let in pp:
                    if let=='.':
                        break
                    else:
                        name+=let
                 
                if Num2Sym[atom].replace('_ghost', '') == name and self.xc.lower() in pp:
                    paths+=['../'+self.pp_path+'/'+pp]
        self.pseudo_paths = paths
        it=0
        for p in paths:
            if '.psf' in p:
                os.system('ln -s '+ p +' '+self.dir+'/'+ Num2Sym[atoms[it]]+'.psf')
                it+=1
            if '.psml' in p:
                os.system('ln -s '+ p +' '+self.dir+'/'+ Num2Sym[atoms[it]]+'.psml')
                it+=1
        
    
    def add_elecs(self,L):
        self.elecs = L
    
    def find_elec_inds(self, tol = 1e-2, correct_electrode_types = False, fast_way = True):
        """
            function finds and shuffles the electrode indices to the indices furthest 
            back, but before the buffer atoms. 
            [Device idx, elec_1_idx, elec_2_idx,..., buffer_idx]
            The atomic positions in the electrodes needs to match the ones
            in self within a tolerance tol. 
            
        """
        
        inds = [ [] for e in self.elecs ]
        it_e = 0
        
        for e in self.elecs:
            if e.elec_RSSE == False:# or e.elec_SurRSSE:
                
                if isinstance(e.elec_bloch, list) or isinstance(e.elec_bloch, np.ndarray):
                    direcs = np.where(np.array(e.elec_bloch)>1)[0]#[0]
                    POS = e.pos_real_space.copy()
                    tiler =  e.pos_real_space.copy()
                    for _d in direcs:
                        A = e.lat[_d]
                        for i in range(1, e.elec_bloch[_d]):
                            POS = np.vstack((POS, A * i + tiler))
                        tiler = POS.copy()
                    #for i in range(1,e.elec_bloch[direc]):
                    #    POS = np.vstack((POS, A * i + e.pos_real_space))
                    #if (np.array(e.elec_bloch)>1).sum()>1:
                    #    print('Several bloch directions not implemented in siesta_python yet')
                    #    error()
                    
                    for i in range(len(POS)):
                        ri = POS[i]
                        if fast_way:
                            dij = np.linalg.norm(ri - self.pos_real_space,axis = 1)
                            idx = np.where(dij<tol)[0]
                            if len(idx)==0:
                                pass
                            elif len(idx)==1:
                                inds[it_e]+=[idx[0]]
                            else:
                                print('found double electrode index')
                                assert 1 == 0
                        else:
                            for j in range(len(self.pos_real_space)):
                                rj = self.pos_real_space[j]
                                if np.linalg.norm(ri-rj) < tol:
                                    inds[it_e]+=[j]
                else:
                    for i in range(len(e.pos_real_space)):
                        ri = e.pos_real_space[i]
                        if fast_way:
                            dij = np.linalg.norm(ri - self.pos_real_space,axis = 1)
                            idx = np.where(dij<tol)[0]
                            if len(idx)==0:
                                pass
                            elif len(idx)==1:
                                inds[it_e]+=[idx[0]]
                            else:
                                print('found double electrode index')
                                assert 1 == 0
                        else:
                            for j in range(len(self.pos_real_space)):
                                rj = self.pos_real_space[j]
                                if np.linalg.norm(ri-rj) < tol:
                                    inds[it_e]+=[j]
                        
            
            elif ((e.elec_RSSE or 
                     e.elec_surRSSE) 
                  and 
                 ((isinstance(e.elec_bloch, list) or 
                     isinstance(e.elec_bloch, np.ndarray)) == False)
                 ):
                R,S = e.get_RS_pos()
                for i in range(len(R)):
                    ri = R[i]
                    for j in range(len(self.pos_real_space)):
                        rj = self.pos_real_space[j]
                        if np.linalg.norm(ri-rj) < tol:
                            inds[it_e]+=[j]
            elif e.elec_RSSE or e.elec_surRSSE:
                R, S   = e.get_RS_pos()
                direcs = np.where(np.array(e.elec_bloch)>1)[0]#[0]
                POS    = R.copy(); tiler =  R.copy()
                for _d in direcs:
                    A = e.lat[_d]
                    for i in range(1, e.elec_bloch[_d]):
                        POS = np.vstack((POS, A * i + tiler))
                    tiler = POS.copy()
                for i in range(len(POS)):
                    ri  = POS[i]
                    dij = np.linalg.norm(ri - self.pos_real_space,axis=1)
                    idx = np.where(dij<tol)[0]
                    if len(idx)==0:
                        pass
                    elif len(idx)==1:
                        inds[it_e]+=[idx[0]]
                    else:
                        print('found double electrode index')
                        assert 1 == 0

                
            
            it_e += 1
        
        idx = np.arange(len(self.pos_real_space))
        np_inds = np.hstack([np.array(ind) for ind in inds])
        
        idx = np.delete(idx, np_inds)
        
        inds_new = []
        n=len(idx)
        
        for idx_e in inds:
            idx = np.hstack((idx, np.array(idx_e)))
            inds_new +=[np.arange(n, n + len(idx_e))]
            n+=len(idx_e)
        
        self.pos_real_space = self.pos_real_space[idx]
        self._rearange_indices = idx
        
        print(inds)
        if correct_electrode_types == True:
            it=0
            for e in self.elecs:
                te = e.s.copy()
                
                if isinstance(e.elec_bloch, list) or isinstance(e.elec_bloch, np.ndarray):
                    
                    te = np.hstack([te for i in range(max(e.elec_bloch))])
                
                self.s[inds[it]] = te
                it+=1
        
        self.s = self.s[idx]
        
        if self.elec_inds is None:
            self.elec_inds = inds_new
        else:
            print('elec inds already given! remove these if you want to use this function\n')
    
    def set_parallel_k(self):
        """ Toggles DiagK siesta option"""
        if len(self.elecs) == 0:
            self.write_more_fdf(['Diag.ParallelOverK True\n'])
    
    def make_device_coords(self):
        if len(self.elecs) == 0:
            print('No electrodes added')
        else:
            pos_new = [self.pos_real_space[i,:] for i in range(len(self.pos[:,0]))];
            atoms_new = [self.s[i] for i in range(len(self.s))]
            elec_inds = []
            ind =len(self.pos[:,0])
            
            for elec in self.elecs:
                na = len(elec.pos[:,0])
                pos_new   += [elec.pos_real_space[i,:] for i in range(na)]
                atoms_new += [elec.s[i] for i in range(na)]
                elec_inds +=[np.arange(0,na)+ind]
                ind+=na
            return np.array(pos_new), np.array(atoms_new), elec_inds
    
    def write_more_fdf(self, L,name='DEFAULT'):
        """
            After self.fdf() has been called, you can add more to the
            fdf file called name afterwards with this function.
            
            L is a list of string like ['TBT.DOS Gf True \n', ...]
        """
        with open(self.dir+'/'+name+'.fdf','a') as f:
            f.write('\n')
            for l in L:
                f.write(l)
            f.write('\n')
            f.close()
    
    def write_analysis_results(self):
        """
        after TBtrans analyze step, write the results to the TS_TBT file.
        
        """
        WRITE=read_analyze(self)
        self.write_more_fdf(WRITE,name = 'TS_TBT')
    
    def run_tbt_analyze_in_dir(self):
        """ See TBtrans documentation for the use of the analyze step.
            This function simply writes the lowest memory result to 
            the TS_TBT file.
        """
        
        
        command =  self.mpi + self.tb_trans_exec  + ' -fdf TBT.Analyze RUN.fdf > RUN_Analyze.out'
        os.chdir(self.dir)
        os.system(command)
        os.chdir('..')
        WRITE = read_analyze(self)
        self.write_more_fdf(WRITE, name = 'TS_TBT')
    
    def run_siesta_in_dir(self, use_subprocess = False, wait = True):
        """
            Runs siesta on the RUN.fdf file located in the directory 
            specified by self.directory_name
            use_subprocess specifies if a subprocess should be spawned, which
            is a background process that does not block further use of python
            until the siesta calculation finishes. The default mode blocks 
            any further python use before the siesta calculation finishes.
            
        
        """
        
        
        
        if self.solution_method=='diagon':
            print('Running Siesta calculation in Directory: '+self.dir+ '!\n')
        elif self.solution_method=='transiesta':
            print('Running TranSiesta calculation in Directory: '+self.dir+ '!\n')
        n='RUN'    
        
        command = self.mpi+self.siesta_exec+'<'+n+'.fdf '+self.print_no_print+' '+n+'.out'
        
        if use_subprocess:
            if wait:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   ).wait()
            else:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   )
            return
        
        os.chdir(self.dir)
        os.system(command)
        os.chdir('..')
    
    
    def run_gulp_in_dir(self, mpi_run =False):
        """ Gulp calculation, inspect code. """
        print('Running Gulp calculation in Directory: '+self.dir+ '!\n')
        os.chdir(self.dir)
        if mpi_run == True:
            os.system(self.mpi + self.gulp_exec+'< Gulp.gin > Gulp.gout')
        else:
            os.system(self.gulp_exec+'< Gulp.gin > Gulp.gout')
        
        os.chdir('..')
        self.gulp_results=read_gulp_results(self)
    
    def run_analyze_in_dir(self, use_subprocess = False, wait = True):
        print('Running Siesta-analyse in Directory:'+self.dir+ '!\n')
        n='RUN'       #self.sl
        if len(self.elecs)==0:
            print('why are you running analyze on something that has no electrodes attached?')
        
        command = self.siesta_exec+'-fdf TS.Analyze '+n+'.fdf '+self.print_no_print+' '+n+'_Analyze.out' 
        
        if use_subprocess:
            if wait:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   ).wait()
                self.write_analysis_results()
    
            else:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   )
                print('Analysis results not written automatically!')
            return
        
        os.chdir(self.dir)
        os.system(command)
        os.chdir('..')
        self.write_analysis_results()
    
    def run_siesta_electrode_in_dir(self, use_subprocess = False, wait = True):
        """Like run_siesta_in_dir, but with the --electrode flag of siesta"""
        
        print('Running Siesta electrode calculation in Directory: '+self.dir+ '!\n')
        n='RUN'      #self.sl
        command = self.mpi+self.siesta_exec+'--electrode <'+n+'.fdf '+self.print_no_print+' '+n+'.out'
        
        if use_subprocess:
            if wait:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   ).wait()
    
            else:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   )
            
            return
        
        os.chdir(self.dir)
        os.system( command )
        os.chdir('..')
    
    def plot_calculated_bands(self):
        """Have not been checked in a while, use sisl.Bandstructure
           and load the Hamiltonian and do it with a python script instead. 
        """
        
        os.chdir(self.dir)
        os.system(self.bands_plot+' <'+self.sl+'.bands '+self.print_no_print+' gnubands.f')
        # PyGnuplot.c('plot "gnubands.f"')
        os.chdir('..')
    def fatband_fdf(self):
        """ Fatbands """
        self.write_more_fdf(['WFS.Write.For.Bands True\n',
                             'COOP.Write True\n'])
        
    
    def run_fatbands(self, proj):
        """ Fatbands """
        
        write_mpr(self, proj)
        os.chdir(self.dir)
        os.system('cp '  + self.sl+'.bands.WFSX ' + self.sl+'.WFSX')
        os.system('fat ' + self.sl )
        files = os.listdir()
        for f in files:
            if '.EIGFAT' in f:
                name = f[:-7]
                os.system('eigfat2plot ' + f +' > ' + name + '.dat')
        
        os.chdir('..')
    
    def custom_bandlines(self, path):
        self.delete_fdf('KP')
        self.write_more_fdf(['%block kgrid.MonkhorstPack\n',
                              str(self.kp[0])+'  0  0  0.0\n',
                              '0  '+str(self.kp[1])+'  0  0.0\n',
                              '0  0  '+str(self.kp[2])+'  0.0\n',
                              '%endblock kgrid.MonkhorstPack\n'], name = 'KP')
        self.write_more_fdf(['BandLinesScale   ReciprocalLatticeVectors'],
                            name = 'KP')
        self.write_more_fdf(['%block BandLines'],
                            name = 'KP')
        self.write_more_fdf(path, 
                            name = 'KP')
        self.write_more_fdf(['%endblock BandLines'],
                            name = 'KP')
        
    
    def run_tbtrans_in_dir(self,DOS_GF= False, DOS_A=False, DOS_A_ALL=False,ORBITAL_CURRENT=False, Custom = [],
                           use_subprocess = False, wait = True, eta_elecs = None):
        """ Run TBtrans calculation in folder directory_name. """
        print('Running TB-Trans in Directory: '+self.dir+ '!\n')
        self.write_tb_trans_kp()
        n = 'RUN'         #self.sl
        command = self.mpi+self.tb_trans_exec + n+'_TBT.fdf '+self.print_no_print+' '+n+'TBTCalc.out'
        
        List = ['\n']
        if DOS_A:
            List+=['TBT.DOS.A T\n']
        if DOS_GF:
            List+=['TBT.DOS.Gf T\n']
        if DOS_GF:
            List+=['TBT.DOS.A.All T\n']
        if ORBITAL_CURRENT:
            List+=['TBT.Current.Orb T\n',
                   'TBT.Symmetry.TimeReversal False\n']
        if len(Custom)!=0:
            for C in Custom:
                List+=[C]
        if eta_elecs is not None:
            List+=['TBT.Elecs.Eta '+str(eta_elecs)+'\n']
        self.write_more_fdf(List,name='TS_TBT')
        
        if use_subprocess:
            if wait:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   ).wait()
    
            else:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   )
            
            return
        
        os.chdir(self.dir)
        os.system(command)
        os.chdir('..')
    
    def get_potential_energy(self):
        """Read the energy of the self consistent calculation """
        return read_total_energy(self)
    def get_fermi_level(self):
        """ Simple read function for the Fermi level"""
        return read_fermi_level(self)
    
    def relaxed_system_energy(self):
        import subprocess
        command = 'grep \'siesta: E_KS(eV)\' ' +self.dir+'/RUN.out'
        res = subprocess.check_output(command, shell=True)
        return float(res[-18:-2])
    
    def dQ(self):
        """Read excess charge from transiesta calculation."""
        
        dQ=None
        with open(self.dir+'/'+'RUN.out','r') as f:
            for l in f:
                if 'Excess charge' in l:
                    dQ=float(l[37:])
                    break
        self.dQ=dQ
    
    def create_added_stuff_dic(self):
        if hasattr(self,'Added_stuff') == False:
            self.Added_stuff = []
    
    def add_stuff(self, dic):
        self.create_added_stuff_dic()
        self.Added_stuff+= [dic]
    
    def Add_Bounded_Plane(self, O, A, B,  Potential, cut_off, spread = 100.0):
        """
            Read Geometry.Hartree siesta documentation.
            O, A, B: (3,) shaped arrays
            Potential: float, 
            cutoff:    float
            spread:    float
            
            Do this function before writting the fdf files with
            self.fdf
            
        """
        
        self.write_more_fdf(['%block Geometry.Hartree\n',
                            ' Bounded plane ' + str(Potential)  + ' eV \n',
                            '  gauss '+str(spread) + ' ' +    str(cut_off  )  + ' Ang\n',
                            '  ' + str(O[0]) + ' ' + str(O[1]) + ' ' + str(O[2]) + ' Ang\n',
                            '  ' + str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]) + ' Ang\n',
                            '  ' + str(B[0]) + ' ' + str(B[1]) + ' ' + str(B[2]) + ' Ang\n',
                            '%endblock Geometry.Hartree\n  ' 
                            ],
                            name = 'STRUCT')
        dic = {'what': 'bounded plane',
               'Potential':Potential,
               'vectors': [O,A,B],
               'spread': spread,
               'cut_off': cut_off}
        
        self.add_stuff(dic)
    
    def Add_Charged_Bounded_Plane(self, O, A, B,  Charge, cut_off, spread, kind = 'gauss'):
        """
            Read Geometry.Charge siesta documentation.
            Like self.Add_Bounded_Plane, but Potential -> Charge
            
            O, A, B: (3,) shaped arrays
            Charge: float, 
            cutoff:    float
            spread:    float
            
            Do this function before writting the fdf files with
            self.fdf
            
        """
        
        string = ['%block Geometry.Charge\n',
                  ' Bounded plane ' + str(Charge)  + ' \n',
                  '  '+ kind +' ' + str(spread) + ' ' +    str(cut_off)  + ' Ang\n',
                  '  ' + str(O[0]) + ' ' + str(O[1]) + ' ' + str(O[2]) + ' Ang\n',
                  '  ' + str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]) + ' Ang\n',
                  '  ' + str(B[0]) + ' ' + str(B[1]) + ' ' + str(B[2]) + ' Ang\n',
                  '%endblock Geometry.Charge\n  ' 
                  ]
        
        self.write_more_fdf(string,
                            name = 'STRUCT')
        
        dic = {'what': 'charged bounded plane',
               'charge':  Charge,
               'vectors': [O,A,B],
               'spread':  spread,
               'cut_off': cut_off,
               'kind':    kind}
        self.add_stuff(dic)
        
    def Add_Charged_Box(self,O, A, B, C, Charge, Ret = False):
        """
            Read Geometry.Charge siesta documentation.
            Adds a box of charge
            
            O, A, B, C: (3,) shaped arrays
            Charge:    float, 
            cutoff:    float
            spread:    float
            
            Do this function before writting the fdf files with
            self.fdf
            
        """
        
        
        if O.shape==(3,):
            string = ['%block Geometry.Charge\n',
                      ' box ' +  str(Charge) + ' \n',
                      '    delta\n'
                      '  ' + str(O[0]) + ' ' + str(O[1]) + ' ' + str(O[2]) + ' Ang\n',
                      '  ' + str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]) + ' Ang\n',
                      '  ' + str(B[0]) + ' ' + str(B[1]) + ' ' + str(B[2]) + ' Ang\n',
                      '  ' + str(C[0]) + ' ' + str(C[1]) + ' ' + str(C[2]) + ' Ang\n',
                      '%endblock Geometry.Charge\n  ' 
                      ]
            
            self.write_more_fdf(string,
                                name = 'STRUCT')
            write_this = string
        elif len(O.shape)==2:
            write_this = ['%block Geometry.Charge\n']
            for i,_ in enumerate(Charge):
                oo = O[i]; aa = A[i]; bb = B[i]; cc = C[i]; charge = Charge[i]
                write_this += [' box ' +  str(charge) + ' \n',
                               '    delta\n'
                               '  ' + str(oo[0]) + ' ' + str(oo[1]) + ' ' + str(oo[2]) + ' Ang\n',
                               '  ' + str(aa[0]) + ' ' + str(aa[1]) + ' ' + str(aa[2]) + ' Ang\n',
                               '  ' + str(bb[0]) + ' ' + str(bb[1]) + ' ' + str(bb[2]) + ' Ang\n',
                               '  ' + str(cc[0]) + ' ' + str(cc[1]) + ' ' + str(cc[2]) + ' Ang\n',
                              ]
            
            write_this += ['%endblock Geometry.Charge\n']
            self.write_more_fdf(write_this, name = 'STRUCT')
        
        dic = {'what': 'charged box',
               'charge':   Charge,
               'vectors': [O,A,B,C]}
        self.add_stuff(dic)
        if Ret:
            return write_this
        
        
    def Add_Charged_Sphere(self, Cent,  R, Ch, cut_off = 5.0):
        """
            Read Geometry.Charge siesta documentation.
            Like self.Add_Bounded_Plane, but Potential -> Charge
            
            Cent: (3,)  or (N,3) shaped array
            Ch:   float or (N, ) array 
            R:    float or (N, ) array
            cut_off: float
            
            Do this function before writting the fdf files with
            self.fdf
            
        """
        
        if Cent.shape==(3,):
            self.write_more_fdf(['%block Geometry.Charge\n',
                                 ' coords ' +  str(Ch) + ' \n',
                                 '    gauss '+str(cut_off)+'  '+str(R)+' Ang\n',
                                 '       1 spheres\n',
                                 '       ' + str(Cent[0]) + ' ' + str(Cent[1]) + ' ' + str(Cent[2]) + ' Ang\n',
                                 '%endblock Geometry.Charge\n  ' 
                                 ],
                                name = 'STRUCT')
        else:
            write_this = ['%block Geometry.Charge\n']
            for i,_ in enumerate(Cent):
                cent = Cent[i]; r = R[i]; ch = Ch[i]; 
                write_this+=[' coords ' +  str(ch) + ' \n',
                             '    gauss '+str(cut_off)+'  '+str(r)+' Ang\n'
                             '       1 spheres\n',
                             '       ' + str(cent[0]) + ' ' + str(cent[1]) + ' ' + str(cent[2]) + ' Ang\n',
                             ]
            write_this+=['%endblock Geometry.Charge']
            self.write_more_fdf(write_this)
            
        
        dic = {'what': 'charged sphere',
               'charge':   Ch,
               'cut_off': cut_off,
               'vectors': [Cent, R]}
        self.add_stuff(dic)
    
    def Real_space_SE(self, ax_decimation, ax_integrate,  
                      supercell, eta,Emin, Emax, dE, 
                      Contour = None, ending = 'TSHS',
                      only_couplings = False,
                      dk = 1000.0, 
                      mu = None,
                      parallel_E = False,
                      num_procs  = 4,
                      write_to_tbtrans = True,
                      dummy_calc = False
                      ):
        
        """
            Straight out of sisl tutorial TB8
            
            see sisl documentation RealSpaceSE for more
            info.
            You should probably consult a tutorial to know how to use
            the this function.
            self.elec_RSSE should be True
            
            The self instance can be two things depending on if self.elec_2E_RSSE
            is True or False. If it is False, the electrode is the small unitcell,
            the Hamiltonian of which gets tiled, and from which the greens function
            is calculated and integrated. If True however, the self is considered
            a device region and should have two electrodes. 
            
        """
        
        if self.elec_RSSE == False:
            print('set elec_RSSE to True!')
            assert 1 == 0
        import tqdm
        
        if hasattr(self,'RSSE_dict'):
            ax_decimation= self.RSSE_dict['ax_decimation']
            ax_integrate = self.RSSE_dict['ax_integrate']
            supercell    = self.RSSE_dict['supercell']
            eta          = self.RSSE_dict['eta']
            Contour      = self.RSSE_dict['Contour']
        
        if Contour is None:
            E = np.arange(Emin, Emax + dE / 2, dE)
        else:
            E = Contour.copy()
        
        H_minimal = sisl.get_sile(self.dir + '/' + self.sl + '.'+ending).read_hamiltonian()
        os.system('mv '+self.dir+'/'+self.sl+'.'+ending+' '+self.dir+'/'+self.sl+'_minimal_old.'+ending)
        if self.elec_2E_RSSE==False:
            RSSE      = sisl.RealSpaceSE(H_minimal, ax_decimation, ax_integrate, supercell, dk = dk)
            H_elec, elec_indices = RSSE.real_space_coupling(ret_indices=True)
            H_elec.write(self.dir + '/'+ self.sl + '.'+ending)
            H = RSSE.real_space_parent()
            indices = np.arange(len(H))
            indices = np.delete(indices, elec_indices)
            indices = np.concatenate([elec_indices, indices])
            np.save(self.dir + '/RS_Coupling_pos', H.xyz[elec_indices])
            np.save(self.dir + '/RS_Coupling_specie', H.atoms.Z[elec_indices])
            if only_couplings:
                return
            gamma = sisl.MonkhorstPack(H_elec, [1] * 3)
        np.save(self.dir+'/elec_RSSE_params.npz', {'ax_integrate': ax_integrate,
                                                   'unfold': supercell,
                                                   'dk': dk})
        
        sisl.io.tableSile(self.dir + '/contour.E', 'w').write_data(E)
        if parallel_E and self.elec_2E_RSSE == False:
            import joblib as Jl
            if write_to_tbtrans:
                _bulk = True
                _coupling = True
            else:
                _bulk = False
                _coupling = True
            global SE_func # for multiprocessing backend
            def SE_func(e):
                return e, RSSE.self_energy(e, bulk=_bulk, coupling=_coupling)
            
            results = Jl.Parallel(n_jobs=num_procs,
                                  backend = joblib_backend,
                                  verbose = 10)(Jl.delayed(SE_func)(e + 1j*eta) for e in E)
            del SE_func
            ResDic = {}
            for r in results:
                ResDic.update({r[0]:r[1]})
        
        if parallel_E and self.elec_2E_RSSE:
            self.calculate_2E_RSSE(Contour, tdir=ax_decimation, 
                                   kdir = ax_integrate, n_jobs = num_procs, 
                                   ty   = supercell[ax_integrate], dummy_calc = dummy_calc)
            f = np.load(self.dir+'/RSSE.npz')
            _coupling = True
            if write_to_tbtrans:
                _bulk     = True
            else:
                _bulk     = False
            ResDic = {}
            nE   = len(f['Ev'])
            Ev   = f['Ev']
            cidx = f['coupling_idx']
            SER  = f['RealspaceSE']
            HR   = f['HR'][cidx[:, None], cidx[None,:]]
            SR   = f['SR'][cidx[:, None], cidx[None,:]]
            
            np.save(self.dir+'/RS_Coupling_pos', f['xyz'])
            np.save(self.dir+'/RS_Coupling_specie', f['Z'])
            if only_couplings:
                return cidx
            for _i in range(nE):
                if _bulk == False:
                    sei = SER[_i]
                else:
                    sei = SR*Ev[_i] - HR - SER[_i]
                ResDic.update({Ev[_i]: sei.copy()})
            gamma = sisl.MonkhorstPack(sisl.get_sile(self.dir+'/2E_RSSE.TSHS').read_hamiltonian(), [1]*3)
        
        with sisl.io.tbtgfSileTBtrans(self.dir +'/' + self.sl +'.TBTGF',version="new") as f:
            if mu is not None:f.write_header(gamma, E + 1j * eta, mu = mu)
            else:             f.write_header(gamma, E + 1j * eta)
            
            for ispin, new_k, k, e in tqdm.tqdm(f):
                if new_k:
                    if self.elec_2E_RSSE==False:
                        f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128),
                                            H_elec.Sk(format='array', dtype=np.complex128))
                    else:
                        f.write_hamiltonian(HR.astype(np.complex128), 
                                            SR.astype(np.complex128))
                if parallel_E: SeHSE = ResDic[e+1j*eta]
                else:          SeHSE = RSSE.self_energy(e + 1j*eta, bulk=True, coupling=True)
                f.write_self_energy(SeHSE)
        
        self.RSSE_Energy_from_to = (Emin, Emax)
        self._supercell = supercell
        self._dk  = dk
        self._ax_integrate  = ax_integrate
        self._ax_decimation = ax_decimation
        if write_to_tbtrans==False and parallel_E:
            return ResDic
        
    def Real_space_SI(self, ax_integrate, supercell, eta, Contour,
                      nsc = (1,3,1), 
                      ending = 'TSHS',
                      Hsurf_in = None,
                      only_couplings = False,
                      dk = 1000.0,
                      mu = None,
                      parallel_E = False,
                      num_procs  = 4,
                      sisl_patch_1 = False,
                      keep_unpicklable_objects = False,
                      ):
        """ Similar to Real_space_SE, refer to tutorials for how to use this.
        """
        import tqdm
        from sisl.physics import RecursiveSI, RealSpaceSI
        if self.elec_RSSE == False:
            print('set elec_RSSE to True!')
            error()
        if self.elec_SurRSSE == False:
            print('set elec_SurRSSE to True!')
            error()
        if self.elec_3D_RSSE:
             self.Real_space_SI_3D()
             return
        if hasattr(self,'SurRSSE_dict'):
            ax_integrate = self.SurRSSE_dict['ax_integrate']
            supercell    = self.SurRSSE_dict['supercell']
            eta          = self.SurRSSE_dict['eta']
            try:
                Contour  = self.SurRSSE_dict['Contour']
            except:
                pass
            nsc          = self.SurRSSE_dict['nsc']
            Hsurf_in     = self.SurRSSE_dict['Hsurf_in']
        E = Contour
        
        def translate(s):
            return s.replace('a1','A').replace('a2','B').replace('a3','C')
        H_minimal = sisl.get_sile(self.dir + '/' + self.sl + '.'+ending).read_hamiltonian()
        SE        = RecursiveSI(H_minimal, translate(self.semi_inf))
        if Hsurf_in is None:
            Hsurf = H_minimal.copy()
        else:
            Hsurf = Hsurf_in.copy()
        Hsurf.set_nsc(nsc)
        
        if isinstance(ax_integrate, int):
            if not nsc[ax_integrate]>1:
                print('Wrong nsc or ax_integrate passed!')
                error()
        if isinstance(ax_integrate, list):
            if not (nsc[ax_integrate]>1).any():
                print('Wrong nsc or ax_integrate passed!')
                error()
        
        SRSSE = RealSpaceSI(SE, Hsurf, 
                            ax_integrate, 
                            unfold = supercell,
                            dk = dk)
        if sisl_patch_1:
            print('---Warning---')
            print('Youre now using a custom real_space_coupling method in the RealSpaceSI class')
            
            from siesta_python.sisl_patches import Alt_real_space_coupling
            SRSSE.real_space_coupling = Alt_real_space_coupling.__get__(SRSSE, RealSpaceSI)
            SRSSE.initialize()
        if keep_unpicklable_objects:
            self._RSSI_SE = SE
            self._RSSI_SRSSE = SRSSE
        
        H_elec, elec_indices = SRSSE.real_space_coupling(ret_indices=True)
        # Before we overwrite the electrode Hamiltonian for TBTrans we move the 
        # minimal electrode Hamiltonian
        _ori_name = self.dir + '/' + self.sl + '.'+ending
        _new_name = self.dir + '/' + self.sl + '_minimal_old.'+ending
        os.system('cp '+_ori_name + ' ' + _new_name)
        Hsurf.write(self.dir + '/' + self.sl + 'Hsurf_RSSI.TSHS')
        np.save(self.dir+'/elec_RSSI_params.npz', {'direc': translate(self.semi_inf),
                                                   'ax_integrate': ax_integrate,
                                                   'unfold': supercell,
                                                   'dk': dk})
        H_elec.write(self.dir + '/'+ self.sl + '.'+ending)
        H = SRSSE.real_space_parent()
        
        np.save(self.dir + '/RS_Coupling_pos', H.xyz[elec_indices])
        np.save(self.dir + '/RS_Coupling_specie', H.atoms.Z[elec_indices])
        if only_couplings:
            return
        
        gamma = sisl.MonkhorstPack(H_elec, [1] * 3)
        sisl.io.tableSile(self.dir + '/contour.E', 'w').write_data(E, np.zeros(E.size) + 0., fmt='.12e')
        
        
        
        
        if parallel_E:
            import joblib as Jl
            global SE_func # for multiprocessing backend
            def SE_func(e):
                return e, SRSSE.self_energy(e, bulk=True, coupling=True)
            results = Jl.Parallel(n_jobs  = num_procs,
                                  backend = joblib_backend,
                                  verbose = 10)(Jl.delayed(SE_func)(e + 1j*eta) for e in E)
            del SE_func
            ResDic = {}
            for r in results:
                ResDic.update({r[0]:r[1]})
        
        with sisl.io.tbtgfSileTBtrans(self.dir +'/' + self.sl +'.TBTGF',version="new") as f:
            if mu is not None:f.write_header(gamma, E + 1j * eta, mu = mu)
            else:             f.write_header(gamma, E + 1j * eta)
            
            for ispin, new_k, k, e in tqdm.tqdm(f):
                
                if new_k:
                    f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128),
                                        H_elec.Sk(format='array', dtype=np.complex128))
                if parallel_E: SeHSE = ResDic[e+1j*eta]
                else:          SeHSE = SRSSE.self_energy(e + 1j*eta, bulk=True, coupling=True)
                f.write_self_energy(SeHSE)
        
        self.RSSE_Energy_from_to = (E.min(), E.max())
        self._supercell = supercell
        self._dk  = dk
        self._ax_integrate  = ax_integrate
    
    def Real_space_SI_3D(self):
        import tqdm
        assert hasattr(self, 'SurRSSE_dict')
        is_pol    = self.spin_pol in ['polarized', 'polarised']
        opts      = self.SurRSSE_dict
        C         = opts['Contour']
        supercell = opts['supercell']
        new_opts  = opts.copy()
        new_opts.pop('mu')
        tx, ty = supercell[0], supercell[1]
        mu     = opts['mu']
        periodic = False
        if len(self.semi_inf) != 3:
            periodic = True
            assert "nk_grid" in new_opts.keys()
            new_opts.update({'do_periodic':True})
            _tmp_g = sisl.MonkhorstPack(sisl.geom.sc(1.0, 1), new_opts['nk_grid'])
            if self.semi_inf =='ac':
                _ks = _tmp_g.k[:,1].copy()
                kxy = (None, _ks )
                assert new_opts['nk_grid'][1]>1
            elif self.semi_inf =='bc':
                _ks = _tmp_g.k[:,0].copy()
                kxy = (_tmp_g.k[:,0].copy(), None )
                assert new_opts['nk_grid'][0]>1
            assert new_opts['nk_grid'][2] == 1
            new_opts.update({'kxy':kxy})
        
        sisl.io.tableSile(self.dir + '/contour.E', 'w').write_data(C, np.zeros(C.size) + 0., fmt='.12e')
        if is_pol == False:
            H_elec = self.calculate_bulksurf_RSSE(**new_opts)
            f = np.load(self.dir+'/3D_bulk_SE.npz')
            gamma = sisl.MonkhorstPack(H_elec, [1] * 3)
            np.save(self.dir + '/RS_Coupling_pos.npy',    f['xyz_elec'])
            np.save(self.dir + '/RS_Coupling_specie.npy', f['s_elec'])
            _ori_name = self.dir + '/' + self.sl +'.TSHS'
            _new_name = self.dir + '/' + self.sl + '_minimal_old.TSHS'
            os.system('cp '+_ori_name + ' ' + _new_name)
            H_elec.write(self.dir+'/'+self.sl+'.TSHS')
            print(H_elec.shape)
            if periodic == False:
                tmp_files = os.listdir('_tmp_se')
                tmp_files.remove('coups_a.npy')
                tmp_files.remove('coups.npy')
                tmp_files.remove('Htot.TSHS')
                tmp_val = np.array([complex(n.replace('.npz','')) for n in tmp_files])
            else:
                gamma = sisl.MonkhorstPack(H_elec, new_opts['nk_grid'])
                flds = [f for f in  os.listdir() if '_tmp_se' in f]
                idx  = np.argsort(np.array([float(f.split('_')[4]) for f in flds]))
                flds = [flds[i] for i in idx]
                tmp_files_k = []
                tmp_val_k   = []
                for fld in flds:
                    _tmp_files = os.listdir(fld)
                    _tmp_files.remove('coups_a.npy')
                    _tmp_files.remove('coups.npy')
                    _tmp_files.remove('Htot.TSHS')
                    _tmp_val     = np.array([complex(n.replace('.npz','')) for n in _tmp_files])
                    tmp_val_k   += [_tmp_val.copy()]
                    tmp_files_k += [_tmp_files]
            
            with sisl.io.tbtgfSileTBtrans(self.dir +'/' + self.sl +'.TBTGF',
                                          version="new") as f:
                if mu is not None:f.write_header(gamma, C, mu = mu)
                else:             f.write_header(gamma, C)
                for ispin, new_k, k, e in tqdm.tqdm(f):
                    try:
                        if periodic == False:
                            fnpz  = np.load('_tmp_se/'+str(e)+'.npz')
                        else:
                            ## PERIODIC CASE
                            kidx  = f.kindex(k)
                            fnpz  = np.load(flds[kidx]+'/'+str(e)+'.npz')
                    except:
                        if periodic == False:
                            dist = np.abs(e - tmp_val)
                            if dist.min()<1e-13:
                                fname_close = tmp_files[np.where(dist == dist.min())[0][0]]
                            else:
                                assert 1 == 0
                            fnpz = np.load('_tmp_se/'+fname_close)
                        else:
                            ## PERIODIC CASE
                            kidx  = f.kindex(k)
                            dist  = np.abs(e - tmp_val_k[kidx])
                            if dist.min()<1e-13:
                                fname_close = tmp_files[np.where(dist == dist.min())[0][0]]
                            else:
                                assert 1 == 0
                            print('Filepath loaded: ' + tmp_files_k[kidx]+'/'+fname_close)
                            fnpz = np.load(tmp_files_k[kidx]+'/'+fname_close)
                    if periodic == False:
                        if new_k:
                            f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128),
                                                H_elec.Sk(format='array', dtype=np.complex128))
                    else:
                        ## PERIODIC CASE
                        if new_k:
                            print('input to H_elec: ', k)
                            f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128, k=k),
                                                H_elec.Sk(format='array', dtype=np.complex128, k=k))
                    
                    SeHSE = fnpz['se_ele']
                    f.write_self_energy(SeHSE)
            os.system('rm -rf _tmp_se*')
        else:
            H_elec1, H_elec2 = self.calculate_bulksurf_RSSE(**new_opts)
            H_elec = sisl.Hamiltonian.fromsp(H_elec1.geometry, 
                                             [H_elec1.tocsr(),
                                              H_elec2.tocsr()],
                                             H_elec1.tocsr(dim=1),
                                             spin='pol')
            
            f1 = np.load(self.dir+'/3D_bulk_SE_up.npz')
            # f2 = np.load(self.dir+'/3D_bulk_SE_dn.npz')
            gamma = sisl.MonkhorstPack(H_elec, [1] * 3)
            np.save(self.dir + '/RS_Coupling_pos.npy',    f1['xyz_elec1'])
            np.save(self.dir + '/RS_Coupling_specie.npy', f1['s_elec1'])
            _ori_name = self.dir + '/' + self.sl +'.TSHS'
            _new_name = self.dir + '/' + self.sl + '_minimal_old.TSHS'
            os.system('cp '+_ori_name + ' ' + _new_name)
            H_elec.write(self.dir+'/'+self.sl+'.TSHS')
            print(H_elec.shape)
            if periodic == False:
                tmp_files1 = os.listdir('_tmp_se_up')
                tmp_files1.remove('coups_a.npy')
                tmp_files1.remove('coups.npy')
                tmp_files1.remove('Htot.TSHS')
                tmp_val1   = np.array([complex(n.replace('.npz','')) for n in tmp_files1])
                tmp_files2 = os.listdir('_tmp_se_dn')
                tmp_files2.remove('coups_a.npy')
                tmp_files2.remove('coups.npy')
                tmp_files2.remove('Htot.TSHS')
                tmp_val2   = np.array([complex(n.replace('.npz','')) for n in tmp_files2])
            else:
                gamma = sisl.MonkhorstPack(H_elec, new_opts['nk_grid'])
                flds1 = [f for f in  os.listdir() if '_tmp_se_up' in f]
                flds2 = [f for f in  os.listdir() if '_tmp_se_dn' in f]
                idx1  = np.argsort(np.array([float(f.split('_')[5]) for f in flds1]))
                idx2  = np.argsort(np.array([float(f.split('_')[5]) for f in flds2]))
                flds1 = [flds1[i] for i in idx1]
                flds2 = [flds2[i] for i in idx2]
                
                tmp_files_k1 = []
                tmp_val_k1   = []
                tmp_files_k2 = []
                tmp_val_k2   = []
                for fld in flds1:
                    _tmp_files = os.listdir(fld)
                    _tmp_files.remove('coups_a.npy')
                    _tmp_files.remove('coups.npy')
                    _tmp_files.remove('Htot.TSHS')
                    _tmp_val     = np.array([complex(n.replace('.npz','')) for n in _tmp_files])
                    tmp_val_k1   += [_tmp_val]
                    tmp_files_k1 += [_tmp_files]
                for fld in flds2:
                    _tmp_files = os.listdir(fld)
                    _tmp_files.remove('coups_a.npy')
                    _tmp_files.remove('coups.npy')
                    _tmp_files.remove('Htot.TSHS')
                    _tmp_val     = np.array([complex(n.replace('.npz','')) for n in _tmp_files])
                    tmp_val_k2   += [_tmp_val]
                    tmp_files_k2 += [_tmp_files]
                stmp_files = [tmp_files_k1, tmp_files_k2]
                stmp_val   = [tmp_val_k1  , tmp_val_k2  ]
            with sisl.io.tbtgfSileTBtrans(self.dir +'/' + self.sl +'.TBTGF',version="new") as f:
                if mu is not None:f.write_header(gamma, C, mu = mu)
                else:             f.write_header(gamma, C)
                for ispin, new_k, k, e in tqdm.tqdm(f):
                    kidx = f.kindex(k)
                    if ispin == 0:
                        if periodic == False:
                            prefix = '_tmp_se_up'
                            tmp_val=tmp_val1
                            tmp_files=tmp_files1
                        else:
                            prefix    = flds1[kidx]
                            tmp_val   = stmp_val[0][kidx]
                            tmp_files = stmp_files[0][kidx]
                    else:
                        if periodic == False:
                            prefix = '_tmp_se_dn'
                            tmp_val=tmp_val2
                            tmp_files=tmp_files2
                        else:
                            prefix    = flds2[kidx]
                            tmp_val   = stmp_val[1][kidx]
                            tmp_files = stmp_files[1][kidx]
                    try:
                        fnpz  = np.load(prefix+'/'+str(e)+'.npz')
                    except:
                        dist = np.abs(e - tmp_val)
                        if dist.min()<1e-13:
                            fname_close = tmp_files[np.where(dist == dist.min())[0][0]]
                        else:
                            assert 1 == 0
                        fnpz = np.load(prefix+'/'+fname_close)
                
                    if periodic == False:
                        if new_k:
                            f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128, spin=ispin),
                                                H_elec.Sk(format='array', dtype=np.complex128))
                    else:
                        if new_k:
                            f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128,  spin=ispin,k=k),
                                                H_elec.Sk(format='array', dtype=np.complex128,  k=k))
                    SeHSE = fnpz['se_ele']
                    f.write_self_energy(SeHSE)
            os.system('rm -rf _tmp_se_up*')
            os.system('rm -rf _tmp_se_dn*')
        self.RSSE_Energy_from_to = (C.real.min(), C.real.max())
    
    def undo_RSSE(self, ending='TSHS'):
        _ori_name = self.dir + '/' + self.sl + '.'+ending
        _new_name = self.dir + '/' + self.sl + '_minimal_old.'+ending
        os.system('rm '+_ori_name)
        os.system('cp '+_new_name + ' ' + _ori_name)
    
    def get_RS_pos(self):
        """Retrieve positions and species of the atoms in the real space
        calculation."""
        
        try:
            p_rs = np.load(self.dir + '/RS_Coupling_pos.npy')
            s_rs = np.load(self.dir + '/RS_Coupling_specie.npy')
            return p_rs, s_rs
        except: 
            return None, None
    
    def reset_minimal_hamiltonian(self, ending = 'TSHS',k = ''):
        _new_name = self.dir + '/' + self.sl + '_minimal_old.'+ending
        old_name  = self.dir + '/' + self.sl + '.'+ending
        os.system('mv '+k+_new_name + ' ' + old_name)
    
    def copy_DM_from(self, object, ftype='TSDE'):
        """
            Copy .DM or TSDE file (.ftype) from the place specified by 'object'
            object is either a string with a path or a object with 
            self.dir attribute (e.g. a siesta_python instance)
        """
        ftype_ ='.'+ftype+' '
        if isinstance(object, str):
            os.system('cp '+object+'/*'+ftype_+self.dir+'/'+self.sl+ftype_)
        if isinstance(object, SiP):
            os.system('cp '+object.dir+'/'+object.sl+ftype_+self.dir+'/'+self.sl+ftype_)
        self.reuse_dm = True
    
    #
    #def copy_siesta_DM_from(self, object):
    #    if isinstance(object, str):
    #        os.system('cp ' + object + '/*.DM ' + self.dir+'/'+self.sl+'.DM')
    #    if isinstance(object, SiP):
    #        os.system('cp ' + object.dir + '/' + object.sl+'.DM ' + self.dir + '/' + self.sl + '.TSDE')
    #    self.reuse_dm = True
    
    def copy_TSHS_from(self, object):
        os.system('cp ' + object.dir + '/' + object.sl+'.TSHS ' + self.dir + '/' + self.sl + '.TSHS')
    
    
    def save_file(self, file, folder, newname):
        try:
            os.mkdir(folder)
        except:
            pass
        os.system('cp ' + object.dir + '/' + file  + ' ' +  folder + '/' + newname )
    
    def mulliken(self):
        idx, C = Mulliken(self.dir + '/RUN.out')
        self.Mulliken_idx = idx
        self.Mulliken_C   = C 
    
    def write_tb_trans_kp(self):
        if self.kp_tbtrans == None and hasattr(self, 'manual_tbtrans_kpoint') == False:
            print('no k-poins for tbtrans!')
            pass
        else:
            name='RUN'   ##self.sl
            with open(self.dir+'/'+name+'.fdf','r') as f:
                lines=[i for i in f]
                f.close()
            with open(self.dir+'/'+name+'_TBT.fdf','w') as f:
                for l in lines:
                    if '%include KP.fdf' in l:
                        f.write('%include KP_TBT.fdf\n')
                    elif 'DEFAULT' in l:
                        pass
                    
                    else: f.write(l)
                f.close()
            
    
    def gen_basis(self):
        os.chdir(self.dir)
        files_before = ld()
        os.system(self.gen_basis_exec + '<RUN.fdf> out')
        for f in ld():
            if f in files_before:
                pass
            else:
                os.system('cp '+f +' orbital_directory/'+f)
                if '.py' in f or f in no_rem:
                    pass
                else:
                    print(f)
                    rem(f)
        os.chdir('..')
    
    def Check_Distances(self,min_dist=0.3):
        T=[]
        for i,pi in enumerate(self.pos_real_space):
            d = ((pi[0]-self.pos_real_space[:,0])**2 + 
                 (pi[1]-self.pos_real_space[:,1])**2 + 
                 (pi[2]-self.pos_real_space[:,2])**2)**0.5
            if ( ( ( self.s >= 0 ) * (d<min_dist) ).sum() )>1:
                print('atoms too close...')
                error()
        #print('No atoms overlapping within ' +str(min_dist) +' Å!')
    
    def delete_fdf(self,name):
        os.chdir(self.dir)
        os.system('rm ' + name  + '.fdf')
        os.chdir('..')
    
    def manual_k_points(self,reduced_k_arr,weights, tbtrans = False):
        k = reduced_k_arr.copy()
        assert len(k) == len(weights)
        L = ['kgrid.File Manual_k.fdf']
        end = '_TBT' if tbtrans else ''
        self.delete_fdf('KP'+end)
        self.write_more_fdf(L, name = 'KP'+end)
        with open(self.dir + '/' + 'Manual_k.fdf', 'w') as f:
            nk = len(k)
            f.write(str(nk) + '\n')
            for i,ki in enumerate(k):
                f.write('  ' + str(i) + '  ' + str(ki[0]) + '  ' + str(ki[1]) + '  '+str(ki[2]) + '  ' + str(weights[i]) + '\n')
    
    def make_meshed_k_grid(self, kx, ky):
        from scipy.spatial import Delaunay
        from funcs import PolyArea
        KX, KY = np.meshgrid(kx, ky, indexing = 'ij')
        Grid = np.vstack((KX.ravel(), KY.ravel()))
        Grid = np.array(Grid)
        tri = Delaunay(Grid)
        p = tri.points
        v = tri.simplices
        Weights = []
        kgrid = []
        
        for t in v:
            center = np.average(p[t],axis = 0)
            area   = PolyArea(p[t,0], p[t,1])
            kgrid   += [center]
            Weights += [area]
        
        kgrid = np.array(kgrid)
        Weights = np.array(Weights)
        kgrid = np.hstack((kgrid, np.zeros((len(kgrid), 1))))
        
        self._manual_kgrid   = kgrid
        self._manual_weights = Weights
        self._manual_kgrid_tri = tri
        self.manual_k_points(kgrid, Weights)
    
    def make_concentrated_k_points(self, points, nk, r, n_points, close_tol = 1e-3):
        from scipy.spatial import Delaunay
        from funcs import PolyArea
        ls = np.linspace
        
        for p in points:
            p[0] = np.mod(p[0], 1)
            p[1] = np.mod(p[1], 1)
        
        Grid = []
        kx,ky = np.meshgrid(ls(0,1, nk[0]), ls(0,1, nk[1]), indexing = 'ij')
        
        
        for i in range(nk[0]):
            for ii in range(nk[1]):
                Grid += [[kx[i,ii], ky[i,ii]]]
        
        for c in points:
            rg  = ls(0,1, n_points+1)[1:] * r
            phi = ls(0,2* np.pi, n_points + 1) [:-1]
            R,P = np.meshgrid(rg,phi,indexing='ij')
            R   = R.ravel(); 
            P   = P.ravel()
            x = R * np.cos(P)
            y = R * np.sin(P)
            x += c[0]
            y += c[1]
            Grid += [[c[0], c[1]]]
            for ip in range(len(x)):
                Grid += [[np.mod(x[ip],1), np.mod(y[ip],1)]]
        
        Grid = np.array(Grid)
        tri = Delaunay(Grid)
        p = tri.points
        v = tri.simplices
        Weights = []
        kgrid = []
        
        for t in v:
            center = np.average(p[t],axis = 0)
            area   = PolyArea(p[t,0], p[t,1])
            kgrid   += [center]
            Weights += [area]
        
        kgrid = np.array(kgrid)
        Weights = np.array(Weights)
        kgrid = np.hstack((kgrid, np.zeros((len(kgrid), 1))))
        
        self._manual_kgrid   = kgrid
        self._manual_weights = Weights
        self._manual_kgrid_tri = tri
        self.manual_k_points(kgrid, Weights)
    
    
    def ideal_cell_charge(self,chargedic,spindeg=True):
        q = 0.0
        for s in self.s:
            q += chargedic[Num2Sym[s]]
        if spindeg:
            q *= 0.5
        return q
    
    def get_contour_from_failed_RSSE(self):
        from time import sleep
        sleep(0.25)
        return read_contour_from_failed_RSSE_calc(self.dir + '/RUN.out')
    
    def toASE(self):
        from ase import Atoms
        return Atoms(positions = self.pos_real_space, cell = self.lat, numbers = self.s)
    
    def fois_gras(self, H,ending = 'TSHS'):
        H.write(self.dir + '/' + self.sl + '.'+ending)
    # Wrapped fois gras for better name
    def manual_H(self, H,ending='TSHS'):
        self.fois_gras(H,ending=ending)
    
    def to_sisl(self, what = 'geom', R = 3.0):
        if what == 'geom':
            A = sisl.Geometry(xyz = self.pos_real_space, lattice = self.lat, atoms = self.s)
            for i in range(len(A._atoms)):
                A._atoms[i] = sisl.Atom(A._atoms[i].Z, R = R)
            
            return A 
        
        elif what == 'fromDFT':
            H = sisl.get_sile(self.dir + '/RUN.fdf').read_hamiltonian()
            S = sisl.get_sile(self.dir + '/RUN.fdf').read_overlap()
            
            return H, S
        
        elif what == 'sile':
            return sisl.get_sile(self.dir + '/RUN.fdf')
        elif what == 'TSHS':
            H = sisl.get_sile(self.dir + '/'+self.sl+'.TSHS').read_hamiltonian()
            S = sisl.get_sile(self.dir + '/'+self.sl+'.TSHS').read_overlap()
            return H, S
        if what == 'fdf':
            H = sisl.get_sile(self.dir+'/RUN.fdf').read_hamiltonian()
            S = sisl.get_sile(self.dir+'/RUN.fdf').read_overlap()
            return H,S
            
    
    def read_TSHS(self, front = None):
        if front is None:
            name = self.sl
        else:
            name = front
        if self.Hubbard_electrode == 'Yes':
            name = 'Hubbard_'+name
        #try:
        return sisl.get_sile(self.dir+'/'+name+'.TSHS').read_hamiltonian()
        #except:
        #    res = sisl.get_sile(self.dir+'/'+name+'.HSX').read_hamiltonian()
    
    def read_minimal_TSHS(self):
        """ Meant for reading the minimal Hamiltonian
            for redoing a real-space calculation
        """
        return sisl.get_sile(self.dir+'/'+self.sl+'_minimal_old.TSHS').read_hamiltonian()
    def read_rssi_surface_TSHS(self):
        """ Meant for reading the minimal Hamiltonian
            for redoing a real-space calculation
        """
        return sisl.get_sile(self.dir+'/'+self.sl+'Hsurf_RSSI.TSHS').read_hamiltonian()
    
    def to_TB_model(self, fH, fS):
        h = self.to_sisl(what = 'geom')
        s = self.to_sisl(what = 'geom')
        H = sisl.Hamiltonian(h)
        S = sisl.Overlap    (s)
        H.set_nsc((3,3,1))
        S.set_nsc((3,3,1))
        H.construct(fH)
        S.construct(fS)
        return {'H': H, 'S': S}
    
    def write_TB_model(self, fH, fS = None, Return = False, nsc = (3,3,1)):
        h = self.to_sisl(what = 'geom')
        l = []
        H = sisl.Hamiltonian(h)
        H.set_nsc(nsc)
        H.construct(fH)
        H.write(self.dir + '/' + self.sl + '.TSHS')
        if fS != None:
            S = sisl.Overlap(h)
            S.set_nsc((3,3,1))
            S.construct(fS)
        
        if Return  == True and fS == None:
            return H
        elif Return == True and fS is not None:
            return H, S
    
    def E_field(self,E):
        self.write_more_fdf(['%block ExternalElectricField\n',
                              str(E)[1:-1]+' V/Ang\n',
                             '%endblock ExternalElectricField\n'],
                              name = 'DEFAULT')
    
    def is_siesta_done(self):
        try:
            with open(self.dir + '/RUN.out', 'r') as f:
                lines = f.readlines()
                if 'Job completed' in lines[-1]:
                    return True
                return False
        except:
            return False
    
    def set(self, key, val):
        self.__dict__[key] = val
    
    def sleep(self, n):
        from time import sleep
        sleep(n)
    
    def scream(self, message):
        with open(self.dir + '/lua_comm/python_scream.txt', 'w') as f:
            for m in message:
                f.write(m)
        
        
        with open(self.dir + '/lua_comm/python_scream_history.txt', 'a') as f:
            f.write('\n --scream at ' + str(datetime.now())+'--\n')
            
            for m in message:
                f.write(m)
    
    def hear(self):
        with open(self.dir + '/lua_comm/siesta_scream.txt', 'w') as f:
            message = f.readlines()
        return message
    
    def set_methfessel_paxton(self, N):
        lines = ['OccupationFunction MP\n',
                 'OccupationMPOrder '+ str(N) + '\n']
        self.write_more_fdf(lines, name = 'DEFAULT')
    
    def get_labelled_indices(self, condition = None):
        idx  = []
        labels = []
        if condition == None:
            def condition(x):
                return True
        
        for i, r in enumerate(self.pos_real_space):
            if condition(r):
                idx+=[i]
                elec_it = 0
                Truth = [True if i in ei else False for ei in self.elec_inds] 
                if sum(Truth)>0:
                    labels += ['Electrode ' + str(Truth.index(True))]
                elif i in self.buffer_atoms:
                    labels += ['Buffer atom']
                else:
                    labels += ['Device atom']
        
        return idx, labels
    
    def Put_Variable_Charged_Sphere_On_All(self, f, R, offset = np.zeros(3), cut_off = 5.0):
        self.write_more_fdf(['%block Geometry.Charge\n'], name = 'STRUCT')
        for ri in self.pos_real_space:
            self.Add_Charged_Sphere(ri-offset, R, f(ri), Block= False, cut_off = cut_off)
        self.write_more_fdf(['%endblock Geometry.Charge\n'], name = 'STRUCT')
        
    def Dipole_Correction_Vacuum(self, Type,  direction, vacuum_point, Origin = None):
        self.write_more_fdf(['Slab.DipoleCorrection ' + Type + '\n'], name = 'DEFAULT')
        
        if Origin is not None:
            pos_str = str(Origin[0]) + ' ' + str(Origin[1]) + ' ' + str(Origin[2])
            self.write_more_fdf(['%block Slab.DipoleCorrection.Origin\n',
                                 '   ' + pos_str + ' Ang\n',
                                 '%endblock\n'])
        
        self.write_more_fdf([
                             '%block Slab.DipoleCorrection.Vacuum\n', 
                             '   direction  ' + str(direction)[1:-1] +    '\n',
                             '   position   ' + str(vacuum_point)[1:-1] + ' Ang\n',
                             '%endblock\n'
                             ]
                            )
        
        dic = {'what':  'dipole correction',
               'origin':       Origin,
               'direction':    direction,
               'vacuum point': vacuum_point}
        self.add_stuff(dic)
    
    def create_mixed_pseudo_dic(self):
        if hasattr(self,'mixed_pseudo_list') == False:
            self.mixed_pseudo_list = []
    
    def add_mixed_pseudo_dic(self, dic):
        self.create_mixed_pseudo_dic()
        self.mixed_pseudo_list+= [dic]
    
    def ps_mixer(self, which, weight, name, use_subprocess = False, wait = False):
        command = self.mix_ps  +   \
                   which[0] + ' ' + \
                   which[1] + ' ' + str(weight) + ' > mixpisout'
        if use_subprocess:
            if wait:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   ).wait()
    
            else:
                Popen('cd '+self.dir + ' && ' + 
                      command, shell = True   )
            
            return
        os.chdir(  self.dir )
        os.system( command  )
        os.chdir('..')
        # print(command)
        with open(self.dir + '/MIXLABEL') as f:
            for l in f:
                mixlabel = l
        command2 = 'cp ' + self.dir + '/' + mixlabel[:-1] + '.psf' + ' ' + self.dir + '/' +name + '.psf'
        os.system(command2)
        with open(self.dir + '/' + mixlabel[:-1] + '.synth') as f:
            lines = []
            for l in f:
                lines += [l]
        dic = [name , lines]
        self.add_mixed_pseudo_dic(dic)
    
    def dump_pseudo_list_to_struct(self):
        ua = unique_list(self.s)
        write_this = []
        it = 0
        write_this += ['%block SyntheticAtoms']
        for d in self.mixed_pseudo_list:
            i = 200+int(d[0].split(sep = '_')[-1])
            species_index = ua.index(i) + 1
            
            write_this+=['\n#### ' + d[0] + '\n']
            for n,l in enumerate(d[1]):
                if n==1:
                    write_this += ['  ' +str(species_index) + '\n']
                
                elif 0<n<4:
                    write_this+=[l]
            it+=1
        write_this += ['%endblock SyntheticAtoms']
        self.write_more_fdf(write_this, name = 'STRUCT')
    
    def set_synthetic_mixes(self, mix_name,frac, atoms):
        if not hasattr(self, '_mixing_dic'):
            self._mixing_dic              = {}
            print('Made dictionary for mixing info')
        self._mixing_dic.update({mix_name:[atoms, frac]})
        assert len(self._mixing_dic.keys())<20
    
    def nnr(self, n_nn = 3, r_cut = 2.5):
        # Slow version of the similar function found in GrainB2
        ps = np.zeros((0,3))
        n = len(self.pos_real_space)
        
        print('nnr')
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                ps = np.vstack((ps, self.pos_real_space + 
                                i * self.lat[0,:]       + 
                                j * self.lat[1,:]))
        
        dij = np.zeros((n, 9 * n))
        for i in range(n):
            dij[i,:] = np.linalg.norm(self.pos_real_space[i] - ps , axis = 1)
        
        dij[dij< 1e-10] = 10 ** 6
        nf  = len(ps)
        nnr = np.zeros((n, n_nn, 3))
        
        for i in range(n):
            ri = self.pos_real_space[i]
            dists_i = dij[i,:]
            cut_idx = np.where(dists_i < r_cut)[0]
            #print(dists_i[cut_idx])
            
            idx = np.argsort(dists_i[cut_idx])[0:n_nn]
            idx = cut_idx[idx]
            
            
            for kk in range(len(idx)):
                rj = ps[idx[kk]] 
                dR = rj - ri
                nnr[i, kk, :] = dR
        
        return nnr
    
    def Find_edges(self, NNR, NN_dist = 1.44, num_NN = 3):
        #print('find edges')
        edge_idx = []
        n, nn = NNR.shape[0:2]
        for i in range(n):
            dij = np.sum(NNR[i, :, :]**2, axis=1)**0.5
            dij[dij<1e-5] = 10 ** 6
            idx = np.where(dij < NN_dist)[0]
            if len(idx) < num_NN:
                edge_idx += [i]
        edge_idx = [i for i in edge_idx if self.s[i]>=0]
        return edge_idx
    
    def Passivate(self, s, dist, NN_dist = 1.5, num_NN = 3, verbose=False, retn = False,
                  condition = None):
        if condition is None:
            def cond(r): return True
        else:
            def cond(r): return condition(r)
        if verbose: print('nnr')
        nnr = self.nnr(n_nn = 10, r_cut = 2 * NN_dist)
        if verbose: print('edge idx')
        edge_idx = self.Find_edges(nnr, NN_dist = NN_dist, num_NN = num_NN)
        new_pos = []
        if verbose: print('loop')
        for i in edge_idx:
            ri = self.pos_real_space[i]
            dR = nnr[i]
            dij = np.linalg.norm(dR, axis = 1)
            dij[dij<1e-5] = 10 ** 6
            dR = dR[dij<NN_dist,:]
            vec = dR.sum(axis = 0)/dR.shape[0]
            vec = -vec / np.linalg.norm(vec)
            new_r = ri + vec * dist
            if not np.isnan(vec).any() and cond(new_r):
                new_pos += [new_r]
        if verbose: print('loop done')
        
        new_pos = np.array(new_pos)
        pd = np.vstack((self.pos_real_space, new_pos))
        td = np.hstack((self.s, s * np.ones(len(new_pos), dtype=np.int32)))
        if retn:
            return pd, td
        self.set_struct(self.lat, pd, td)
        
    def Passivate_with_molecule(self, mol, dist, NN_dist = 1.5, num_NN = 3, 
                                align_axis = np.array([1,0,0]),
                                cond_filling = False, 
                                cond=None):
        """mol: ASE molecule
           dist: distance mol is placed from edge
           NN_dist: nearest neighbor distance, used by edge finding algo
           num_NN: default number of nearest neighbor in bulk
           align axis: does something??
           cond_filling: if there is a condition for when to place a molecule somewhere
           cond: function that takes the positions of a molecule placed at a point and return a bool
            """
        
        align_axis = align_axis / np.linalg.norm(align_axis)
        nnr = self.nnr(n_nn = 10, r_cut = 2 * NN_dist)
        edge_idx = self.Find_edges(nnr, NN_dist = NN_dist, num_NN = num_NN)
        new_pos = []
        new_t   = []
        tb  = mol.numbers
        na = len(tb)
        new_pos = np.zeros((na * len(edge_idx),3))
        new_t   = np.zeros( na * len(edge_idx)   , dtype = np.int32)
        it = 0
        for i in edge_idx:
            ri = self.pos_real_space[i]
            dR = nnr[i]
            dij = np.linalg.norm(dR, axis = 1)
            dij[dij<1e-5] = 10 ** 6
            dR = dR[dij<NN_dist,:]
            vec = dR.sum(axis = 0)/dR.shape[0]
            vec = -vec / np.linalg.norm(vec)
            #theta = np.sign(vec[1]) * np.arccos( vec.dot(align_axis) )
            theta = np.arctan2(vec[1], vec[0])
            M_rot = np.array([[np.cos(theta), -np.sin(theta),0],
                              [np.sin(theta),  np.cos(theta),0],
                              [0,                  0,        1]])
            
            for KK in range(na):
                new_pos[it * na + KK, :] = ri + vec * dist + M_rot.dot(mol.positions[KK])
                new_t  [it * na + KK] = tb[KK]
            it+=1
        n_atoms_mol  =  len(mol.numbers)
        N_new        =  len(new_pos)//n_atoms_mol
        if cond_filling:
            idx = []
            for ii in range(N_new):
                if cond(new_pos[ii*n_atoms_mol : ii*n_atoms_mol + n_atoms_mol]):
                    for jj in range(n_atoms_mol):
                        idx += [ii*n_atoms_mol + jj]
            new_pos = new_pos[idx]
            new_t   = new_t[idx]
        
        pd    = np.vstack((self.pos_real_space, new_pos))
        td    = np.hstack((self.s, new_t))
        self.set_struct(self.lat, pd, td)
    
    def x_to_z(self,x = 'x',z = 'z', update_sym_dic = True):
        if x == 'x' and z == 'z':
            U = np.array([[0,0,1],
                          [0,1,0],
                          [1,0,0]])
            switch = [2,1,0]
            
        if x == 'y' and z == 'z':
            U = np.array([[1,0,0],
                          [0,0,1],
                          [0,1,0]])
            switch = [0,2,1]
        if x == 'x'  and z =='y':
            U = np.array([[0,1,0],
                          [1,0,0],
                          [0,0,1]])
            switch = [1,0,2]
        
        if update_sym_dic:
            def rear(k):
                return [k[switch[0]], k[switch[1]],k[switch[2]]]
            new_sym_dic = {}
            for i in self.sym_dic.keys():
                kp = self.sym_dic[i]
                new_sym_dic.update({i:rear(kp)})
            self.sym_dic = new_sym_dic
        pd = self.pos_real_space[:,switch]
        lat = U.T.dot(self.lat).dot(U)
        val = 0
        val += self.Standardize_cell 
        
        self.Standardize_cell = False
        self.set_struct(lat, pd, self.s)
        self.Standardize_cell = bool(val)
    
    def tile_dm(self, n, axis):
        d = sisl.get_sile(self.dir + '/' + self.sl + '.DM').read_density_matrix()
        d = d.tile(n,axis )
        d.write(self.dir + '/' + self.sl + '.DM')
    # def Methfessel_Paxton(self, N, plot = False):
    #     lines = ['OccupationFunction MP', 
    #              'OccupationMPOrder '+ str(N)]
    #     if plot  == True:
    #         from scipy.special import hermite
            
        
    #     self.write_more_fdf()
    def move(self,T, move_elecs = True):
        T = np.array(T)
        new_pos = self.pos_real_space + T
        self.set_struct(self.lat, new_pos, self.s)
        if move_elecs:
            for e in self.elecs:
                e.move(T)
    def pretty_cell(self):
        lat = self.lat.copy()
        ux  = np.array([1,0,0])
        uy  = np.array([0,1,0])
        if ux.dot(lat[0])<0:
            lat[0]*=-1
        if uy.dot(lat[1])<0:
            lat[1]*=-1
        self.set_struct(lat,self.pos_real_space, self.s)
    
    def cell_offset(self):
        return np.floor(self.pos).astype(int)
    def cell_deviation(self):
        return self.UC_idx - self.center_cell() 
    def center_cell(self):
        v = np.average(self.UC_idx, axis=0)
        return np.round(v, 0).astype(int)
        
        
    
    def make_pretty(self, ase_pbc = True):
        geom  =self.to_sisl()
        self.pretty_cell()
        self.move(geom.center(what='cell') - geom.center())
        self.Wrap_unit_cell(ase_pbc = ase_pbc)
    
    # def wrap_elecs(self):
    #     pos_new = self.pos_real_space.copy()
    #     for ie in range(len(self.elecs)):
    #         elec = self.elecs[ie]
    #         elec.Wrap_unit_cell()
    #         dr = elec.pos_real_space - self.pos_real_space[self.elec_inds[ie]]
    #         pos_new[self.elec_inds[ie]] += dr
    #     self.set_struct(self.lat, pos_new, self.s)
        
    
    def Wrap_unit_cell(self,shrink=1e-5,PRINT=False,strict_z=False, cell = None, max_y_tiles = 5,
                       use_ase = True, wrap_elecs = True, wrap_buffer = True, ase_pbc = True, move_elec = True):
        if use_ase:
            sisl_obj = self.to_sisl()
            idx = [i for i in range(len(self.s)) 
                     if i not in list(np.hstack(self.elec_inds)) 
                     and i not in self.buffer_atoms ]
            print(idx)
            if wrap_elecs and self.elec_inds is not None:
                idx += list(np.hstack(self.elec_inds))
            if wrap_buffer:
                idx += self.buffer_atoms
            
            sisl_obj = sisl_obj.sub(idx)
            ase_obj  = sisl_obj.to.ase()
            ase_obj.wrap(pbc = ase_pbc)
            new_pos  = self.pos_real_space.copy()
            new_pos[idx] = ase_obj.positions
            self.set_struct(self.lat, new_pos, self.s)
            if move_elec:
                for ie in range(len(self.elecs)):
                    elec = self.elecs[ie]
                    new_elec_pos = self.pos_real_space[self.elec_inds[ie]]
                    dr           = new_elec_pos - elec.pos_real_space
                    pos          = elec.pos_real_space + dr
                    elec.set_struct(elec.lat, pos, elec.s)
            
            return
        
        from OldEMCode.Build_eps_from_planes import Structure
        if cell is None:
            a1=self.lat[0,:].copy(); a2=self.lat[1,:].copy(); a3=self.lat[2,:].copy()
        else:
            a1, a2, a3 = cell.copy()
        
        a1-= shrink * a1/np.linalg.norm(a1); a2-= shrink * a2/np.linalg.norm(a2)
        a3-= shrink * a3/np.linalg.norm(a3)
        
        
        #zero = -shrink * a1/np.linalg.norm(a1) - shrink * a2/np.linalg.norm(a2)-shrink * a3/np.linalg.norm(a3)
        if strict_z==False: 
            zero=np.array([0,0,-shrink])
            a1 += zero
            a2 += zero
            a3 += zero
        else: zero = np.zeros(3)
        points  = [zero,a1,a1+a2,a2,zero+a3,a1+a3,a1+a2+a3,a2+a3]
        faces = [[0,1,2,3][::-1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        Cell = Structure(points,faces,convex = True)
        P_mid = (a1+a2+a3)/2
        Try   = Cell.Inside_Struc_Convex(P_mid,full_return=True)
        if PRINT==True:
            print(Try, Try.shape)
        it=0
        for x in Try[0,:]:
            if x > 0: 
                faces[it] = faces[it][::-1]
                if PRINT==True:
                    print('face ' +str(it)+' reversed')
            it+=1
        Cell = Structure(points,faces,convex = True)
        Try2 = Cell.Inside_Struc_Convex(P_mid)
        if PRINT==True:
            print(Try2)
        Truth = Cell.Inside_Struc_Convex(self.pos_real_space)
        T=[]
        for i in [0,-1,1]:
            for j in range(-max_y_tiles,max_y_tiles+1):
                if i!=j:
                    T+=[[i,j]]
        
        new_xyz = self.pos_real_space.copy()
        for i in range(len(Truth)):
            pi=new_xyz[i]#self.pos_real_space[i,:]
            if Truth[i] == False:
                Break=False
                for t in T:
                    if Cell.Inside_Struc_Convex(pi+self.lat[0,:]*t[0]+self.lat[1,:]*t[1]) and Break==False:
                        new_xyz[i] = pi+self.lat[0,:]*t[0]+self.lat[1,:]*t[1]
                        Break=True
        self.set_struct(self.lat, new_xyz, self.s)
    
    def Visualise(self,T=[[0,0]],axes=[0,1], Mull_step = -1, 
                  Mull_map = Identity, Mull_which = 0, 
                  adjust_size = 1, annotate = False,
                  size = 20.0, colors = None, drawcell = True,rotax = None, theta = None):
        import matplotlib.pyplot as plt
        color=np.zeros((len(self.pos_real_space),3))
        if colors is None:
            colors = [[0,0,1], [1,0,0],[0,1,0], 
                      [0, 0.5, 0.5],[0.5,0,0.5], [0.5, 0.5, 0.0],
                      [0.25, 1.0, 0.0]]
        us = unique_list(self.s)
        for i,pi in enumerate(self.pos_real_space):
            if self.elec_inds is not None:
                in_elec = False
                for ei in self.elec_inds:
                    if i in ei:
                        color[i,:] = np.array([0.66,0.66,1])
                        in_elec = True
                if in_elec == False:
                    color[i,:] = np.array(colors[us.index(self.s[i])])
            else:
                color[i,:] = np.array(colors[us.index(self.s[i])])
            
            if i in self.buffer_atoms:
                color[i,:] = np.array([0.5,0,1])
        #print(color)
        if hasattr(self, 'Mulliken_C'):
            print('Mulliken plot')
            C = self.Mulliken_C
            idx = self.Mulliken_idx
            Vals = [C[np.where(idx == a), Mull_which][0,Mull_step] for a in np.arange(len(self.pos_real_space))]
            Vals = np.array(Vals)
            Mull = Vals.copy()
            Vals = Vals/Vals.max()
            Vals = Mull_map(Vals)
            for t in T:
                if t == [0,0]:
                    shade = 1.0
                else:
                    shade = 1.0
                Tv = self.lat[0,:]*t[0]+self.lat[1,:]*t[1]
                r_plot=self.pos_real_space+Tv
                plt.scatter(r_plot[:,axes[0]],r_plot[:,axes[1]],c=shade * color, s  = adjust_size * np.pi*Vals**2*100)
                if annotate == True:
                    
                    for ii in range(len(r_plot)):
                        pos = (r_plot[ii, axes[0]],r_plot[ii, axes[1]] )
                        plt.annotate(str(Mull[ii]),pos)
                        
        
        else:
            print('Normal plot')
            
            for t in T:
                if t == [0,0]:
                    shade = 1.0
                else:
                    shade = 1.0
                Tv = self.lat[0,:]*t[0]+self.lat[1,:]*t[1]
                r_plot=self.pos_real_space+Tv
                ss = self.s.copy()
                COLOR = color.copy()
                if rotax is not None and theta is not None:
                    matrix = rotation_matrix(rotax, np.deg2rad(theta))
                    r_plot = (matrix@(r_plot.T)).T
                    sortax = [0,1,2]
                    sortax.pop(axes[0])
                    sortax.pop(axes[1])
                    sortidx = np.argsort(r_plot[:,sortax[0]])[::-1]
                    r_plot = r_plot[sortidx,:]
                    COLOR = COLOR[sortidx]
                    #ss = self.s[sortidx].copy()
                
                plt.scatter(r_plot[:,axes[0]],r_plot[:,axes[1]],c=shade * COLOR, s = size)
                if annotate == True:
                    for i,pi in enumerate(self.pos_real_space):
                        plt.annotate(str(self.s[i]), (pi[axes[0]],pi[axes[1]]) )
        
        if hasattr(self, 'Added_stuff'):
            Sphere_C = []
            R_vec = []
            for d in self.Added_stuff:
                
                if d['what'] == 'charged sphere':
                    Cent, R = d['vectors']
                    x,y = Cent[axes]
                    Sphere_C += [[x,y]]
                    R_vec    +=  [R]
                elif d['what'] == 'dipole correction':
                    pass
                
                else:
                    vecs = d['vectors']
                    O = vecs[0];
                    A = vecs[1];
                    B = vecs[2];
                    C = (O+A + O + B)/2
                    plt.arrow(O[0], O[1], A[0], A[1], color = 'y')
                    plt.arrow(O[0], O[1], B[0], B[1], color = 'y')
                    x, y = C[0], C[1]
                    plt.text(x, y, d['what'])
            if Sphere_C!=[]:
                Sphere_C = np.array(Sphere_C)
                R_vec = np.array(R_vec)
                plt.scatter(Sphere_C[:,0],Sphere_C[:,1], s = R_vec * 30, facecolors='none', edgecolors='y')
        if drawcell:
            plt.arrow(0, 0, self.lat[0,0],self.lat[0,1], linestyle = 'dashed')
            plt.arrow(0, 0, self.lat[1,0],self.lat[1,1], linestyle = 'dashed')
        plt.axis('equal')
        plt.savefig(self.dir + '/Visualise.svg')
    
    def figures(self, nk = 600, subE = None, manual_D = None, 
                axes=[0,1], custom_bp = None, spin=0, 
                bs_lim = (-5, 5), color_o = 'darkblue',color_uo='royalblue'):
        try:
            from Zandpack.plot import plt
        except:
            import matplotlib.pyplot as plt
        
        Elab = r'Energy [eV]'
        Tlab = r'Transmission'
        Dlab = r'DOS [1/eV]'
        self.Visualise(axes=axes)
        plt.close()
        try:
            try:
                tbt = self.read_tbt()
                T   = tbt.transmission()
                E   = tbt.E
            except:
                tbt = np.load(self.dir+'/siesta.fakeTBT.npz')
                T   = tbt['transmission']
                E   = tbt['E']
            finally:
                pass
            if subE is None:
                subE = np.arange(len(E))
            plt.plot(E[subE], T[subE])
            plt.xlim(E.min(), E.max())
            plt.ylim(0, None)
            plt.xlabel(Elab)
            plt.ylabel(Tlab)
            plt.savefig(self.dir+'/Transmission.svg')
            plt.close()
            try:
                if manual_D is None:
                    try:
                        D = tbt.DOS()
                    except:
                        D = tbt['DOS']
                    finally:
                        D = np.zeros(len(E))
                plt.plot(E[subE], D[subE])
                plt.xlim(E.min(), E.max())
                plt.ylim(0, None)
                plt.xlabel(Elab)
                plt.ylabel(Dlab)
                plt.savefig(self.dir+'/DOS.svg')
            except:
                pass
            
        except:
            pass
        plt.close()
        if self.elecs is None:
            return
        def Map(s):
            if 'gamma' in s.lower():
                s =  '\Gamma'
            out = r'$'+s+'$'
            return out
        for i,e in enumerate(self.elecs):
            if custom_bp is None:
                b1,b2 = e.make_bandpath()
            else:
                b1,b2 = custom_bp[i]
            b2 = [Map(b2i) for b2i in b2]
            h  = e.read_TSHS()
            #print(h)
            band = sisl.BandStructure(h, b1, nk, b2)
            # ev = np.zeros((len(bs.k), h.no))
            # for i,ki in enumerate(bs.k):
            #     ev[i] = h.eigh(k=ki)
            # plt.plot
            bs = band.apply.array.eigh(spin=spin)
            lk, kt, kl = band.lineark(True)
            plt.xticks(kt, kl)
            plt.xlim(0, lk[-1])
            plt.ylim([bs_lim[0], bs_lim[1]])
            plt.ylabel('$E-E_F$ [eV]')
            for bk in bs.T:
                bk2 = bk.copy()
                bk2[bk2>0]=np.nan
                plt.plot(lk, bk2,color=color_o)
                bk2 = bk.copy()
                bk2[bk2<0]=np.nan
                plt.plot(lk, bk2,color=color_uo, alpha=0.5)
                
            plt.savefig(e.dir+'/Bandstructure'+str(spin)+'.svg')
            plt.close()
            e.Visualise(axes=axes)
            plt.close()
            
        
    
    
    
    def read_basis_and_geom(self):
        from sisl.io.siesta.basis import ionncSileSiesta
        files = ld(self.dir)
        files = [file for file in files if '.ion.nc' in file]
        B = [ionncSileSiesta(self.dir + '/'+f).read_basis() for f in files]
        geom = None
        try:
            geom = sisl.get_sile(self.dir + '/RUN.out').read_geometry()
        except:
            pass
        
        return B, geom
    
    def ase_visualise(self, elec_replace = None, buffer_replace=None):
        from ase.visualize import view
        ats = self.toASE()
        if elec_replace is not None and self.elec_inds is not None:
            for ei in self.elec_inds:
                ats.numbers[ei] = elec_replace
        if len(self.buffer_atoms) > 0 and buffer_replace is not None:
            ats.numbers[self.buffer_atoms] = buffer_replace
        view(ats)
    
    def run_TS_RSSE_calc(self, tile, tbt_contour, a1 = 0,a2 = 1, 
                         eta      = 0.0, 
                         Contour  = None, 
                         fix_dir  = 'C',
                         eta_negf = 0.01,
                         buffer_cond = None,
                         DOS_GF    = False, 
                         DOS_A     = False, 
                         DOS_A_ALL = False,
                         ORBITAL_CURRENT=False, 
                         Custom    = [],
                         manual_pp = [],
                         relax     = False,
                         movenomove_func= None,
                         initial_num_E = 45,
                         skip_tbtrans = False,
                         only_fail  = False,
                         dk         = 1000.0,
                         dk_fail    = 10.0,
                         parallel_E = False,
                         num_procs  = 4,
                         reuse_fdf  = False,
                         mix_ps = [], # list of several [(at1_i, at2_i), frac_i, mix_i]
                         find_elec_tol = 1e-2,
                         run_tbt_analyze=True):
        """ 
            tile: number of supercells, may also be set with the RSSE/SurRSSE dictionaries
            tbt_contour: Contour for tbtrans calculation
            a1, a2: decimation and integration directions
            eta: probably zero
            Contour: Guess for initial contour for transiesta calculation
            fix_dir: TS.Hartree.Fix  + this direction
            eta_negf: broadening on the greens function calculation for the NEGF calculation in the small window
            ...... document this
            dk: real space integration fineness
            
        """
        from time import sleep
        elec_rsse_idx = [i for i in range(len(self.elecs)) if self.elecs[i].elec_RSSE]
        if Contour is None:
            Contour = np.linspace(-1,1,initial_num_E)+0.25j # Dummy contour
        self.custom_tbtrans_contour = tbt_contour
        for i,e in enumerate(self.elecs):
            if reuse_fdf == False:
                e.fdf(manual_pp = manual_pp)
            elif isinstance(reuse_fdf, list):
                if reuse_fdf[i] == False:
                    e.fdf(manual_pp = manual_pp)
            else:
                pass
            e.run_siesta_electrode_in_dir()
            if i in elec_rsse_idx:
                if e.elec_SurRSSE == False:
                    if hasattr(e, 'RSSE_dict'): e.RSSE_dict['Contour'] = Contour
                    e.Real_space_SE(a1,a2,tile, eta, 0, 0, 2/50, Contour = Contour, dk=dk_fail, mu=self.Chem_Pot[i],
                                    parallel_E=parallel_E, num_procs = num_procs, dummy_calc =False)
                else:
                    if hasattr(e, 'SurRSSE_dict'): e.SurRSSE_dict['Contour'] = Contour
                    if isinstance(a2, list): nsc = np.array([1 for i in range(3)]); nsc[a2] = 3
                    else:                    nsc =          [1 for i in range(3)] ; nsc[a2] = 3
                    e.Real_space_SI(a2, tile, eta, Contour, nsc, dk=dk_fail, mu=self.Chem_Pot[i],
                                    parallel_E=parallel_E, num_procs = num_procs)
        self.Visualise()
        self.find_elec_inds(tol = find_elec_tol)
        if buffer_cond is not None:
            self.set_buffer_atoms(buffer_cond)
        self.fdf(manual_pp = manual_pp)
        self.write_more_fdf(['TS.Hartree.Fix -'+fix_dir], name = 'TS_TBT')
        if self.NEGF_calc:
            self.write_more_fdf(['TS.Contours.nEq.Eta '+str(eta_negf)+' eV'], name = 'TS_TBT')
        if len(mix_ps)>0:
            for commands in mix_ps:
                at1, at2 = commands[0]
                frac     = commands[1]
                name     = commands[2]
                self.ps_mixer([ at1, at2], frac, name)
            self.dump_pseudo_list_to_struct()
        
        if run_tbt_analyze:
            self.run_analyze_in_dir()
        self.run_siesta_in_dir()
        if only_fail:
            return
        sleep(2.0)
        print('\n\n')
        print('The first transiesta calculation is meant to fail! The input contour is read from the failed calculation.\n')
        TS_Contour = self.get_contour_from_failed_RSSE()
        print('\n Read contour with shape ', TS_Contour.shape, '\n')
        TS_Contour = TS_Contour[:,0] + 1j * TS_Contour[:,1] # get the contour transiesta asks for
        for i,e in enumerate(self.elecs):
            if reuse_fdf == False:
                e.fdf(manual_pp = manual_pp)
            elif isinstance(reuse_fdf, list):
                if reuse_fdf[i] == False:
                    e.fdf(manual_pp = manual_pp)
            else:
                pass
            
            e.run_siesta_electrode_in_dir()
            if i in elec_rsse_idx:
                if e.elec_SurRSSE == False:
                    if hasattr(e, 'RSSE_dict'): e.RSSE_dict['Contour'] = TS_Contour
                    e.Real_space_SE(a1,a2,tile, eta, 0, 0, 2/50, 
                                    Contour = TS_Contour,     
                                    dk=dk, 
                                    mu=self.Chem_Pot[i],
                                    parallel_E=parallel_E, num_procs = num_procs)
                else:
                    if hasattr(e, 'SurRSSE_dict'): e.SurRSSE_dict['Contour'] = TS_Contour
                    if isinstance(a2, list): nsc = np.array([1 for i in range(3)]); nsc[a2] = 3
                    else:                    nsc =          [1 for i in range(3)] ; nsc[a2] = 3
                    #nsc = [1 for i in range(3)]; nsc[a2] = 3
                    e.Real_space_SI(a2, tile, eta,                         
                                    TS_Contour, 
                                    nsc,
                                    dk=dk, 
                                    mu=self.Chem_Pot[i],
                                    parallel_E=parallel_E, num_procs = num_procs)
        
        if relax:
            from funcs import listinds_to_string, numpy_inds_to_string
            move = [i+1 for i in range(len(self.s)) 
                    if movenomove_func(self,i) == True]
            nomove = [i+1 for i in range(len(self.s)) 
                      if movenomove_func(self,i) == False]
            outp1 = listinds_to_string(numpy_inds_to_string(np.array(move)))
            outp2 = listinds_to_string(numpy_inds_to_string(np.array(nomove)))
            if len(outp1)>100:
                outp1 = outp1.replace('[','').replace(']','')
                outp1 = outp1.split(',')
                outp2 = outp2.replace('[','').replace(']','')
                outp2 = outp2.split(',')
                for i in range(len(outp1)):
                    outp1[i] = 'clear [' + outp1[i]+']'
                for i in range(len(outp2)):
                    outp2[i] = 'atom [' + outp2[i] + '] 0 0 0'
                constraints = outp1 + outp2
                
            else:
                C1  = 'clear ' + outp1
                C2  = 'atom  ' + outp2+' 0 0 0'
                constraints = [C1,C2]
            self.fdf_relax(constraints)
            self.write_more_fdf(['TS.Hartree.Fix -'+fix_dir], name = 'TS_TBT')
        else:
            self.fdf(manual_pp = manual_pp)
            self.write_more_fdf(['TS.Hartree.Fix -'+fix_dir], name = 'TS_TBT')
            if self.NEGF_calc:
                self.write_more_fdf(['TS.Contours.nEq.Eta '+str(eta_negf)+' eV'], name = 'TS_TBT')
            if len(mix_ps)>0:
                for commands in mix_ps:
                    at1, at2 = commands[0]
                    frac     = commands[1]
                    name     = commands[2]
                    self.ps_mixer([ at1, at2 ], frac, name)
                self.dump_pseudo_list_to_struct()

        if run_tbt_analyze:
            self.run_analyze_in_dir()
        self.run_siesta_in_dir()
        if skip_tbtrans:
            return
        
        for i,e in enumerate(self.elecs):
            if reuse_fdf == False:
                e.fdf(manual_pp = manual_pp)
            elif isinstance(reuse_fdf, list):
                if reuse_fdf[i] == False:
                    e.fdf(manual_pp = manual_pp)
            else:
                pass
            
            e.run_siesta_electrode_in_dir()
            if i in elec_rsse_idx:
                if e.elec_SurRSSE == False:
                    if hasattr(e, 'RSSE_dict'): e.RSSE_dict['Contour'] = tbt_contour
                    e.Real_space_SE(a1,a2,tile, eta, 0, 0, 2/50, Contour = tbt_contour,dk=dk, mu=self.Chem_Pot[i],
                                    parallel_E=parallel_E, num_procs = num_procs)
                else:
                    if hasattr(e, 'SurRSSE_dict'): e.SurRSSE_dict['Contour'] = tbt_contour
                    nsc = np.array([1 for i in range(3)]); nsc[a2] = 3
                    e.Real_space_SI(a2, tile, eta, tbt_contour, nsc, dk=dk, mu=self.Chem_Pot[i],
                                    parallel_E=parallel_E, num_procs = num_procs)
        
        if run_tbt_analyze:
            self.run_analyze_in_dir()
        self.run_tbtrans_in_dir(DOS_GF=DOS_GF, DOS_A=DOS_A,ORBITAL_CURRENT=ORBITAL_CURRENT, Custom=Custom)
        
        
    def self_energy_from_tbtrans(self, E, k):
        self.save_SE      = True
        self.save_SE_only = True
        _kp_tbtrans       = self.kp_tbtrans.copy()
        self.kp_tbtrans   = None
        self.manual_tbtrans_kpoint = k
        self.custom_tbtrans_contour = E
        self.fdf()
        self.run_tbtrans_in_dir()
        delattr(self,'manual_tbtrans_kpoint')
        self.kp_tbtrans = _kp_tbtrans
        self.save_SE = False
        self.save_SE_only = False
        self.custom_tbtrans_contour = None
        SE,inds = read_SE_from_tbtrans(self.dir + '/' + self.sl+'.TBT.SE.nc')
        os.remove(self.dir + '/' + self.sl+'.TBT.SE.nc')
        return SE, inds
    
    def tbtrans_H_S_btd_pivot(self):
        t   = sisl.get_sile(self.dir + '/'+self.sl+'.TBT.nc')
        p   = t.pivot()
        btd = t.btd()
        H = sisl.get_sile(self.dir + '/'+self.sl+'.TSHS').read_hamiltonian()
        S = sisl.get_sile(self.dir + '/'+self.sl+'.TSHS').read_overlap()
        self.tbtrans_params_dic = {'H': H,
                                   'S': S,
                                   'pivot': p,
                                   'btd': btd}
        
    
    def solve_qp_equation(self,E, k, its = 10):
        if not hasattr(self, 'tbtrans_params_dic'):
            self.tbtrans_H_S_btd_pivot()
        
        p        = self.tbtrans_params_dic['pivot']
        sk       = self.tbtrans_params_dic['S'].Sk(k = k).toarray()
        hk       = self.tbtrans_params_dic['H'].Hk(k = k).toarray()
        esk, vsk = np.linalg.eigh(sk[p, :][:, p])
        lowdin   = vsk.dot(np.diag(1/np.sqrt(esk))).dot(vsk.T.conj())
        i_lowdin = vsk.dot(np.diag(np.sqrt(esk))).dot(vsk.T.conj())
        
        E0   = 0.0
        E0  += E
        
        out1 = np.zeros(its,dtype = np.complex128)
        out2 = np.zeros(its,dtype = np.complex128)
        for count in range(its):
            se, inds = self.self_energy_from_tbtrans(np.array([E0]), k)
            se       = [se[i][0,0] for i in range(len(se))]
            
            inds     = [np.array(inds[i])[:,np.newaxis] for i in range(len(inds))]
            QP_Ham    = hk.copy()
            iSE       = np.zeros(hk.shape, dtype=complex)
            for e in range(len(se)):
                QP_Ham[inds[e], inds[e].T]  += (se[e] + se[e].conj().T)/2#se[e].real
                iSE    [inds[e], inds[e].T] += 1j*(se[e] - se[e].conj().T)/2#se[e].imag
            QP_Ham = QP_Ham[p][:,p]
            iSE    =    iSE[p][:,p]
            QP_Ham = lowdin @ QP_Ham @ lowdin
            iSE    = lowdin @ iSE @ lowdin
            e,v    = np.linalg.eigh(QP_Ham)
            D      = np.abs(e - E0)
            idx = np.where(D==D.min())
            idx = idx[0][0]
            E0  = e[idx]
            vec = v[:,idx]
            out1[count] = E0
            out2[count] = (vec.conj()).dot(iSE).dot(vec)
        return E0 ,i_lowdin.dot(vec), out1, out2
    
    def sisl_sub(self,idx):
        g = self.to_sisl()
        g = g.sub(idx)
        xyz = g.xyz
        numbers = g.atoms.Z.copy()
        self.set_struct(self.lat, xyz, numbers)
    def move_atom(self, idx, T):
        _T = np.array(T)
        new_xyz = self.pos_real_space.copy()
        new_xyz[idx] += _T
        self.set_struct(self.lat, new_xyz, self.s)
    
    def add_buffer(self, elec, N, direc,vac = 0.0):
        g = self.to_sisl()
        e = self.elecs[elec].to_sisl()
        sign = 1 if '+' in self.elecs[elec].semi_inf else -1
        e = e.tile(N,direc).move(sign*e.cell[direc])
        g = g.add_vacuum(vac, direc)
        g = g.add(e)
        xyz = g.xyz
        self.set_struct(g.cell.copy(), xyz, g.atoms.Z.copy())
    
    def pickle(self, filename = None):
        """Saves calculator to file"""
        
        import pickle as pkl
        if filename is None:
            filename = self.dir
        f = open(filename +'.SiP', 'wb')
        pkl.dump(self, f)
        f.close()
    def read_tbt(self, spin = None):
        if spin is None:
            return sisl.get_sile(self.dir+'/'+self.sl + '.TBT.nc')
        elif spin == 'UP':
            return sisl.get_sile(self.dir+'/'+self.sl + '.TBT_UP.nc')
        elif spin == 'DN':
            return sisl.get_sile(self.dir+'/'+self.sl + '.TBT_DN.nc')
        elif spin == 0:
            return sisl.get_sile(self.dir+'/'+self.sl + '.TBT_UP.nc')
        elif spin == 1:
            return sisl.get_sile(self.dir+'/'+self.sl + '.TBT_DN.nc')
            
    
    
    def add_atom(self, pos= None, s = None):
        if isinstance(pos, SiP):
            self.add_atom(pos.pos_real_space, pos.s)
            return
        if isinstance(pos, sisl.Geometry):
            self.add_atom(pos.xyz, pos.atoms.Z.copy())
            return
        
        if pos.shape==(3,):
            N = len(self.s)
            newpos = np.zeros((N+1,3))
            news   = np.zeros(N+1,dtype=int)
            newpos[0:N,:] = self.pos_real_space[:,:]
            newpos[N, :]  = pos
            news[0:N]     = self.s[:]
            news[N  ]     = s
            self.set_struct(self.lat, newpos, news)
        elif len(pos.shape)==2:
            N = len(self.s)
            Nn = len(pos)
            newpos = np.zeros((N+Nn,3))
            news   = np.zeros(N+Nn,dtype=int)
            newpos[0:N,:]  = self.pos_real_space[:,:]
            newpos[N:, :]  = pos
            news[0:N]      = self.s[:]
            news[N:  ]     = s
            self.set_struct(self.lat, newpos, news)
    
    
    
    
    
    
    def projection(self, Emin, Emax, sub_orbital = [], eigenstates = True, custom_v = None, return_iG = False, fromwhat='TSHS',
                   nolowdin=False):
        from Gf_Module.Gf import read_SE_from_tbtrans, Greens_function
        from Block_matrices.Block_matrices import Blocksparse2Numpy
        if self.save_SE == False:
            print('You should save the self energies for a proper projection to be done')
            error()
        
        print('This might only work for DFT')
        H, S     = self.to_sisl(what = fromwhat)
        SE, inds = read_SE_from_tbtrans( self.dir + '/siesta.TBT.SE.nc')
        sile     = sisl.get_sile(self.dir + '/siesta.TBT.nc')
        try:
            print('Succesfully read fermilevel')
            E_F = sisl.get_sile(self.dir + '/RUN.fdf').read_fermi_level()
        except:
            print('failed to read fermilevel')
            E_F = 0.0
        
        p   = sile.pivot()
        btd = sile.btd()
        
        if len(sub_orbital) != 0:
            new_pivot = np.array([i for i in p if i in sub_orbital])
            new_btd   = []
            cs        = np.cumsum(btd)
            cs = np.hstack((np.zeros(1), cs)).astype(np.int32)
            for II in range(len(cs)-1):
                count = 0
                for III in p[cs[II]:cs[II+1]]:
                    if III in sub_orbital:
                        count+=1
                new_btd += [count]
            p   = np.array(new_pivot).astype(np.int32)
            btd = np.array(new_btd  ).astype(np.int32)
        
        
        P   = [0]
        for b in btd:
            P+= [P[-1] + b ]
        
        self_Es   = [ SE   ]
        self_inds = [ inds ]
        Piv       = [  p   ]
        Part      = [  P   ]
        Eg = sile.E
        kv = sile.k
        
        #print(SE[0].shape, Eg.shape)
        Sys = Greens_function(H, Piv, Part, sisl_S = S) 
        Sys.set_SE(self_Es, self_inds)
        Sys.set_eta(0.0)
        Sys.set_ev(Eg)
        Sys.set_kv(kv)
        if return_iG:
            return Sys.iG(0, nolowdin=nolowdin)+ (Sys,)
        
        iG, Gam, Lowdin, Hamiltonian, Overlap, self_energies = Sys.iG(0)
        
        slices = Hamiltonian.all_slices
        
        Ortho_Hamiltonian    =  Lowdin[0].BDot(Hamiltonian).BDot(Lowdin[0])   # S^-1/2   H   S^-1/2
        Ortho_SE             = [Lowdin[0].BDot(SE         ).BDot(Lowdin[0])   # S^-1/2 Sigma S^-1/2
                                for SE in self_energies]
        
        #del Hamiltonian, Lowdin, self_energies
        H  =  Blocksparse2Numpy(Ortho_Hamiltonian,slices)
        SE = [Blocksparse2Numpy(SE,slices) for SE in Ortho_SE]
        S  =  Blocksparse2Numpy(Overlap, slices)
        
        if eigenstates:
            e,v = np.linalg.eigh(H)
            
            idx = []
            for i in range(H.shape[-1]):
                if ((Emin<e[:,:,i])*(e[:,:,i]<Emax)).any():
                    idx+=[i]
        
            e = e[:,:,idx]
            v = v[:,:,:,idx]
        elif custom_v is not None:
            v = custom_v
        else:
            print('Could not make sense of your input to the projection rutine')
            error()
        
        def proj(M):
            return (v.transpose(0,1,3,2).conj())@M@v
        print(H.shape)
        H  =  proj(H); Ns = H.shape[-1]
        Sn = np.zeros(H.shape, dtype = np.complex128); Sn[:,:,np.arange(Ns), np.arange(Ns)] = 1
        SE =  [proj(se) for se in SE]
        
        SiP_proj = SiP(self.lat, self.pos_real_space, s = self.s, 
                       directory_name = self.dir + '_proj_['+str(Emin)+'; '+str(Emax)+']',
                       sm = self.sm, sl = self.sl, 
                       )
        
        Dr = SiP_proj.dir
        sl = SiP_proj.sl
        
        # 😄🍊🤮 python can do emojis
        
        np.savez(Dr + '/' +sl+'.fakeTBT.npz',
                 transmission = sile.transmission(),
                 real_pivot   = sile.pivot(),
                 real_btd     = sile.btd(),
                 pivot        = np.arange(Ns),
                 btd          = [Ns],
                 E            = sile.E,
                 _tbtTk_full  = np.array([sile.transmission(kavg = i) 
                                          for i in range(len(sile.k))]),
                 kv           = sile.k,
                 wkpt         = sile.wkpt,
                 E_F          = E_F
                 )
        
        if self.custom_tbtrans_contour is not None:
            Cont = self.custom_tbtrans_contour.astype(np.complex128)
        else:
            Cont = sile.E.real
        
        np.savez(Dr + '/' +sl+'.fakeTBT.SE.npz',
                 SE   = SE,
                 inds = np.arange(Ns),
                 Contour = Cont,
                 )
        
        np.savez(Dr + '/' +sl+'.fakeTSHS.npz',
                 H = H[:,0,:,:],
                 S = Sn[:,0,:,:],
                 k = sile.k)
        
        np.savez(Dr + '/'+sl+'Proj2ortho.npz',
                 projvec  = v,
                 Overlap  = S,
                 Lowdin   =  Blocksparse2Numpy(Lowdin[0],slices),
                 )
        
        return SiP_proj
    
    def Hubbard_electrode(self,up0,dn0, SE = False, add_spin = True, 
                          U = 3.0, kT = 0.025, kp = [102,1,1],
                          n0 = None,return_MFH=False):
        from hubbard import HubbardHamiltonian, density
        H0     = sisl.get_sile(self.dir+'/'+self.sl+'.TSHS').read_hamiltonian()
        if add_spin:
            H0 = H0.transform(spin = sisl.Spin('polarized'))
        MFH_H0 = HubbardHamiltonian(H0, U=U, nkpt=kp, kT=kT)
        MFH_H0.set_polarization(up0, dn = dn0)
        dn   = MFH_H0.converge(density.calc_n, tol=1e-10)
        dist = sisl.get_distribution('fermi_dirac', smearing=kT)
        Ef_elec = MFH_H0.H.fermi_level(MFH_H0.mp, q=MFH_H0.q, distribution=dist)
        print("Electrode Ef = ", Ef_elec)
        # Shift each electrode with its Fermi-level and write it to netcdf file
        MFH_H0.H.shift(-Ef_elec)
        MFH_H0.H.write(self.dir+'/Hubbard_'+self.sl+'.TSHS')
        np.save(self.dir+'/Density.npy',MFH_H0.n)
        np.save(self.dir+'/Shift.npy', -Ef_elec)
        self.Hubbard_electrode = 'Yes'
        if return_MFH:
            return MFH_H0
        
    
    def calculate_hubbard_transport(self, Contour, start = 10):
        from sisl import RecursiveSI
        from funcs import get_btd_partition_of_matrix
        from scipy.sparse import csr_matrix
        import matplotlib.pyplot as plt
        
        SEs = []
        def translate(dr):
            if '-a1' in e.semi_inf: return '-A'
            if '+a1' in e.semi_inf: return '+A'
            if '-a2' in e.semi_inf: return '-B'
            if '+a2' in e.semi_inf: return '+B'
            if '-a3' in e.semi_inf: return '-C'
            if '+a3' in e.semi_inf: return '+C'
            
        for e in self.elecs:
            #if hasattr(e, 'Hubbard_electrode'):
            #    SEs += [RecursiveSI(e.read_TSHS('Hubbard_'+e.sl), 
            #                        translate(e.semi_inf)) ]
            #else:
            SEs += [RecursiveSI(e.read_TSHS(e.sl), 
                                translate(e.semi_inf)) ]
        
        
        
        eSEs_s0 = [np.array([se.self_energy(e,spin=0) for e in Contour]) 
                   for se in SEs]
        eSEs_s1 = [np.array([se.self_energy(e,spin=1) for e in Contour])
                   for se in SEs]
        eSEs    = [np.vstack([eSEs_s0[i][None,:,:,:], eSEs_s1[i][None,:,:,:]])
                   for i in  range(len(SEs))]
        
        Hd      = self.read_TSHS()
        SE_inds = []
        for i,e in enumerate(self.elecs):
            se_inds = []
            for j in self.elec_inds[i]:
                Is = Hd.a2o(j)
                if isinstance(Is,np.int64): se_inds += [Is]
                if isinstance(Is,list):     se_inds += Is
            SE_inds += [se_inds]
        nonzeros = Hd.Hk()
        for inds in SE_inds:
            ones = csr_matrix((Hd.no, Hd.no))
            ones[np.array(inds)[:,None],np.array(inds)[None,:]] = 1
            nonzeros+=ones
        nonzeros = abs(nonzeros)
        piv,btd,part = _BTDP(nonzeros, start)
        kw_dict   = {}
        for i in range(len(SEs)):
            kw_dict.update({'SE_'+str(i):eSEs[i]})
            kw_dict.update({'inds_'+str(i):SE_inds[i]})
        
        np.savez(self.dir + '/' +self.sl+'.fakeTBT.SE.npz',
                 **kw_dict,
                 Contour = Contour,
                 Mode2=1
                 )
        
        H = np.vstack([Hd.Hk(spin=0).toarray()[None,:,:],
                       Hd.Hk(spin=1).toarray()[None,:,:]
                      ])
        
        S = np.vstack([Hd.Sk(spin=0).toarray()[None,:,:],
                       Hd.Sk(spin=1).toarray()[None,:,:]
                      ])
        
        kv = np.array([[0,0          , 0],
                       [0,0.123456789, 0]])
        
        np.savez(self.dir + '/' +self.sl+'.fakeTSHS.npz',
                 H = H,
                 S = S,
                 k = kv)
        
        #replace the real file with the fake
        os.system('mv '+self.dir+'/'+self.sl+'.TSHS ' +self.dir+'/'+self.sl+'.hiddenTSHS ')
        
        
        G0 = np.eye(Hd.no)[None,:,:] * Contour[:,None,None] - Hd.Hk(spin = 0).toarray()
        G1 = np.eye(Hd.no)[None,:,:] * Contour[:,None,None] - Hd.Hk(spin = 1).toarray()
        SEf0 = np.zeros((len(SEs),len(Contour),)+Hd.Hk().shape, dtype=complex)
        SEf1 = np.zeros((len(SEs),len(Contour),)+Hd.Hk().shape, dtype=complex)
        for i in range(len(SEs)):
            idx = np.array(SE_inds[i])
            #print(eSEs_s0[i].shape,SEf0[i,:, idx[:,None], idx[None,:]].shape)
            SEf0[i][:, idx[:,None], idx[None,:]] = eSEs_s0[i]
            SEf1[i][:, idx[:,None], idx[None,:]] = eSEs_s1[i]
        G0 -= SEf0.sum(axis=0)
        G1 -= SEf1.sum(axis=0)
        G0 = np.linalg.inv(G0)
        G1 = np.linalg.inv(G1)
        Tij0 = np.zeros((len(Contour), len(SEs), len(SEs)),dtype=complex)
        Tij1 = np.zeros((len(Contour), len(SEs), len(SEs)),dtype=complex)
        
        for i in range(len(SEs)):
            Gam0i = 1j * (SEf0[i] - SEf0[i].conj().transpose(0,2,1) )
            Gam1i = 1j * (SEf1[i] - SEf1[i].conj().transpose(0,2,1) )
            for j in range(i+1,len(SEs)):
                Gam0j       = 1j * (SEf0[j] - SEf0[j].conj().transpose(0,2,1) )
                Gam1j       = 1j * (SEf1[j] - SEf1[j].conj().transpose(0,2,1) )
                Tij0[:,i,j] = np.trace(Gam0i@G0@Gam0j@(G0.conj().transpose(0,2,1)), axis1=1,axis2=2)
                Tij1[:,i,j] = np.trace(Gam1i@G1@Gam1j@(G1.conj().transpose(0,2,1)), axis1=1,axis2=2)
        
        transmission = Tij0[:,0,1]+Tij1[:,0,1]
        
        np.savez(self.dir + '/' +self.sl+'.fakeTBT.npz',
                 transmission = transmission,
                 real_pivot   = piv,
                 real_btd     = btd,
                 pivot        = piv,
                 btd          = btd,
                 E            = Contour.real,
                 _tbtTk_full  = np.array([Tij0[:,0,1], 
                                          Tij1[:,0,1]]),
                 kv           = kv,
                 wkpt         = [1, 1],
                 E_F          = 0.0
                 )
    def is_RSSE(self):
        if self.elec_2E_RSSE:
            return True
        if self.elec_RSSE:
            return True
        if self.elec_SurRSSE:
            return True
        return False
    def which_type_SE(self):
        if self.is_RSSE() == False:
            return 'normal'
        else:
           if (self.elec_RSSE == True
               and self.elec_SurRSSE==False
               and self.elec_2E_RSSE==False):
               return 'RSSE'
           elif (self.elec_RSSE == True
                 and self.elec_SurRSSE==True
                 and self.elec_2E_RSSE==False):
               return 'surfaceRSSE'
           elif (self.elec_RSSE == True
                 and self.elec_SurRSSE==True
                 and self.elec_2E_RSSE==True):
               return 'twoelecRSSE'
           elif (self.elec_RSSE == True and
                 self.elec_3D_RSSE==True):
               return '3Dbulk'
           else:
               print('couldnt determine RSSE type?')
               return 'unknown'
              
              
           
        return False
    
    def calculate_2E_RSSE(self, Ev, tdir=0, kdir=1, n_jobs = 4, 
                          tol = (1e-6, 1e-4), ty= 1,
                          workers =1,saveGR = False,
                          SE_tol =1e-5, dummy_calc = False,
                          atoms_coupling = True):
        from SelfEnergyCalculators        import Decimation
        from siesta_optics.LinearResponse import sisl2array
        from siesta_optics.LinearResponse import sisl2coup
        
        if len(self.elecs)!=2:
            print('This function can only use two electrodes!')
            return
        kwdict = {}
        
        Hm, Sm = self.elecs[0].to_sisl('fdf')
        Hp, Sp = self.elecs[1].to_sisl('fdf')
        Hd, Sd = self.to_sisl('fdf')
        self.Deci      = Decimation
        ny             = Hd.nsc[kdir]//2
        idx_e1, idx_e2 = self.elec_inds
        idx_b          = self.buffer_atoms
        
        def a2o_d(i):
            return Hd.a2o(i, all=True)
        
        
        _oidx_e1  = [ list(a2o_d(i)) if isinstance(a2o_d(i), np.ndarray) else [a2o_d(i)] for i in idx_e1 ]
        oidx_e1   = []
        for u in _oidx_e1:
            oidx_e1 += u
        
        _oidx_e2  = [ list(a2o_d(i)) if isinstance(a2o_d(i), np.ndarray) else [a2o_d(i)] for i in idx_e2 ]
        oidx_e2   = []
        for u in _oidx_e2:
            oidx_e2 += u
        
        _oidx_b  = [ list(a2o_d(i)) if isinstance(a2o_d(i), np.ndarray) else [a2o_d(i)] for i in idx_b ]
        oidx_b   = []
        for u in _oidx_b:
            oidx_b += u
        
        oidx_d   = list(set([i for i in range(Hd.no)]) - set(oidx_e1) - set(oidx_e2) - set(oidx_b))
        _eidx    = (list(idx_e1)  + list(idx_e2) )
        #_oeidx   = (list(oidx_e1) + list(oidx_e2))
        _didx    = [i for i in range(Hd.na) if i not in (_eidx+idx_b)]
        #_odidx= [i for i in range(Hd.no) if i not in _oeidx]
        
        kwdict.update({'buffer_orbs'  : oidx_b})
        kwdict.update({'buffer_atoms' : idx_b })
        
        Hdsub    = Hd.sub(_didx)
        Sdsub    = Sd.sub(_didx)
        #print(Hdsub.shape)
        if len(self.buffer_atoms)!=0:
            print('Buffer atoms present')
            #oidx_d = [i for i in oidx_d if i not in self.buffer_atoms]
            #Hdsub    = Hdsub.remove(self.buffer_atoms)
            #Sdsub    = Sdsub.remove(self.buffer_atoms)
        
        noL, noD, noR  = Hm.no, Hdsub.no, Hp.no
        ArL = np.zeros((4, Hd.nsc[kdir], noL, noL),dtype=np.complex128)
        ArD = np.zeros((2, Hd.nsc[kdir], noD, noD),dtype=np.complex128)
        ArR = np.zeros((4, Hd.nsc[kdir], noR, noR),dtype=np.complex128)
        Vdl = np.zeros((   Hd.nsc[kdir], noD, noL),dtype=np.complex128)
        Vdr = np.zeros((   Hd.nsc[kdir], noD, noR),dtype=np.complex128)
        Sdl = np.zeros((   Hd.nsc[kdir], noD, noL),dtype=np.complex128)
        Sdr = np.zeros((   Hd.nsc[kdir], noD, noR),dtype=np.complex128)
        
        if kdir==1:
            y = np.array([0,1,0])
            Tv= np.array([1,0,0])
        else:
            y = np.array([1,0,0])
            Tv= np.array([0,1,0])
        #print(y)
        for ic, i in enumerate(range(-ny, ny+1)):
            ArL[0,ic],ArL[2,ic]  = sisl2coup(Hm, Sm, i*y + 0*Tv)
            ArL[1,ic],ArL[3,ic]  = sisl2coup(Hm, Sm, i*y - 1*Tv)
            
            ArR[0,ic],ArR[2,ic]  = sisl2coup(Hp, Sp, i*y + 0*Tv)
            ArR[1,ic],ArR[3,ic]  = sisl2coup(Hp, Sp, i*y + 1*Tv)
            
            ArD[0,ic], ArD[1,ic] = sisl2coup(Hdsub, Sdsub, i*y)
            
            _h,_s = sisl2coup(Hd, Sd, i*y)
            #print(len(oidx_d))
            #print(len(oidx_e1))
            #print(Vdl.shape)
            Vdl[ic], Sdl[ic] = _h[oidx_d,:][:,oidx_e1], _s[oidx_d,:][:,oidx_e1]
            Vdr[ic], Sdr[ic] = _h[oidx_d,:][:,oidx_e2], _s[oidx_d,:][:,oidx_e2]
        
        # assert np.allclose(Vdl[0], Vdl[2].T)
        # assert np.allclose(Vdr[0], Vdr[2].T)
        self.RSSEdata = {'Ev' : Ev,
                         'ArL': ArL,
                         'ArR': ArR,
                         'ArD': ArD,
                         'Vdl': Vdl,
                         'Vdr': Vdr,
                         'Sdr': Sdr,
                         'Sdl': Sdl
                         }
        
        assert np.allclose(ArD[:,0], ArD[:,2].transpose(0,2,1))
        assert np.allclose(ArR[[0,2],0], ArR[[0,2],2].transpose(0,2,1))
        assert np.allclose(ArL[[0,2],0], ArL[[0,2],2].transpose(0,2,1))
        # return ArL, ArR, ArD, Vdl, Vdr, Sdl, Sdr
        if dummy_calc == False:
            RES = self.Deci.par_integrate_GkLDR(Ev, ArL, ArD, ArR, Vdl, Vdr, Sdl, Sdr,
                                                n_jobs=n_jobs, tol=tol, 
                                                ty=ty, workers = workers)
        #else:
        #    _I = np.eye(noD * ty)
        #    RES = [_I[None,:,:] for e in Ev]
        #    #print(ArD[0,0].shape)
            
            
        
        if ty == 1:
            HR  = ArD[0,ny]
            SR  = ArD[1,ny]
            vals       = np.ones(3)
            vals[kdir] = Hdsub.nsc[kdir]
            _Hdsub = Hdsub.copy()
            _Hdsub.set_nsc(vals)
            _Hdsub.write(self.dir + '/RSSE_tiled_Hd.TSHS')
        else:
            Hdsub2     = Hdsub.copy()
            vals       = np.ones(3)
            vals[kdir] = Hdsub.nsc[kdir]
            Hdsub2.set_nsc(vals)
            Hdsub2     = Hdsub2.tile(ty,kdir)
            Hdsub2.set_nsc((1,1,1))
            HR         = Hdsub2.Hk().toarray()
            SR         = Hdsub2.Sk().toarray()
            kwdict.update({'Htilexyz':Hdsub2.xyz.copy(),'tile':ty})
            Hdsub2.write(self.dir + '/RSSE_tiled_Hd.TSHS')
        
        G            = np.array([res[0] for res in RES])
        SER          = Ev[:, None, None]*SR-HR-np.linalg.inv(G)
        coupling_idx = np.unique(np.where((np.abs(SER)>SE_tol).sum(axis=0))[0])
        
        if ty==1:
            o2a = [Hdsub.o2a(i) for i in coupling_idx]
            o2a_nodub = []
            for i in o2a:
                if i not in o2a_nodub: o2a_nodub.append(i)
        else:
            o2a = [Hdsub2.o2a(i) for i in coupling_idx]
            o2a_nodub = []
            for i in o2a:
                if i not in o2a_nodub: o2a_nodub.append(i)
        if atoms_coupling:
            new_cidx = np.array([], dtype=int)
            _dummy   = _Hdsub if ty == 1 else Hdsub2
            for ia in o2a_nodub:
                new_cidx = np.hstack((new_cidx, np.arange(_dummy.a2o(ia), _dummy.a2o(ia+1))))
            coupling_idx = new_cidx.copy()
        
        kwdict.update({'o2a_cidx':o2a, 'o2a_nodub':o2a_nodub})
        SER          = SER[:,coupling_idx,:][:,:,coupling_idx].astype(np.complex64)
        if saveGR:
            kwdict.update({'RealspaceG':G.astype(np.complex64)})
        if ty == 1:
            XYZ  = Hdsub.xyz.copy()[o2a_nodub]
            XYZo = Hdsub.xyz.copy()[o2a]
            S    = Hdsub.atoms.Z[o2a_nodub]
            So   = Hdsub.atoms.Z[o2a]
            Hdsub.sub(o2a_nodub).write(self.dir+'/2E_RSSE.TSHS')
        else:
            XYZ  = Hdsub2.xyz.copy()[o2a_nodub]
            XYZo = Hdsub2.xyz.copy()[o2a]
            S    = Hdsub2.atoms.Z[o2a_nodub]
            So   = Hdsub2.atoms.Z[o2a]
            Hdsub2.sub(o2a_nodub).write(self.dir+'/2E_RSSE.TSHS')
        
        np.savez_compressed(self.dir+'/RSSE.npz',
                            RealspaceSE = SER,
                            Dvcxyz      = self.pos_real_space,
                            coupling_idx= coupling_idx,
                            xyz         = XYZ,
                            xyzo        = XYZo,
                            Z           = S,
                            Zo          = So,
                            Ev          = Ev,
                            HR          = HR,
                            SR          = SR,
                            e1_atoms    = idx_e1,
                            e2_atoms    = idx_e2,
                            e1_orbs     = oidx_e1,
                            e2_orbs     = oidx_e2,
                            **kwdict
                           )
        self.Deci = None
        self.old_sl = self.sl + ''
        self.sl   = '2E_RSSE'
    
    def calculate_bulksurf_RSSE(self, Contour=None, supercell=None, init_grid_N=8, 
                                tol=1e-2, minarea=1e-5, return_bulkSE=False,
                                tol_elec_pos=1e-3, bulk_se = False, 
                                return_calc=False,
                                nprocs=1, add_command='',reuse=False, debug_with_sisl=False,
                                pivot_start = None, start = "mpirun",prog_suffix="",
                                use_se_interp = "False", 
                                # NEW
                                do_periodic = False,
                                kxy = None,
                                nk_grid = None,
                                ):
        if do_periodic == False:
            assert kxy is None
            assert nk_grid is None
        assert Contour is not None
        assert supercell is not None
        assert self.elec_3D_RSSE
        assert len(self.elecs)==1
        is_pol = self.spin_pol in ['polarized', 'polarised']
        if (np.imag(Contour)<1e-7).any():
            print('You have a very small broadening in your energy sampling. Consider setting e.g. eta=1e-3j.')
        np.save('inputE.npy', Contour)
        if len(self.buffer_atoms)==0:
            sHname = self.dir+'/'+self.sl+'.TSHS'
        else:
            print('SiP removed buffer atoms: ', self.buffer_atoms)
            Hs = self.read_TSHS()
            Hs = Hs.remove(self.buffer_atoms)
            sHname = 'hsurface.TSHS'
            Hs.write(sHname)
        tx, ty = supercell[0], supercell[1]
        E=self.elecs[0]
        if E.elec_bloch is None:
            se_tx=1; se_ty=1
        else:
            se_tx=E.elec_bloch[0]
            se_ty=E.elec_bloch[1]
        if pivot_start is None:
            piv_str = 'None'
        else:
            piv_str = str(pivot_start)
        # rank=0 is just distributing the energies, makes no computation
        if is_pol == False:
            command =   start+" "+add_command+" calcbulkse"+prog_suffix+" "\
                      " Dir=$PWD Contour=inputE.npy"+\
                      " tol="+str(tol)+" minarea="+str(minarea)+" init_grid_N="+str(init_grid_N)+\
                      " surface_ham="+sHname+" bulk_ham="+E.dir+"/"+E.sl+".TSHS "+\
                      " se_tx="+str(se_tx)+" se_ty="+str(se_ty)+" tx="+str(tx)+" ty="+str(ty)+\
                      " tol_elec_pos="+str(tol_elec_pos)+" bulk_se="+str(bulk_se) + " debug_with_sisl="+str(debug_with_sisl)+\
                      " pivot_start="+piv_str + " use_se_interpolation="+use_se_interp+" "
            if do_periodic:
                assert kxy[0] is None or kxy[1] is None
                command = command.replace('calcbulkse', 'calc_bulk_line_se')
                CL  = []
                nk  = len(kxy[0]) if kxy[1] is None else len(kxy[1])
                assert (max(nk_grid)//2+1 == nk) or (max(nk_grid) == nk)
                for ik in range(nk):
                    cl  = command.split(" ")
                    _kn = str(kxy[1][ik]) if kxy[0] is None  else str(kxy[0][ik])
                    _kx = kxy[0][ik] if (kxy[0] is not None) else None
                    _ky = kxy[1][ik] if (kxy[1] is not None) else None
                    cl += [' kx='+str(_kx)+' ',' ky='+str(_ky)+' ',
                           ' tmp_save_dir=_tmp_se_k_'+str(_kn)+' ', ' ;']
                    CL += cl
                command = " ".join([p for p in CL if (   "minarea"     not in p 
                                                      or "init_grid_N" not in p)])
                
            if reuse==False:
                os.system(command)
        else:
            _Hs = sisl.get_sile(sHname).read_hamiltonian()
            _Hb = E.read_TSHS()
            Hs1 = sisl.Hamiltonian.fromsp(_Hs.geometry, 
                                          _Hs.tocsr(dim=0),
                                          _Hs.tocsr(dim=2),)
            Hb1 = sisl.Hamiltonian.fromsp(_Hb.geometry, 
                                          _Hb.tocsr(dim=0),
                                          _Hb.tocsr(dim=2),)
            nup = 'hsurface_up.TSHS'
            nupb= 'hbulk_up.TSHS'
            Hs1.write(nup)
            Hb1.write(nupb)
            
            command1 =   start+" "+add_command+" calcbulkse "+prog_suffix+" "\
                      " Dir=$PWD Contour=inputE.npy"+\
                      " tol="+str(tol)+" minarea="+str(minarea)+" init_grid_N="+str(init_grid_N)+\
                      " surface_ham="+nup+"  bulk_ham="+nupb+" "+\
                      " se_tx="+str(se_tx)+" se_ty="+str(se_ty)+" tx="+str(tx)+" ty="+str(ty)+\
                      " tol_elec_pos="+str(tol_elec_pos)+" bulk_se="+str(bulk_se) + " debug_with_sisl="+str(debug_with_sisl)+\
                      " pivot_start="+piv_str + " use_se_interpolation="+use_se_interp+" "+\
                      " tmp_save_dir=_tmp_se_up"
            
            Hs1 = sisl.Hamiltonian.fromsp(_Hs.geometry, 
                                          _Hs.tocsr(dim=1),
                                          _Hs.tocsr(dim=2),)
            Hb1 = sisl.Hamiltonian.fromsp(_Hb.geometry, 
                                          _Hb.tocsr(dim=1),
                                          _Hb.tocsr(dim=2),)
            ndn = 'hsurface_dn.TSHS'
            ndnb= 'hbulk_dn.TSHS'
            Hs1.write(ndn)
            Hb1.write(ndnb)
            
            command2 =   start+" "+add_command+" calcbulkse "+prog_suffix+" "\
                      " Dir=$PWD Contour=inputE.npy"+\
                      " tol="+str(tol)+" minarea="+str(minarea)+" init_grid_N="+str(init_grid_N)+\
                      " surface_ham="+ndn+"  bulk_ham="+ndnb+" "+\
                      " se_tx="+str(se_tx)+" se_ty="+str(se_ty)+" tx="+str(tx)+" ty="+str(ty)+\
                      " tol_elec_pos="+str(tol_elec_pos)+" bulk_se="+str(bulk_se) + " debug_with_sisl="+str(debug_with_sisl)+\
                      " pivot_start="+piv_str + " use_se_interpolation="+use_se_interp+" "+\
                      " tmp_save_dir=_tmp_se_dn"
            
            if do_periodic:
                assert kxy[0] is None or kxy[1] is None
                command1 = command1.replace('calcbulkse', 'calc_bulk_line_se')
                command2 = command2.replace('calcbulkse', 'calc_bulk_line_se')
                CL1 = []
                CL2 = []
                nk  = len(kxy[0]) if kxy[1] is None else len(kxy[1])
                assert (max(nk_grid)//2+1 == nk) or (max(nk_grid) == nk)
                for ik in range(nk):
                    cl1  = command1.split(" ")
                    cl2  = command2.split(" ")
                    _kn = str(kxy[1][ik]) if kxy[0] is None  else str(kxy[0][ik])
                    _kx = kxy[0][ik] if (kxy[0] is not None) else None
                    _ky = kxy[1][ik] if (kxy[1] is not None) else None
                    cl1 += [' kx='+str(_kx)+' ',' ky='+str(_ky)+' ',
                           ' tmp_save_dir=_tmp_se_up_k_'+str(_kn)+' ', ' ;']
                    cl2 += [' kx='+str(_kx)+' ',' ky='+str(_ky)+' ',
                           ' tmp_save_dir=_tmp_se_dn_k_'+str(_kn)+' ', ' ;']
                    CL1 += cl1
                    CL2 += cl2
                command1 = " ".join([p for p in CL1 if (   "minarea"     not in p 
                                                       or "init_grid_N" not in p)])
                command2 = " ".join([p for p in CL2 if (   "minarea"     not in p 
                                                       or "init_grid_N" not in p)])
            if reuse==False:
                print('-------------------')
                print('-------------------')
                print('---- Spin-up ------')
                os.system(command1)
                print('-------------------')
                print('-------------------')
                print('---- Spin-dn ------')
                os.system(command2)
        
        ##SE_v = np.zeros((len(Contour), len(Xio), len(Xio)),
        ##                 dtype=np.complex64)
        # print(SE_v.shape)
        ## tmp_files = os.listdir('_tmp_se')
        ## tmp_files.remove('coups_a.npy')
        ## tmp_files.remove('coups.npy')
        ## tmp_files.remove('Htot.TSHS')
        ## tmp_val   = np.array([complex(n.replace('.npz','')) for n in tmp_files])
        ## for i in range(len(Contour)):
        ##     ei = Contour[i]
        ##     try: 
        ##         f  = np.load('_tmp_se/'+str(ei)+'.npz')
        ##     except:
        ##         dist = np.abs(ei - tmp_val)
        ##         if dist.min()<1e-14:
        ##             fname_close = tmp_files[np.where(dist == dist.min())[0][0]]
        ##         else:
        ##             assert 1 == 0
        ##         f = np.load('_tmp_se/'+fname_close)
        ##     SE_v[i,:,:] = f['se_ele']
        ## os.system('rm -rf _tmp_se')
        
        
        if is_pol == False:
            if do_periodic == False:
                folder_name = '_tmp_se/'
            else:
                folder_name = '_tmp_se_k_0.0/'
            try: 
                Htot = sisl.get_sile(folder_name + 'Htot.TSHS').read_hamiltonian()
                Xidx = np.load(folder_name+'coups_a.npy')
            except:
                print('-----------------------------------------------------------')
                print('FAILED GETTING DEFAULT DIRECTORY FOR H AND COUPLINGS!!!!!!')
                print('===> DOING A SELF-SEARCH!!')
                print('-----------------------------------------------------------')
                candidates = [fld for fld in os.listdir() if '_tmp_se_' in fld]
                if len(candidates) == 0:
                    print('SEARCH FAILED')
                    assert 1 == 0
                print('USING '+candidates[0])
                Htot = sisl.get_sile(candidates[0] + '/Htot.TSHS').read_hamiltonian()
                Xidx = np.load(candidates[0]+'/coups_a.npy')
            
            Xio  = np.hstack([np.arange(Htot.a2o(i), Htot.a2o(i+1))
                              for i in Xidx])
            #H0 =  Htot.Hk()[Xio[:, None], Xio[None, :]].toarray()
            #S0 =  Htot.Sk()[Xio[:, None], Xio[None, :]].toarray()
            xyz_elec = Htot.xyz[Xidx]
            s_elec   = Htot.atoms.Z[Xidx].copy()
            print(xyz_elec.shape, )
            np.savez_compressed(self.dir+'/3D_bulk_SE.npz', 
                                ###  SE=SE_v, 
                                #H0=H0, S0=S0, 
                                xyz_elec=xyz_elec, 
                                Xio=Xio, Xidx=Xidx, s_elec = s_elec)
            return Htot.sub(Xidx)
        else:
            ########
            if do_periodic == False:
                Htot1 = sisl.get_sile('_tmp_se_up/Htot.TSHS').read_hamiltonian()
                Xidx1 = np.load('_tmp_se_up/coups_a.npy')
                Htot2 = sisl.get_sile('_tmp_se_dn/Htot.TSHS').read_hamiltonian()
                Xidx2 = np.load('_tmp_se_dn/coups_a.npy')
            else:
                try:
                    Htot1 = sisl.get_sile('_tmp_se_up_k_0.0/Htot.TSHS').read_hamiltonian()
                    Xidx1 = np.load('_tmp_se_up_k_0.0/coups_a.npy')
                    Htot2 = sisl.get_sile('_tmp_se_dn_k_0.0/Htot.TSHS').read_hamiltonian()
                    Xidx2 = np.load('_tmp_se_dn_k_0.0/coups_a.npy')
                except:
                    print('-----------------------------------------------------------')
                    print('FAILED GETTING DEFAULT DIRECTORY FOR H AND COUPLINGS!!!!!!')
                    print('===> DOING A SELF-SEARCH!!')
                    print('-----------------------------------------------------------')
                    candidates1 = [fld for fld in os.listdir() if ('_tmp_se_' in fld and 'up' in fld)]
                    candidates2 = [fld for fld in os.listdir() if ('_tmp_se_' in fld and 'dn' in fld)]
                    if len(candidates1) == 0 or len(candidates2) == 0:
                        print('SEARCH FAILED')
                        assert 1 == 0
                    Htot1 = sisl.get_sile(candidates1[0]+'/Htot.TSHS').read_hamiltonian()
                    Xidx1 = np.load(candidates1[0]+'/coups_a.npy')
                    Htot2 = sisl.get_sile(candidates2[0]+'/Htot.TSHS').read_hamiltonian()
                    Xidx2 = np.load(candidates2[0]+'/coups_a.npy')
            
            Xio1  = np.hstack([np.arange(Htot1.a2o(i), Htot1.a2o(i+1))
                              for i in Xidx1])
            #H01 =  Htot1.Hk()[Xio1[:, None], Xio1[None, :]].toarray()
            #S01 =  Htot1.Sk()[Xio1[:, None], Xio1[None, :]].toarray()
            xyz_elec1 = Htot1.xyz[Xidx1]
            s_elec1   = Htot1.atoms.Z[Xidx1].copy()
            np.savez_compressed(self.dir+'/3D_bulk_SE_up.npz', 
                                #H01=H01, S01=S01, 
                                xyz_elec1=xyz_elec1, 
                                Xio1=Xio1, Xidx1=Xidx1, s_elec1 = s_elec1)
            ######## 
            Xio2  = np.hstack([np.arange(Htot2.a2o(i), Htot2.a2o(i+1))
                              for i in Xidx2])
            #H02 =  Htot2.Hk()[Xio2[:, None], Xio2[None, :]].toarray()
            #S02 =  Htot2.Sk()[Xio2[:, None], Xio2[None, :]].toarray()
            xyz_elec2 = Htot2.xyz[Xidx2]
            s_elec2   = Htot2.atoms.Z[Xidx2].copy()
            np.savez_compressed(self.dir+'/3D_bulk_SE_dn.npz', 
                                #H02=H02, S02=S02, 
                                xyz_elec2=xyz_elec2, 
                                Xio2=Xio2, Xidx2=Xidx2, s_elec2 = s_elec2)
            print(xyz_elec1.shape, )
            return Htot1.sub(Xidx1), Htot2.sub(Xidx2)
        
                
        
        
        
        
    def from_custom(self,H, SEs, Ev, kv, S = None, SE_inds = None, E_F = 0.0, eta = 1e-3, tol = 1e-10, manual_piv = None):
        """
            H: Hamiltonian (nk, no, no) array.
            SEs: list of (nk, nE, no, no) array with self-energies, 
                 one for each electrode. Can also be a list of objects with the
                 self_energy method.
            E  : (nE, ) array with energy points where self-energies have
                 been sampled.
        """
        
        if (Ev.imag==0.0).all():
            E = Ev + 1j * eta
        else:
            E = Ev 
        
        nk, no    = H.shape[0], H.shape[1]
        nE        = len(E)
        n_lead    = len(SEs)
        kw_dict   = {}
        
        for i in range(n_lead):
            if not isinstance(SEs[i],np.ndarray):
                sise   = SEs[i]
                NewSE  = np.zeros((nk,nE,sise.no,sise.no),dtype=complex)
                for ik in range(nk):
                    for ie in range(nE):
                        NewSE[ik,ie] = sise.self_energy(E[ie],k=kv[ik])
                SEs[i] = NewSE
        
        for i in range(len(SEs)):
            kw_dict.update({'SE_'+str(i):SEs[i]})
            if isinstance(SE_inds,list):
                kw_dict.update({'inds_'+str(i):SE_inds[i]})
            else:
                kw_dict.update({'inds_'+str(i):np.arange(no)})
        
        np.savez_compressed(self.dir + '/' +self.sl+'.fakeTBT.SE.npz',
                            **kw_dict,
                            Contour = E,
                            Mode2=1
                            )
        
        if S is None:
            _S = H.copy()
            _S[:,:,:] = 0.0
            for i in range(no):
                _S[:,i,i] = 1.0
        else:
            _S = S
        
        np.savez_compressed(self.dir + '/' +self.sl+'.fakeTSHS.npz',
                            H =  H,
                            S = _S,
                            k = kv)
        # nk, nE, no, no
        G  = np.zeros((nk,nE, no,no),dtype=complex)
        G += _S[:,None,:,:] * E[None,:,None,None] - H[:,None,:,:]
        for i in range(len(SEs)):
            idx = np.array(kw_dict['inds_'+str(i)])
            G[...,idx[:, None],idx[None,:]] -= SEs[i]
        G = np.linalg.inv(G)
        Tij0 = np.zeros((nk,nE, n_lead, n_lead),dtype=complex)
        for i in range(len(SEs)):
            Gam0i  = 1j*(SEs[i] - SEs[i].conj().transpose(0,1,3,2))
            idxi   = kw_dict['inds_'+str(i)]
            for j in range(i+1,len(SEs)):
                Gam0j = 1j*(SEs[j] - SEs[j].conj().transpose(0,1,3,2))
                idxj  = kw_dict['inds_'+str(j)]
                # Trace of Product.
                M1       = Gam0i@(G[:,:, idxi,:][...,idxj])
                M2       = Gam0j@(G.conj().transpose(0,1,3,2)[:,:,idxj,:][...,idxi])
                Tij0[:,:,i,j] = (M1*(M2.transpose(0,1,3,2))).sum(axis=(2,3))
        
        transmission =  Tij0[:,:,0,1].sum(axis=0)
        # Pivotting
        sortmat = csr_matrix((no,no))
        iH,jH   = np.where((np.abs(H)>tol).any(axis=0))
        sortmat[iH, jH] = 1.0
        for i in range(len(SEs)):
            idx = np.array(kw_dict['inds_'+str(i)])
            sortmat[idx[:,None],idx[None,:]] = 1.0
        if manual_piv is None:
            piv,btd,part = _BTDP(sortmat, no//10)
        else:
            piv,btd,part = _BTDP(sortmat, manual_piv)
            
        np.savez_compressed(self.dir + '/' +self.sl+'.fakeTBT.npz',
                            transmission = transmission,
                            real_pivot   = np.nan,
                            real_btd     = np.nan,
                            pivot        = piv,
                            btd          = btd,
                            E            = E.real,
                            _tbtTk_full  = Tij0,
                            kv           = kv,
                            wkpt         = np.ones(len(kv))/len(kv),
                            E_F          = E_F,
                            sortmat      = sortmat.todense()
                            )
    
    def find_polygons(self, exclude_edge=True,tile=[1,5,1], 
                      bulk_connections = 3, NN_dist = 1.7, 
                      dist_from_edge=3.0, start_direc = np.array([0,1]),
                      sign = 1):
        print('Only works for 2D materials!')
        from time import sleep
        import matplotlib.pyplot as plt
        
        na = len(self.pos_real_space)
        nuc = np.prod(tile)
        RSC = np.zeros((na * nuc, 2))
        R   = self.pos_real_space[:,[0,1]]
        UC_vec = np.zeros((nuc, 2))
        it  = 0
        for ix in range(-(tile[0]//2),tile[0]//2+1):
            for iy in range(-(tile[1]//2),tile[1]//2+1):
                for iz in range(-(tile[2]//2),tile[2]//2+1):
                    T          = self.lat[0,[0,1]]*ix + self.lat[1,[0,1]]*iy + self.lat[2,[0,1]]*iz
                    RSC[it*na:(it+1)*na] = R + T
                    UC_vec[it] = T 
                    it+=1
        del it
        
        Dij = np.linalg.norm(R[:,None,:] - RSC[None,:,:], axis=2)
        #Indices with atoms that has the bulk number of nearest neighbors
        IDX_b  = np.where(((Dij<NN_dist) * (Dij>0.1)).sum(axis=1)>=bulk_connections)[0]
        IDX_e  = np.where(((Dij<NN_dist) * (Dij>0.1)).sum(axis=1)< bulk_connections)[0]
        
        RI, RE = R[IDX_b], R[IDX_e]
        Dij    = np.linalg.norm(RI[:,None,:] - RE[None,:,:],axis=2)
        IDXII  = np.where((Dij>dist_from_edge).all(axis=1))
        RI     = RI[IDXII]
        deg    = 90 * np.pi/180
        rot90  = np.array([[np.cos(deg), np.sin(deg)],
                           [-np.sin(deg), np.cos(deg)]])
        polys  = []
        #plt.scatter(RSC[:,0],RSC[:,1])
        #plt.scatter(RI[:,0],RI[:,1] )
        #plt.scatter(R[:,0], R[:,1]+0.4, marker='*',color='k')
        
        for i in range(len(RI)):
            ri    = RI[i].copy()
            dij   = np.linalg.norm(ri - RSC, axis=1)
            idxNN = np.where((dij < NN_dist) * (dij>0.1))[0]
            
            dR    = RSC[idxNN] - ri
            
            dots  = dR @ start_direc
            idx   = np.where(dots == dots.max())[0][0]
            bondR = dR[idx] #- ri
            r     = ri + bondR
            poly_points = [ r.copy()]
            its = 0
            poly_dist = np.array([[1]])
            while (poly_dist>0.1).all():
                dij    =  np.linalg.norm(r - RSC, axis=1)
                idxNN  =  np.where((dij < NN_dist) * (dij>0.1))[0]
                dR     =  RSC[idxNN] - r
                
                nR     =  np.linalg.norm(dR,axis=1)
                angles =  np.arccos((dR@bondR)/( nR *np.linalg.norm(bondR)))
                angles[np.isnan(angles)] = np.pi
                sort   =  np.argsort(angles)
                Right  =  sign*rot90 @ bondR
                dots   =  dR@Right
                idx    =  np.where(dots==dots.max())[0][0]
                
                bondR         = dR[idx]
                r            += bondR
                poly_points  += [r.copy()]
                arr           = np.array(poly_points)
                poly_dist     = np.linalg.norm(arr[:, None,:] - arr[None,:,:],axis=2)
                idx           = np.arange(len(poly_dist))
                poly_dist[idx,idx]+=100.0
                its+=1
            polys += [np.array(poly_points)[:-1].copy()]
        
        return polys
    
            
               
            
        
            
            
            
            
        
        
        
        
        
        
        
        
        
    

        
    
    
    
    # def use_custom_tbtgf(self):
    #     self.elec_RSSE = not self.elec_RSSE
    #     print('set elec_RSSE variable to ', self.elec_RSSE)
        

    
    

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def load_calculation(Devdir,  old_basis='SZ'):
    from funcs import read_electrode
    e = read_electrode(Devdir + '/TS_TBT.fdf')
    elecs = []
    for ed in e:
        l,r,s = np.random.random((3,3)), np.random.random((10,3)), np.arange(10)+1
        elecs += [SiP(l, r, s, 
                      directory_name = ed, 
                      overwrite = 'reuse',
                      basis = old_basis,
                      )
                 ]
    
    Dev = SiP(l, r, s, 
              directory_name = Devdir, 
              overwrite = 'reuse',
              basis = old_basis,
              elecs = elecs
              )
    return Dev, elecs

class sisl_replica_for_projection:
    def __init__(self, sip):
        self.geom = sip.to_sisl(what = 'geom')
        self.SiP  = sip

class dftb_charge:
    def __init__(self, file, atoms, norbs):
        self.file = file
        self.norbs = norbs
        self.atoms = atoms
        self.createa2o()
        self.dic = {}
    def a2o(self,ia,lo):
        orbs = self.norbs[:ia].sum()
        return orbs + lo
    def createa2o(self):
        array = np.ones((len(self.atoms), self.norbs.max()),dtype=int)*(-1)# * np.nan
        no_tot= self.norbs.sum()
        it = 0
        for ia in range(len(self.atoms)):
            for lo in range(self.norbs[ia]):
                array[ia, lo] = it
                it+=1
        assert no_tot == it
        self.iaio2i = array.astype(int)
    def set_iaio2i(self,idx):
        self.iaio2i = idx
    def set_Q(self, Qv, idx = None):
        Qtot  = Qv.sum()
        f     = open(self.file,'r')
        li    = f.readlines()
        f.close()
        line  = ''+li[1]
        split = line.split()
        split[-1] = '  '+str(Qtot)+'\n'
        li[1] ='  '.join(split)
        if idx is None:
            idx = self.iaio2i
        shift = 2
        for io in range(len(Qv)):
            ia,lo = np.where(idx == io)
            ia    = ia[0]; lo = lo[0]
            q     = Qv[io]
            if q<0.0 or np.isnan(q):
                continue
            line  = '' + li[ia + shift]
            split = line.split()
            split[lo] = str(q)[0:15]
            li[ia + shift] = ' '.join(split) + '\n'
        f = open(self.file,'w')
        for l in li:
            f.write(l)
        f.close()
    def read_Q(self, idx=None, label = None):
        if idx is None:
            idx = self.iaio2i
        li    = open(self.file,'r').readlines()
        Qv    = np.zeros(idx.max()+1)
        for ia, line in enumerate(li[2:-1]):
            split = line.split()
            Qs    = [float(split[i]) for i in range(len(split))]
            Qv[idx[ia, 0:len(Qs)]] = Qs
        if label is not None:
            self.dic.update({label:Qv})
        return Qv
    
            
            
        
        
    
    
    
