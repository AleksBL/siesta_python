import numpy as np
import os
import sisl
import spglib as spg
import seekpath
from time import time
#import PyGnuplot
from funcs import unique_list, Num2Sym,read_analyze,write_fdf,write_gin, read_geom_from_fdf
from funcs import read_gulp_results,read_total_energy, write_relax_fdf, read_siesta_relax, read_mulliken
#from Build_eps_from_planes import Structure

terminal = os.system; rem = os.remove; ld = os.listdir
wd = os.getcwd()
PP_path = '../pp'
#put siesta and tbtrans in path....#
M = 'mpirun '
sp = 'siesta '
tb_trans_exec='tbtrans '
B = 'gnubands '
pdos_exec='fmpdos'
gulp_path='gulp '
gen_basis='gen-basis '

if os.uname().nodename == 'aleks-ideapad':
    sp = '/home/aleks/siesta-master/Obj/'+sp
    tb_trans_exec = '/home/aleks/siesta-master/Util/TS/TBtrans/'+tb_trans_exec
    B = '/home/aleks/siesta-master/Util/Bands/'+B
    pdos_exec='/home/aleks/siesta-master/Util/Contrib/APostnikov/'+pdos_exec
    gulp_path='/home/aleks/gulp-5.2/Src/'+gulp_path
    gen_basis='/home/aleks/siesta-master/Util/Gen-basis/'+gen_basis

elif os.uname().nodename == 'indsæt_pc_navn_og_paths_hvis ikke 15-21 passer':
    ########### PATHs ###############
    pass

def error():
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
    def __init__(self,lat,pos,s,
                 directory_name='noname',
                 sm = 'siesta',
                 sl = 'siesta',
                 spin_pol = 'F',
                 energy_shift = None,
                 dm_mixing_weight = None,
                 number_pulay = None,
                 dm_tol = '1.d-4',
                 xc = 'gga',
                 basis = 'SZP',
                 mesh_cutoff = 150,
                 max_scf_it  = 300, 
                 kgridcutoff = None,
                 solution_method = 'diagon',
                 electronic_temperature_mev = 25,
                 calc_bands = False,
                 kp = None,
                 k_shift=0.0,
                 TwoDim = True,
                 write_matrices=True,
                 pp_path = PP_path,
                 wd = wd,
                 siesta_exec=sp,
                 tb_trans_exec=tb_trans_exec,
                 gen_basis = gen_basis,
                 mpi=M,
                 PDOS_EXEC=pdos_exec,
                 Gulp_exec=gulp_path,
                 Bands_plot = B,
                 Standardize_cell=False,
                 elec_inds = None,
                 print_console=False,
                 Voltage = None,
                 Chem_Pot = None,
                 semi_inf = None,
                 elecs = [],
                 buffer_atoms=[],
                 kp_tbtrans = None,
                 save_EP = False,
                 trans_emin=-1,
                 trans_emax= 1,
                 trans_delta=0.02,
                 NEGF_calc=False,
                 reuse_dm=False,
                 set_pdos=None,
                 save_es=False,
                 overwrite=False,
                 mixer='Pulay',
                 n_hist=8,
                 elec_bloch = None,
                 mix_what = 'Hamiltonian',
                 elec_RSSE = False
                 ):
        
        assert pos.shape[0] == len(s)
        Reuse=False
        for i in ld(wd):
            if directory_name==i:
                print('Directory called '+i+' already exists! Remove it? (y/reuse/n)')
                if overwrite==True:
                    inp='y'
                elif overwrite == 'reuse':
                    inp = 'reuse'
                else:
                    inp=input()
                if inp=='y':
                    for j in ld(wd+'/'+i):
                        name=wd+'/'+i+'/'+j
                        print('file '+name+' removed!\n')
                        if '.py' in name or j in no_rem:
                            pass
                        else:
                            if j == 'orbital_directory':
                                for k in ld(name):
                                    print('file '+name+k+' removed!\n')
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
        if Reuse == True:
            try:
                p,lat = read_geom_from_fdf(directory_name + '/' + 'STRUCT.fdf')
                pos = p[:,0:3]
                s   = p[:,  3]
            except:
                print('Couldnt read STRUCT.fdf, stopping')
                assert 1 == 0
            if 'siesta.TBT.nc' in ld(directory_name):
                os.system('cp ' + directory_name + '/siesta.TBT.nc '  + directory_name + '/old_siesta.TBT.nc ')
        
        ###### Initialize all the stuff neeeded to write a siesta .fdf file. 
        self.dir=directory_name
        self.lat   = np.array(lat)
        self.s     = s
        self.sname =[Num2Sym[i] for i in s]
        pos   = np.array(pos)
        self.xc    = xc
        A_mat=self.lat.T
        ### lattice vectors now in columns
        A_mat_inv=np.linalg.inv(A_mat)
        positions=[]
        for i in range(len(pos[:,0])):
            #[a1,a2,a3]^-1 * p  = p_frac
            fpos_i=np.dot(A_mat_inv,np.asarray([pos[i,0],pos[i,1],pos[i,2]]))
            positions=positions + [[float(fpos_i[0]),float(fpos_i[1]),float(fpos_i[2])]]
        self.pos = np.array(positions)
        self.pos_real_space=np.zeros(self.pos.shape)
        for i in range(len(self.pos)):
            #p=[a1,a2,a3] * p_frac
            self.pos_real_space[i,:] = A_mat.dot(self.pos[i,:])
        self.cell_spg = (self.lat,self.pos,s)
        try:
            self.get_path = seekpath.get_path(self.cell_spg,True)
        except:
            self.get_path = None
        try:
            self.sym_dic = self.get_path['point_coords']
        except:
            self.sym_dic=None
        try:
            self.k_path = seekpath.get_path(self.cell_spg,True)['path']
        except:
            self.k_path = None 
        try:
            self.sym=spg.get_spacegroup(self.cell_spg,symprec=1e-3)
        except:
            self.sym=None
        try:
            self.standard_cell = spg.standardize_cell(self.cell_spg)
        except:
            self.standard_cell = None
        
        self.sm = sm
        self.sl = sl
        self.spin_pol = spin_pol
        self.energy_shift = energy_shift
        self.dm_mixing_weight = dm_mixing_weight
        self.number_pulay = number_pulay
        self.dm_tol = dm_tol
        self.xc = xc
        self.basis = basis
        self.mesh_cutoff = mesh_cutoff
        self.max_scf_it = max_scf_it
        self.min_scf_it = 3
        self.kgridcutoff = kgridcutoff
        self.solution_method = solution_method
        self.electronic_temperature_mev = electronic_temperature_mev
        self.calc_bands = calc_bands
        self.kp = kp
        self.k_shift = k_shift
        self.TwoDim = TwoDim
        if Standardize_cell==True:
            self.standardise_cell()
        self.write_matrices=write_matrices
        self.elecs=elecs
        self.pp_path=pp_path
        self.siesta_exec =siesta_exec
        self.pdos_exec=PDOS_EXEC
        self.gulp_exec=Gulp_exec
        self.bands_plot=Bands_plot
        self.wd = wd
        self.elec_inds = elec_inds
        self.mpi=mpi
        if print_console == True:
            self.print_no_print='|tee'
        else:
            self.print_no_print='>'
        self.Voltage = Voltage
        self.Chem_Pot=Chem_Pot
        self.semi_inf = semi_inf
        self.BTD_ALGO=None
        self.tb_trans_exec=tb_trans_exec
        self.gen_basis_exec = gen_basis
        self.kp_tbtrans=kp_tbtrans
        self.save_EP=save_EP
        self.trans_emin=trans_emin
        self.trans_emax=trans_emax
        self.trans_delta=trans_delta
        self.NEGF_calc=NEGF_calc
        self.reuse_dm=reuse_dm
        self.dQ=None
        self.set_pdos=set_pdos
        self.contours=[]
        self.save_electrostatic_potential=save_es
        self.buffer_atoms=buffer_atoms
        self.mixer=mixer
        self.n_hist=n_hist
        self.elec_bloch = elec_bloch
        self.mix_what = mix_what
        self.elec_RSSE = elec_RSSE
        self.Check_Distances()
        
        print('self.pos is reduced coordinates, self.pos_real_space is actual coordinates!\n')
        
        non_ortho=False
        for i in range(3):
            for j in range(3):
                dot=self.lat[i,:].dot(self.lat[j,:])
                if i!=j and np.abs(dot)>1e-10:
                    non_ortho = True
        self.non_ortho=non_ortho
        
        if TwoDim==True and kp is not None and print_console and self.sym is not None:
            print('2-D crystal with space group ' + self.sym)
    
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
        self.buffer_atoms=inds_buffer
    
    def fdf(self, eta = 1e-2):
        write_fdf(self, eta = eta)
        if self.Voltage is None:
            pass
        elif self.Voltage!=0:#  or self.non_ortho==True:
            self.write_more_fdf(['TS.Hartree.Fix -A'])
        self.get_pseudo_paths()
    
    def fdf_relax(self, Constraints, force_tol = 0.01, max_it = 1000):
        write_relax_fdf(self, Constraints = Constraints, force_tol = force_tol, max_it = max_it)
        self.get_pseudo_paths()
    
    def read_relax(self):
        p,l = read_siesta_relax(self.dir + '/RUN.out')
        self.relaxed_pos = p
        self.relaxed_lat = l
    
    def gin(self,cell_opti=[],f=None, var_cell = None, relax_only = False):
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
        
        write_gin(self,cell_opti=cell_opti,fix=fix, relax_only = relax_only )
    
    def get_pseudo_paths(self):
        paths=[]
        atoms = unique_list(self.s)
        Dir=self.pp_path
        PPs=ld(Dir)
        for atom in atoms:
            for pp in PPs:
                name=''
                for let in pp:
                    if let=='.':
                        break
                    else:
                        name+=let
                if Num2Sym[atom] == name and self.xc in pp:
                    paths+=['../'+self.pp_path+'/'+pp]
        self.pseudo_paths = paths
        it=0
        for p in paths:
            os.system('ln -s '+ p +' '+self.dir+'/'+ Num2Sym[atoms[it]]+'.psf')
            it+=1
    
    def add_elecs(self,L):
        self.elecs = L
    
    def find_elec_inds(self, tol = 1e-2):
        inds = [ [] for e in self.elecs ]
        it_e = 0
        for e in self.elecs:
            if e.elec_RSSE == False:
                for i in range(len(e.pos_real_space)):
                    ri = e.pos_real_space[i]
                    for j in range(len(self.pos_real_space)):
                        rj = self.pos_real_space[j]
                        if np.linalg.norm(ri-rj) < tol:
                            inds[it_e]+=[j]
            
            if e.elec_RSSE == True:
                R,S = e.get_RS_pos()
                for i in range(len(R)):
                    ri = R[i]
                    for j in range(len(self.pos_real_space)):
                        rj = self.pos_real_space[j]
                        if np.linalg.norm(ri-rj) < tol:
                            inds[it_e]+=[j]
            
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
        self.s = self.s[idx]
        
        
        if self.elec_inds is None:
            self.elec_inds = inds_new
        else:
            print('elec inds already given! remove these if you want to use this function\n')
        
    
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
        with open(self.dir+'/'+name+'.fdf','a') as f:
            f.write('\n')
            for l in L:
                f.write(l)
            f.write('\n')
            f.close()
    
    def write_analysis_results(self):
        WRITE=read_analyze(self)
        self.write_more_fdf(WRITE,name = 'TS_TBT')
    
    def run_siesta_in_dir(self):
        if self.solution_method=='diagon':
            print('Running Siesta calculation in Directory: '+self.dir+ '!\n')
        elif self.solution_method=='transiesta':
            print('Running TranSiesta calculation in Directory: '+self.dir+ '!\n')
        
        n='RUN'     #self.sl
        os.chdir(self.dir)
        os.system(self.mpi+self.siesta_exec+'<'+n+'.fdf '+self.print_no_print+' '+n+'.out')
        os.chdir('..')
    
    def run_gulp_in_dir(self, mpi_run =False):
        print('Running Gulp calculation in Directory: '+self.dir+ '!\n')
        os.chdir(self.dir)
        if mpi_run == True:
            os.system(self.mpi + self.gulp_exec+'< Gulp.gin > Gulp.gout')
        else:
            os.system(self.gulp_exec+'< Gulp.gin > Gulp.gout')
        
        os.chdir('..')
        self.gulp_results=read_gulp_results(self)
    
    def run_analyze_in_dir(self):
        print('Running Siesta-analyse in Directory:'+self.dir+ '!\n')
        n='RUN'       #self.sl
        if len(self.elecs)==0:
            print('why are you running analyze on something that has no electrodes attached?')
        os.chdir(self.dir)
        os.system(self.siesta_exec+'-fdf TS.Analyze '+n+'.fdf '+self.print_no_print+' '+n+'_Analyze.out' )
        os.chdir('..')
        self.write_analysis_results()
    
    def run_siesta_electrode_in_dir(self):
        print('Running Siesta electrode calculation in Directory: '+self.dir+ '!\n')
        n='RUN'      #self.sl
        os.chdir(self.dir)
        os.system(self.mpi+self.siesta_exec+'--electrode <'+n+'.fdf '+self.print_no_print+' '+n+'.out')
        os.chdir('..')
    
    def plot_calculated_bands(self):
        os.chdir(self.dir)
        os.system(self.bands_plot+' <'+self.sl+'.bands '+self.print_no_print+' gnubands.f')
        # PyGnuplot.c('plot "gnubands.f"')
        os.chdir('..')
    
    def run_tbtrans_dir(self,DOS_GF= False, DOS_A=False, DOS_A_ALL=False,ORBITAL_CURRENT=False, Custom = [] ):
        print('Running TB-Trans in Directory: '+self.dir+ '!\n')
        self.write_tb_trans_kp()
        n = 'RUN'         #self.sl
        List = ['\n']
        if DOS_A:
            List+=['TBT.DOS.A T\n']
        if DOS_GF:
            List+=['TBT.DOS.Gf T\n']
        if DOS_GF:
            List+=['TBT.DOS.A.All T\n']
        if ORBITAL_CURRENT:
            List+=['TBT.Current.Orb T\n']
        if len(Custom)!=0:
            for C in Custom:
                List+=[C]
        
        
        self.write_more_fdf(List,name='TS_TBT')
        os.chdir(self.dir)
        os.system(self.mpi+self.tb_trans_exec + n+'_TBT.fdf '+self.print_no_print+' '+n+'TBTCalc.out')
        os.chdir('..')
    
    def system_energy(self):
        return read_total_energy(self)
    
    def relaxed_system_energy(self):
        import subprocess
        command = 'grep \'siesta: E_KS(eV)\' ' +self.dir+'/RUN.out'
        res = subprocess.check_output(command, shell=True)
        return float(res[-18:-2])
    
    def dQ(self):
        dQ=None
        with open(self.dir+'/'+self.sl+'.out','r') as f:
            for l in f:
                if 'Excess charge' in l:
                    dQ=float(l[37:])
                    break
        self.dQ=dQ
    
    def Add_Bounded_Plane(self, O, A, B,  Potential, cut_off, spread = 100.0):
        self.write_more_fdf(['%block Geometry.Hartree\n',
                            ' Bounded plane ' + str(Potential)  + ' eV \n',
                            '  gauss '+str(spread) + ' ' +    str(cut_off  )  + ' Ang\n',
                            '  ' + str(O[0]) + ' ' + str(O[1]) + ' ' + str(O[2]) + ' Ang\n',
                            '  ' + str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]) + ' Ang\n',
                            '  ' + str(B[0]) + ' ' + str(B[1]) + ' ' + str(B[2]) + ' Ang\n',
                            '%endblock Geometry.Hartree\n  ' 
                            ],
                            name = 'STRUCT')
    
    def Add_Charged_Bounded_Plane(self, O, A, B,  Charge, cut_off, spread, kind = 'gauss'):
        self.write_more_fdf(['%block Geometry.Charge\n',
                            ' Bounded plane ' + str(Charge)  + ' \n',
                            '  '+ kind +' ' + str(spread) + ' ' +    str(cut_off)  + ' Ang\n',
                            '  ' + str(O[0]) + ' ' + str(O[1]) + ' ' + str(O[2]) + ' Ang\n',
                            '  ' + str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]) + ' Ang\n',
                            '  ' + str(B[0]) + ' ' + str(B[1]) + ' ' + str(B[2]) + ' Ang\n',
                            '%endblock Geometry.Charge\n  ' 
                            ],
                            name = 'STRUCT')
    
    def Add_Charged_Box(self,O, A, B, C, Charge):
        self.write_more_fdf(['%block Geometry.Charge\n',
                            ' box ' +  str(Charge) + ' \n',
                            '    delta\n'
                            '  ' + str(O[0]) + ' ' + str(O[1]) + ' ' + str(O[2]) + ' Ang\n',
                            '  ' + str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]) + ' Ang\n',
                            '  ' + str(B[0]) + ' ' + str(B[1]) + ' ' + str(B[2]) + ' Ang\n',
                            '  ' + str(C[0]) + ' ' + str(C[1]) + ' ' + str(C[2]) + ' Ang\n',
                            '%endblock Geometry.Charge\n  ' 
                            ],
                            name = 'STRUCT')
    
    def Real_space_SE(self, ax_decimation, ax_integrate,  supercell, eta,Emin, Emax, dE):
        #
        #Straight out of sisl tutorial TB8
        #
        if self.elec_RSSE == False:
            print('set elec_RSSE to True!')
            assert 1 == 0
        
        import tqdm
        E = np.arange(Emin, Emax + dE / 2, dE)
        H_minimal = sisl.get_sile(self.dir + '/' + self.sl + '.TSHS').read_hamiltonian()
        print(H_minimal.toASE().numbers)
        RSSE = sisl.RealSpaceSE(H_minimal, ax_decimation, ax_integrate, supercell)
        H_elec, elec_indices = RSSE.real_space_coupling(ret_indices=True)
        H_elec.write(self.dir + '/'+ self.sl + '.TSHS')
        H = RSSE.real_space_parent()
        # Create the truedevice by re-arranging the atoms
        indices = np.arange(len(H))
        indices = np.delete(indices, elec_indices)
        indices = np.concatenate([elec_indices, indices])
        np.save(self.dir + '/RS_Coupling_pos', H.xyz[elec_indices])
        np.save(self.dir + '/RS_Coupling_specie', H.toASE().numbers[elec_indices])
        eta = eta * 1j
        gamma = sisl.MonkhorstPack(H_elec, [1] * 3)
        sisl.io.tableSile(self.dir + '/contour.E', 'w').write_data(E, np.zeros(E.size) + dE)
        
        with sisl.io.tbtgfSileTBtrans(self.dir +'/' + self.sl +'.TBTGF') as f:
            f.write_header(gamma, E + eta)
            for ispin, new_k, k, e in tqdm.tqdm(f):
                if new_k:
                    f.write_hamiltonian(H_elec.Hk(format='array', dtype=np.complex128))
                SeHSE = RSSE.self_energy(e + eta, bulk=True, coupling=True)
                f.write_self_energy(SeHSE)
        self.RSSE_Energy_from_to = (Emin, Emax)
        
        
        
        # # Now re-arange matrix
        # H = H.sub(indices)
    
    
    # def Get_Mulliken(self):
    # def Add_Spheres(self, Radii, Cs):
    #     Write = []
    #     n = len(Radii)
    #     Write+=['%block Geometry.Hartree']
        
    #     for R, i in enumerate(Radii):
    #         c = Cs[i]
    #         W
            
    def get_RS_pos(self):
        try:
            p_rs = np.load(self.dir + '/RS_Coupling_pos.npy')
            s_rs = np.load(self.dir + '/RS_Coupling_specie.npy')
            return p_rs, s_rs
        except: 
            return None, None
    
    def save_DM(self):
        fn2 = self.sl+'.TSDE'
        os.system('cp '+self.dir+'/'+fn2+ ' Save_DM/'+self.dir+'_'+fn2)
    
    def save_TSHS(self):
        fn2 = self.sl+'.TSHS'
        os.system('cp '+self.dir+'/'+fn2+ ' Save_HS/'+self.dir+'_'+fn2)
    
    def save_TBT(self):
        fn2 = self.sl+'.TBT.nc'
        os.system('cp '+self.dir+'/'+fn2+ ' Save_TBT_nc/'+self.dir+'_'+fn2)
    
    def save_ES(self):
        os.system('cp '+self.dir+'/'+'ElectrostaticPotential.grid.nc ' + 'Save_ES/ElectrostaticPotential'+self.sl+self.dir+'_'+'.grid.nc')
    
    def save_TP(self):
        os.system('cp '+self.dir+'/'+'TotalPotential.grid.nc ' + 'Save_TP/TotalPotential'+self.sl+self.dir+'_'+'.grid.nc')
    
    def load_saved_DM(self,saved_name, new_name):
        print('Loaded '+saved_name +' as initial guess for density matrix\n')
        os.system('cp Save_DM/'+ saved_name +' '+self.dir+'/'+ new_name)
    
    def load_saved_HS(self,saved_name, new_name):
        print('Loaded '+saved_name +' as initial guess for Hamiltonian\n')
        os.system('cp Save_HS/'+ saved_name +' '+self.dir+'/'+ new_name)
    
    def copy_DM_from(self, object):
        os.system('cp ' + object.dir + '/' + object.sl+'.TSDE ' + self.dir + '/' + self.sl + '.TSDE')
    
    def save_DM_TSHS(self):
        self.save_DM()
        self.save_TSHS()
    
    def save_TBT_ES_TP(self):
        self.save_TBT()
        self.save_ES()
        self.save_TP()
    
    def write_tb_trans_kp(self):
        if self.kp_tbtrans == None:
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
                    else: f.write(l)
                f.close()
            
            # with open(self.dir+'/'+name+'_TBT.fdf','w') as f:
            #     write_k = 0; it_k=0
            #     for l in lines:
            #         if '%endblock kgrid.MonkhorstPack' in l: write_k = 0
            #         if write_k == 1 and it_k<3:
            #             if it_k==0:
            #                 f.write('    '+str(self.kp_tbtrans[0]) + '   '+str(0)+'   '+str(0)+' '+str(self.k_shift)+'\n')
            #             if it_k==1:
            #                 f.write('    '+str(0) + '   '+str(self.kp_tbtrans[1])+'   '+str(0)+' '+str(self.k_shift)+'\n')
            #             if it_k==2:
            #                 f.write('    '+str(0) + '   '+str(0)+'   '+str(self.kp_tbtrans[2])+' '+str(self.k_shift)+'\n')
            #             it_k+=1
            #         else: f.write(l)      
            #         if '%block kgrid.MonkhorstPack' in l: write_k = 1
            #     f.close()
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
            if ((d<min_dist).sum())>1:
                print('atoms too close bruh')
                error()
                
        
        
        
    
    # def Wrap_unit_cell(self,shrink=1e-5,PRINT=False,strict_z=False):
    #     a1=self.lat[0,:].copy()
    #     a2=self.lat[1,:].copy()
    #     a3=self.lat[2,:].copy()
    #     a1-= shrink * a1/np.linalg.norm(a1)
    #     a2-= shrink * a2/np.linalg.norm(a2)
    #     a3-= shrink * a3/np.linalg.norm(a3)
        
    #     #zero = -shrink * a1/np.linalg.norm(a1) - shrink * a2/np.linalg.norm(a2)-shrink * a3/np.linalg.norm(a3)
    #     if strict_z==False: 
    #         zero=np.array([0,0,-shrink])
    #         a1 += zero
    #         a2 += zero
    #         a3 += zero
    #     else: zero = np.zeros(3)
    #     points  = [zero,a1,a1+a2,a2,zero+a3,a1+a3,a1+a2+a3,a2+a3]
    #     faces = [[0,1,2,3][::-1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    #     Cell = Structure(points,faces,convex = True)
    #     P_mid = (a1+a2+a3)/2
    #     Try   = Cell.Inside_Struc_Convex(P_mid,full_return=True)
    #     if PRINT==True:
    #         print(Try, Try.shape)
    #     it=0
    #     for x in Try[0,:]:
    #         if x > 0: 
    #             faces[it] = faces[it][::-1]
    #             if PRINT==True:
    #                 print('face ' +str(it)+' reversed')
    #         it+=1
    #     Cell = Structure(points,faces,convex = True)
    #     Try2 = Cell.Inside_Struc_Convex(P_mid)
    #     if PRINT==True:
    #         print(Try2)
    #     Truth = Cell.Inside_Struc_Convex(self.pos_real_space)
    #     T=[]
    #     for i in range(-1,2):
    #         for j in range(-1,2):
    #             if i!=j:
    #                 T+=[[i,j]]
        
    #     for i in range(len(Truth)):
    #         pi=self.pos_real_space[i,:]
    #         if Truth[i] == False:
    #             print('lol')
    #             Break=False
    #             for t in T:
    #                 if Cell.Inside_Struc_Convex(pi+self.lat[0,:]*t[0]+self.lat[1,:]*t[1]) and Break==False:
    #                     self.pos_real_space[i,:] = pi+self.lat[0,:]*t[0]+self.lat[1,:]*t[1]
    #                     Break=True
    #                     print('Atom '+str(i)+' Wrapped inside unit-cell with lattice-vector-combination '+str(t[0])+','+str(t[1])+'\n')
    
    def Visualise(self,T=[[0,0]],axes=[0,1]):
        import matplotlib.pyplot as plt
        color=np.zeros((len(self.pos_real_space),3))
        
        colors = [[0,0,1], [1,0,0],[0,1,0], [0.5,0.5],[0.5,0,0.5]]
        
        
        for i,pi in enumerate(self.pos_real_space):
            if i in self.elec_inds[0]:
                color[i,:] = np.array([0,0.5,1])
            elif i in self.elec_inds[1]:
                color[i,:] = np.array([0,0.5,1])
            elif self.buffer_atoms != []:
                if self.buffer_atoms[0]-1<=i<=self.buffer_atoms[1]:
                    color[i,:] = np.array([0.5,0,1])
            
            else:
                us = unique_list(self.s)
                color[i,:] = np.array(colors[us.index(self.s[i])])
        
        for t in T:
            Tv = self.lat[0,:]*t[0]+self.lat[1,:]*t[1]
            r_plot=self.pos_real_space+Tv
            plt.scatter(r_plot[:,axes[0]],r_plot[:,axes[1]],c=color)
        plt.axis('equal')
    
    
