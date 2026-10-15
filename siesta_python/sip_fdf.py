"""Free functions extracted from class SiP (input-file writers).

Bodies are copied verbatim from siesta_python.py with `self` renamed to `sip`.
Keep them in sync: the class methods are thin wrappers that call these.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from funcs import (barebones_RUN, Num2Sym, unique_list,
                   write_fdf, write_relax_fdf, write_gin, write_dftb_hsd)
from sip_shared import ld, ossystem



def write_more_fdf(sip, L,name='DEFAULT'):
    """
        After self.fdf() has been called, you can add more to the
        fdf file called name afterwards with this function.

        L is a list of string like ['TBT.DOS Gf True \n', ...]
    """
    with open(sip.dir+'/'+name+'.fdf','a') as f:
        f.write('\n')
        for l in L:
            f.write(l)
        f.write('\n')
        f.close()


def delete_fdf(sip,name):
    os.chdir(sip.dir)
    ossystem('rm ' + name  + '.fdf')
    os.chdir('..')


def set_parallel_k(sip):
    """ Toggles DiagK siesta option"""
    if len(sip.elecs) == 0:
        sip.write_more_fdf(['Diag.ParallelOverK True\n'])


def E_field(sip,E):
    sip.write_more_fdf(['%block ExternalElectricField\n',
                          str(E)[1:-1]+' V/Ang\n',
                         '%endblock ExternalElectricField\n'],
                          name = 'DEFAULT')


def set_methfessel_paxton(sip, N):
    lines = ['OccupationFunction MP\n',
             'OccupationMPOrder '+ str(N) + '\n']
    sip.write_more_fdf(lines, name = 'DEFAULT')


def fatband_fdf(sip):
    """ Fatbands """
    sip.write_more_fdf(['WFS.Write.For.Bands True\n',
                         'COOP.Write True\n'])


def custom_bandlines(sip, path):
    sip.delete_fdf('KP')
    sip.write_more_fdf(['%block kgrid.MonkhorstPack\n',
                          str(sip.kp[0])+'  0  0  0.0\n',
                          '0  '+str(sip.kp[1])+'  0  0.0\n',
                          '0  0  '+str(sip.kp[2])+'  0.0\n',
                          '%endblock kgrid.MonkhorstPack\n'], name = 'KP')
    sip.write_more_fdf(['BandLinesScale   ReciprocalLatticeVectors'],
                        name = 'KP')
    sip.write_more_fdf(['%block BandLines'],
                        name = 'KP')
    sip.write_more_fdf(path, 
                        name = 'KP')
    sip.write_more_fdf(['%endblock BandLines'],
                        name = 'KP')


def manual_k_points(sip,reduced_k_arr,weights, tbtrans = False):
    k = reduced_k_arr.copy()
    assert len(k) == len(weights)
    L = ['kgrid.File Manual_k.fdf']
    end = '_TBT' if tbtrans else ''
    sip.delete_fdf('KP'+end)
    sip.write_more_fdf(L, name = 'KP'+end)
    with open(sip.dir + '/' + 'Manual_k.fdf', 'w') as f:
        nk = len(k)
        f.write(str(nk) + '\n')
        for i,ki in enumerate(k):
            f.write('  ' + str(i) + '  ' + str(ki[0]) + '  ' + str(ki[1]) + '  '+str(ki[2]) + '  ' + str(weights[i]) + '\n')


def Add_Bounded_Plane(sip, O, A, B,  Potential, cut_off, spread = 100.0):
    """
        Read Geometry.Hartree siesta documentation.
        O, A, B: (3,) shaped arrays
        Potential: float, 
        cutoff:    float
        spread:    float

        Do this function before writting the fdf files with
        self.fdf
    """
    sip.write_more_fdf(['%block Geometry.Hartree\n',
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
    sip.add_stuff(dic)


def Add_Charged_Bounded_Plane(sip, O, A, B,  Charge, cut_off, spread, kind = 'gauss'):
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
    sip.write_more_fdf(string,
                        name = 'STRUCT')
    dic = {'what': 'charged bounded plane',
           'charge':  Charge,
           'vectors': [O,A,B],
           'spread':  spread,
           'cut_off': cut_off,
           'kind':    kind}
    sip.add_stuff(dic)


def Add_Charged_Box(sip,O, A, B, C, Charge, Ret = False):
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
        sip.write_more_fdf(string,
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
        sip.write_more_fdf(write_this, name = 'STRUCT')

    dic = {'what': 'charged box',
           'charge':   Charge,
           'vectors': [O,A,B,C]}
    sip.add_stuff(dic)
    if Ret:
        return write_this


def Add_Charged_Sphere(sip, Cent,  R, Ch, cut_off = 5.0):
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
        sip.write_more_fdf(['%block Geometry.Charge\n',
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
        sip.write_more_fdf(write_this)


    dic = {'what': 'charged sphere',
           'charge':   Ch,
           'cut_off': cut_off,
           'vectors': [Cent, R]}
    sip.add_stuff(dic)


def add_stuff(sip, dic):
    sip.create_added_stuff_dic()
    sip.Added_stuff+= [dic]


def create_added_stuff_dic(sip):
    if hasattr(sip,'Added_stuff') == False:
        sip.Added_stuff = []


def Dipole_Correction_Vacuum(sip, Type,  direction, vacuum_point, Origin = None):
    sip.write_more_fdf(['Slab.DipoleCorrection ' + Type + '\n'], name = 'DEFAULT')

    if Origin is not None:
        pos_str = str(Origin[0]) + ' ' + str(Origin[1]) + ' ' + str(Origin[2])
        sip.write_more_fdf(['%block Slab.DipoleCorrection.Origin\n',
                             '   ' + pos_str + ' Ang\n',
                             '%endblock\n'])

    sip.write_more_fdf([
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
    sip.add_stuff(dic)


def write_tb_trans_kp(sip):
    if sip.kp_tbtrans == None and hasattr(sip, 'manual_tbtrans_kpoint') == False:
        print('no k-poins for tbtrans!')
        pass
    else:
        name='RUN'   ##self.sl
        with open(sip.dir+'/'+name+'.fdf','r') as f:
            lines=[i for i in f]
            f.close()
        with open(sip.dir+'/'+name+'_TBT.fdf','w') as f:
            for l in lines:
                if '%include KP.fdf' in l:
                    f.write('%include KP_TBT.fdf\n')
                elif 'DEFAULT' in l:
                    pass

                else: f.write(l)
            f.close()


def barebone_fdf(sip, eta = 0.0):
    barebones_RUN(sip, eta = eta)


def ideal_cell_charge(sip,chargedic,spindeg=True):
    q = 0.0
    for s in sip.s:
        q += chargedic[Num2Sym[s]]
    if spindeg:
        q *= 0.5
    return q


def get_pseudo_paths(sip, manual = [], labs = None):
    paths=[]

    atoms = unique_list(sip.s) + manual
    atoms = [a for a in atoms if a <= 200]

    Dir   = sip.pp_path
    PPs   = ld(Dir)
    for atom in atoms:
        for pp in PPs:
            name=''
            for let in pp:
                if let=='.':
                    break
                else:
                    name+=let
            if Num2Sym[atom].replace('_ghost', '') == name and sip.xc.lower() in pp:
                paths+=['../'+sip.pp_path+'/'+pp]
    sip.pseudo_paths = paths
    it=0
    for p in paths:
        if '.psf' in p:
            ossystem('ln -s '+ p +' '+sip.dir+'/'+ Num2Sym[atoms[it]]+'.psf')
            it+=1
        if '.psml' in p:
            ossystem('ln -s '+ p +' '+sip.dir+'/'+ Num2Sym[atoms[it]]+'.psml')
            it+=1


def fdf(sip, eta = 0.0, manual_pp = [], fix_hartree = False, parallel_k=False):
    """
        Standardised fdf files for running siesta

    """

    write_fdf(sip, eta = eta)
    if sip.Voltage is None:
        pass
    elif sip.Voltage!=0 and fix_hartree:
        sip.write_more_fdf(['TS.Hartree.Fix -A'])
    sip.get_pseudo_paths(manual = manual_pp)
    if hasattr(sip, '_mixing_dic'):
        for k in sip._mixing_dic.keys():
            atoms, frac = sip._mixing_dic[k]
            sip.ps_mixer(atoms, frac, k)
        sip.dump_pseudo_list_to_struct()

    if sip.dic != None:
        keys = sip.dic.keys()
        keys = [k for k in keys if 'dftb' not in k]
        for k in keys:
            v = sip.dic[k]
            if len(k.split())==1:
                which = 'DEFAULT'
            else:
                which = k.split(' ')[0]
                k = k.split(' ')[1]
            sip.write_more_fdf([k+ ' ' + v], name = which)
    if parallel_k:
        sip.set_parallel_k()


def fdf_relax(sip, Constraints, force_tol = 0.01, max_it = 1000, 
              parallel_k=False, variable_cell = False):
    if sip.solution_method=='transiesta':
        sip.fdf()
    write_relax_fdf(sip, Constraints = Constraints, 
                    force_tol = force_tol, max_it = max_it,
                    variable_cell = variable_cell)
    if sip.solution_method=='transiesta':
        sip.write_more_fdf(['%include TS_TBT.fdf'], name = 'RUN')
    sip.get_pseudo_paths()
    if parallel_k:
        sip.set_parallel_k()


def gin(sip,cell_opti=[],f=None, var_cell = None, relax_only = False, phonons=False):
    fix=[]
    if f is None and var_cell is None:
        pass
    elif f is not None:
        for i in range(len(sip.pos_real_space)):
            pi=sip.pos_real_space[i,:]
            if f(pi)==True:
                fix+=[[0,0,0]]
            else:
                fix+=[[1,1,1]]
    elif var_cell is not None:
        for i in range(len(sip.pos_real_space)):
            if i in var_cell:
                fix+=[[1,1,1]]
            else:
                fix+=[[0,0,0]]

    write_gin(sip,cell_opti=cell_opti,fix=fix, relax_only = relax_only, phonons=phonons)


def dftb_hsd(sip, Scc = 'Yes', write_cell = True, angmom_dic = {},
                   ReadInititialCharges = 'No', WriteChargesAsText = 'Yes', ReadChargesAsText = 'No',
                   WriteRealHS='No', CalculateForces='No', SkipChargeTest='No',
                   ThirdOrderFull='No', HubDeriv=None,DampCor=None, 
                   Spin=None, InitSpin=None, SpinConstants=None, CustomHubbard = None,
                   UnpairedElectrons = 0.0, PolynomialRepulsive=None,
                   ):

    write_dftb_hsd(sip, Scc = Scc, write_cell = write_cell, angmom_dic = angmom_dic,
                         ReadInitialCharges = ReadInititialCharges, WriteChargesAsText=WriteChargesAsText,
                         ReadChargesAsText=ReadChargesAsText, WriteRealHS    = WriteRealHS, 
                         CalculateForces=CalculateForces,     SkipChargeTest = SkipChargeTest,
                         ThirdOrderFull = ThirdOrderFull, HubDeriv=HubDeriv,DampCor=DampCor,
                         Spin = Spin, InitSpin = InitSpin, SpinConstants = SpinConstants,
                         CustomHubbard = CustomHubbard, UnpairedElectrons = UnpairedElectrons,
                         PolynomialRepulsive=PolynomialRepulsive, 
                         )


    if sip.dic is None:
        sip.dic = {'dftb_angmom': angmom_dic}
    else:
        sip.dic.update({'dftb_angmom': angmom_dic})
