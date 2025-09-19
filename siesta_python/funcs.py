#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:59:30 2020

@author: aleks
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sisl

ls = os.listdir

Num2Sym={1:'H', 2: 'He', 3:'Li',
         4:'Be',5: 'B', 6:'C',
         7:'N', 8:'O', 9:'F',
         10:'Ne', 11:'Na', 12:'Mg', 
         13:'Al', 14:'Si', 15:'P',
         16:'S',  17:'Cl', 18:'Ar',
         19:'K',  20:'Ca', 21:'Sc',
         22:'Ti', 23:'V',  24:'Cr', 25:'Mn',
         26:'Fe', 27:'Co', 28:'Ni', 29:'Cu',
         30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 34:'Se',
         35:'Br', 36:'Kr', 37:'Rb', 38:'Sr', 39:'Y',  40:'Zr',
         41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd',
         47:'Ag', 48:'Cd', 49:'In', 50:'Sn', 51:'Sb', 52: 'Te',
         53:'I',  54:'Xe', 55:'Cs', 56:'Ba', 
         77:'Ir', 78:'Pt', 79:'Au',
         
         201: 'mix_1',
         202: 'mix_2',
         203: 'mix_3',
         204: 'mix_4',
         205: 'mix_5',
         206: 'mix_6',
         207: 'mix_7',
         208: 'mix_8',
         209: 'mix_9',
         210: 'mix_10',
         211: 'mix_11',
         212: 'mix_12',
         213: 'mix_13',
         214: 'mix_14',
         215: 'mix_15',
         216: 'mix_16',
         217: 'mix_17',
         218: 'mix_18',
         219: 'mix_19',
         220: 'mix_20',
         -1:  'H_ghost',
         -2:  'He_ghost',
         -3:  'Li_ghost',
         -4:  'Be_ghost',
         -5:  'B_ghost',
         -6:  'C_ghost',
         -7:  'N_ghost',
         -79: 'Au_ghost'
         }
Sym2Num = {v: k for k, v in Num2Sym.items()}

def symbol_dic(x):
    if x=='GAMMA':
        return r'\GAMMA'
    else:
        return x

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        
def read_geom_from_fdf(file):
    #import sisl
    g = sisl.get_sile(file).read_geometry()
    r = g.xyz.copy()
    s = g.atoms.Z.copy()
    lat = g.cell.copy()
    R = np.zeros((len(r), 4))
    R[:,0:3] = r[:,:]
    R[:,3]   = s[:]
    return R, lat


def oldread_geom_from_fdf(file):
    pos = []
    read_lat = 0
    read_pos_type = 0
    lat_vecs = np.zeros((3,3))
    label_dict = {}
    read_labels = 0
    assert '.fdf' in file
    with open(file,'r') as F:
        it_lat=0
        f=list(F)
        for l in f:
            if '%endblock ChemicalSpeciesLabel' in l: read_labels = 0
            if read_labels == 1:
                pls = l.split()
                label_dict.update({int(pls[0].strip()) : int(pls[1].strip())})
            if '%block ChemicalSpeciesLabel' in l: read_labels = 1
        for l in f:
            if '%endblock AtomicCoordinatesAndAtomicSpecies' in l: read_pos_type = 0
            if '%endblock LatticeVectors' in l: read_lat = 0
            if read_pos_type==1:
                temp = np.fromstring(l,sep=' ')
                temp[3]=label_dict[temp[3]]
                pos+=[temp]
                
            if read_lat == 1:
                lat_vecs[it_lat,:] = np.fromstring(l,sep=' ')
                it_lat += 1
            
            if '%block AtomicCoordinatesAndAtomicSpecies' in l: read_pos_type = 1
            if '%block LatticeVectors' in l: read_lat = 1
        
    return np.asarray(pos),lat_vecs

def unique_list(l):
    o=[]
    for i in l:
        if i not in o:
            o.append(i)
    return o

def find_list(L, val):
    return [i for i, x in enumerate(L) if x == val]
def find_np_list(L, val):
    return [i for i, x in enumerate(L) if (x == val).all()]


def site_idx_to_orbital_idx(idx, sisl_H):
    idx_list = []
    it_o = 0
    out = []
    for i, a in enumerate(sisl_H.atoms):
        In = False
        if i in idx:
            idx_list += [i]
            In = True
        dummy = []
        for o in a.orbitals:
            if In:
                dummy += [it_o]
            it_o += 1
        
        if In:
            out+=[dummy]
    
    Out = []
    for i in idx:
        Out+=out[idx_list.index(i)]
    
    return Out

    
    
def R(x,n=6):
    return str(np.round(x,n))[:n+2]

def write_fdf(object,eta = 1e-2):
    custom_labs = hasattr(object, 'pseudolabel_func')
    ua = unique_list(object.s)
    if custom_labs:
        lablist = [object.pseudolabel_func(i)
                   for i in range(object.s.shape[0])]
        ual = unique_list(lablist)
    with open(object.dir+'/RUN.fdf','w') as f:
        f.write('%include STRUCT.fdf\n')
        f.write('%include KP.fdf\n')
        f.write('%include DEFAULT.fdf\n')
        f.write('%include TS_TBT.fdf\n')
        f.write('\n')
    f.close()
    with open(object.dir+'/'+'DEFAULT'+'.fdf','w') as f:
        f.write('SystemName          ' + object.sm + '\n')
        f.write('SystemLabel         ' + object.sl + '\n')
        f.write('Spin      '+ object.spin_pol+ '\n')
        f.write('PAO.BasisSize      '+ object.basis+    '\n')
        if object.write_matrices==True:
            f.write('TS.HS.Save True\n')
        if object.save_potential_grids:
            f.write('SaveElectrostaticPotential T\n')
            f.write('SaveTotalPotential True\n')
        f.write('SCF.Mixer.Method '+object.mixer+'\n')
        f.write('SCF.Mixer.History '+str(object.n_hist)+'\n')
        f.write('SCF.Mix ' + str(object.mix_what) + '\n')
        f.write('XC.Functional '+object.xc+'\n')
        f.write('XC.Authors '+object.xc_authors+ '\n')
        
        if object.lua_script != None:
            f.write('LUA.Script ' +object.lua_script +  '\n')
            f.write('LUA.Debug True\n')
        
        if object.energy_shift == None:
            pass
        else:
            f.write('PAO.EnergyShift	'+ str(object.energy_shift) +'  meV\n')
        
        if object.mesh_cutoff ==None:
            pass
        else:
            f.write('MeshCutoff            ' + str(object.mesh_cutoff)+' Ry \n')
        f.write('MaxSCFIterations      ' + str(object.max_scf_it)+'\n')
        if object.dm_mixing_weight == None:
            pass
        else:
            f.write('DM.MixingWeight       ' + str(object.dm_mixing_weight)+'\n')
        f.write('DM.Tolerance          ' + str(object.dm_tol)+'\n')
        if object.kgridcutoff==None:
            pass
        else:
            f.write('kgridcutoff           ' + str(object.kgridcutoff)+' Ang \n')
        
        f.write('SolutionMethod        ' + str(object.solution_method)+'\n')
        f.write('ElectronicTemperature ' + str(object.electronic_temperature_mev)+' meV \n')
        if object.print_mulliken == True:
            f.write('MullikenInSCF T\n')
            f.write('WriteMullikenPop 1\n')
        
        if object.reuse_dm==True:
            f.write('DM.UseSaveDM true\n')
        if object.denchar_files == True:
            f.write('Write.Denchar True\n')
        
        if isinstance(object.set_pdos,list)==True:
            f.write('%block ProjectedDensityOfStates\n   ')
            f.write('  '+str(object.set_pdos[0])+'.0  '+str(object.set_pdos[1])+'.0'+' '+str(object.set_pdos[2])+' '+str(object.set_pdos[3])+' eV\n')
            #f.write('eV\n')
            #f.write('%block ProjectedDensityOfStates\n   ')
            #f.write('  '+str(object.set_pdos[0])+'.0  '+str(object.set_pdos[1])+'.0'+' '+str(object.set_pdos[2])+' '+str(object.set_pdos[3])+' \n')
            #f.write('eV\n')
            f.write('%endblock ProjectedDensityOfStates\n')
            f.write('%block PDOS.kgrid.MonkhorstPack\n')
            for i in range(3):
                f.write('  ')
                for j in range(3):
                    if i==j:
                        f.write('  '+str(object.set_pdos[4][i]))
                    else:
                        f.write('  '+str(0))
                    if j==2:
                        f.write('  '+str(object.k_shift))
                f.write('\n')
            f.write("%endblock PDOS.kgrid.MonkhorstPack\n")
            
    f.close()
    
    with open(object.dir+'/STRUCT.fdf','w') as f:
        f.write('NumberOfAtoms       ' + str(len(object.s)) + '\n')
        if custom_labs== False:
            f.write('Number of Species   ' + str(len(ua)) + '\n')
        else:
            f.write('Number of Species   ' + str(len(ual)) + '\n')
        it=1
        atomnr_to_label={}
        
        f.write('%block ChemicalSpeciesLabel \n')
        if custom_labs==False:
            for atomnr in ua:
                f.write(' '+str(it) + '  ' + str(int(atomnr)) + '  '+Num2Sym[atomnr]+'\n')
                atomnr_to_label.update({str(ua[it-1]):str(it)})
                it+=1
        else:
            for atomnr in ual:
                f.write(' '+str(it) + '  ' + str(int(object.s[lablist.index(ual[it-1])])) + '  '+ual[it-1]+'\n')
                atomnr_to_label.update({str(ual[it-1]):str(it)})
                it+=1
        f.write('%endblock ChemicalSpeciesLabel \n \n')
        f.write('LatticeConstant	1.00 Ang\n')
        f.write('%block LatticeVectors\n')
        for i in range(3):
            f.write('  ')
            for j in range(3):
                f.write('  '+str(np.round(object.lat[i,j],6)))
            f.write('\n')
        f.write('%endblock LatticeVectors\n \n')
        
        
        f.write('AtomicCoordinatesFormat Ang\n')
        f.write('%block AtomicCoordinatesAndAtomicSpecies \n')
        for i in range(object.pos_real_space.shape[0]):
            if hasattr(object, 'pseudolabel_func') == False:
                f.write('   '+str(R(object.pos_real_space[i,0]))+'   '
                             +str(R(object.pos_real_space[i,1]))+'   '
                             +str(R(object.pos_real_space[i,2]))+'  '
                             +str(atomnr_to_label[str(object.s[i])]) +'\n')
            else:
                f.write('   '+str(R(object.pos_real_space[i,0]))+'   '
                             +str(R(object.pos_real_space[i,1]))+'   '
                             +str(R(object.pos_real_space[i,2]))+'  '
                             +str(atomnr_to_label[str(object.pseudolabel_func(i))]) +'  '
                             +str(i+1) + '  ' + str(object.pseudolabel_func(i))+' '+'\n'
                             )
            
            
        f.write('%endblock AtomicCoordinatesAndAtomicSpecies\n')
    f.close()
    with open(object.dir+'/KP.fdf','w') as f:
        if object.kp is not None:
            f.write('\n')
            f.write('%block kgrid.MonkhorstPack\n')
            for i in range(3):
                f.write('  ')
                for j in range(3):
                    if i==j:
                        f.write('  '+str(object.kp[i]))
                    else:
                        f.write('  '+str(0))
                    if j==2:
                        f.write('  '+str(object.k_shift))
                f.write('\n')
            f.write('%endblock kgrid.MonkhorstPack\n')
        bz_path_names=[]
        bz_path = []
        bz_path_num=[1]
        if object.calc_bands==True:
            v = [object.sym_dic[i] for i in object.sym_dic.keys()]
            Names  = [i for i in object.sym_dic.keys()]
            for i,kp in enumerate(v):
                name = Names[i]
                #kp = object.sym_dic[i[0]]
                
                if object.TwoDim == True:
                    if kp[2]<1e-10:
                        bz_path_names+=[name]
                        bz_path+=[kp]
                        bz_path_num +=[50]
                if object.TwoDim == False:
                    bz_path_names+=[name]
                    bz_path+=[kp]
                    bz_path_num +=[200]
        
        if object.calc_bands ==True:
            print(bz_path)
            f.write('\nBandLinesScale   ReciprocalLatticeVectors\n')
            f.write('%block BandLines\n')
            it=0
            for i in bz_path_names:
                f.write(str(bz_path_num[it])+ ' ' + str(bz_path[it][0]) + '  ' + str(bz_path[it][1]) + '  ' +str(bz_path[it][2]) + '  ' + symbol_dic(bz_path_names[it])+'\n')
                it+=1
            f.write('%endblock BandLines\n')
    
    with open(object.dir+'/KP_TBT.fdf','w') as f:
        if object.kp_tbtrans is not None:
            f.write('\n')
            f.write('%block kgrid.MonkhorstPack\n')
            for i in range(3):
                f.write('  ')
                for j in range(3):
                    if i==j:
                        f.write('  '+str(object.kp_tbtrans[i]))
                    else:
                        f.write('  '+str(0))
                    if j==2:
                        f.write('  '+str(object.k_shift))
                f.write('\n')
            f.write('%endblock kgrid.MonkhorstPack\n')
        elif hasattr(object,'manual_tbtrans_kpoint'):
            _KP = object.manual_tbtrans_kpoint
            f.write('%block TBT.k\n')
            f.write('list 1\n')
            f.write('   '+str(_KP[0]) + ' ' +str(_KP[1]) + ' ' + str(_KP[2]) + '\n')
            f.write('%endblock\n')
    
    with open(object.dir+'/TS_TBT.fdf','w') as f:
        if len(object.buffer_atoms)==0:
            pass
        elif isinstance(object.buffer_atoms, np.ndarray) or isinstance(object.buffer_atoms, list):
            f.write('TS.Atoms.Buffer ' +  listinds_to_string(numpy_inds_to_string(np.array(object.buffer_atoms) + 1)) +'\n')
            f.write('TBT.Atoms.Buffer ' +  listinds_to_string(numpy_inds_to_string(np.array(object.buffer_atoms) + 1)) +'\n')
        
        elif isinstance(object.buffer_atoms,str):
            f.write(object.buffer_atoms + '\n')
            f.write('TBT.Atoms.Buffer' + object.buffer_atoms.split()[1] + '\n')
        if len(object.Device_atoms) == 0:
            pass
        elif isinstance(object.Device_atoms, np.ndarray) or isinstance(object.Device_atoms, list):
            f.write('TBT.Atoms.Device ' +  listinds_to_string(numpy_inds_to_string(np.array(object.Device_atoms) + 1)) +'\n')
            
        elif isinstance(object.Device_atoms,str):
            f.write(object.Device_atoms)
        if object.save_SE == True:
            f.write('TBT.SelfEnergy.Save True\n')
            if object.save_SE_only == True:
                f.write('TBT.SelfEnergy.Only True\n')
            
        
        
        # elif len(object.buffer_atoms)==2:
        #     f.write('TS.Atoms.Buffer ['+str(object.buffer_atoms[0])+' -- '+str(object.buffer_atoms[1])+']\n')
        
        if len(object.elecs)>0:
            
            Unique_Chem_Pot = unique_list(object.Chem_Pot)
            Which_letter = [Unique_Chem_Pot.index(mu) for mu in object.Chem_Pot]
            
            f.write('TBT.T.Bulk True\n')
            f.write('TBT.DOS.Elecs True\n')
            f.write('TBT.DOS.A.All True\n')            
            
            xyz=['x','y','z','a','b','c','d','e','f']
            
            if object.NEGF_calc==True:    
                f.write('TS.Voltage '+str(object.Voltage)+' eV \n')
            f.write('%block TS.Elecs\n')
            it=0
            for e in object.elecs:
                f.write('  '+e.sl+'\n')
                it+=1
            f.write('%endblock TS.Elecs\n')
            it=0
            f.write('%block TS.ChemPots\n')
            for CP in Unique_Chem_Pot:
                f.write('  '+xyz[it]+'\n')
                it+=1
            f.write('%endblock TS.ChemPots\n')
            
            it=0
            for CP in Unique_Chem_Pot:
                let=xyz[it]
                block='TS.ChemPot.'+ let
                f.write('%block '+block+'\n')
                f.write('  mu '+str(CP)+' eV\n')
                f.write('  contour.eq\n')
                f.write('    begin\n')
                f.write('      C-'+let+'\n')
                f.write('      T-'+let+'\n')
                f.write('    end\n')
                f.write('%endblock '+block+'\n')
                it+=1
            
            it=0
            for CP in Unique_Chem_Pot:
                let=xyz[it]
                if object.contour_settings is None:
                    f.write('%block TS.Contour.C-'+let+'\n')
                    f.write('part circle\n')
                    f.write('   from '+str(CP)+' eV -40. eV to '+str(CP)+' eV -10 kT \n')
                    f.write('     points 25\n')
                    f.write('      method g-legendre\n')
                    f.write('%endblock\n')
                    
                    f.write('%block TS.Contour.T-'+let+'\n')
                    f.write('part tail\n')
                    f.write('   from prev to inf\n')
                    f.write('     points 10\n')
                    f.write('      method g-fermi\n')
                    f.write('%endblock\n')
                else:
                    CS = object.contour_settings[object.Chem_Pot.index(CP)]
                    v1 = CS['V1']
                    v2 = CS['V2']
                    np1= CS['Np_1']
                    np2= CS['Np_2']
                    
                    f.write('%block TS.Contour.C-'+let+'\n')
                    f.write('part circle\n')
                    f.write('   from '+str(CP)+' eV '+v1+' eV to '+str(CP)+' eV '+v2+' kT \n')
                    f.write('     points '+np1+'\n')
                    f.write('      method g-legendre\n')
                    f.write('%endblock\n')
                    
                    f.write('%block TS.Contour.T-'+let+'\n')
                    f.write('part tail\n')
                    f.write('   from prev to inf\n')
                    f.write('     points '+np2+'\n')
                    f.write('      method g-fermi\n')
                    f.write('%endblock\n')
                
                it+=1
            
            if object.NEGF_calc==True:
                f.write('%block TS.Contours.nEq\n')
                f.write('  neq\n')
                f.write('%endblock TS.Contours.nEq\n')
                f.write('%block TS.Contour.nEq.neq\n')
                f.write('  part line\n')
                f.write('   from '+str(min(object.Chem_Pot))+' eV - 5 kT to '+str(max(object.Chem_Pot))+' eV + 5 kT\n')
                f.write('     delta 0.01 eV\n')
                f.write('      method mid-rule\n')
                f.write('%endblock TS.Contour.nEq.neq\n')
            
            it=0
            for e in object.elecs:
                f.write('%block TS.Elec.'+e.sl+'\n')
                f.write('   HS ../'+e.dir+'/'+e.sl+'.TSHS\n')
                if e.elec_bloch is not None and e.elec_bloch != [1,1,1]:
                    f.write('   ' + 'bloch ' +str(e.elec_bloch[0]) +' ' + str(e.elec_bloch[1]) +' ' + str(e.elec_bloch[2]) +' \n')
                f.write('   chemical-potential '+ xyz[Which_letter[it]]+'\n')
                f.write('   semi-inf-direction '+e.semi_inf+'\n')
                if e.elec_RSSE == True:
                    f.write('   out-of-core True\n')
                    f.write('   Gf ../'+e.dir+'/'+e.sl+'.TBTGF\n')
                    f.write('   Gf.Reuse True\n')
                f.write('   electrode-position '+str(object.elec_inds[it][0]+1)+'\n')
                if object.custom_tbtrans_contour is not None:
                    f.write('   Eta -1.0 eV\n')
                
                f.write('%endblock TS.Elec.'+e.sl+'\n')
                it+=1
        
        n_ooc = sum([e.elec_RSSE for e in object.elecs])
        f.write('\n')
        if object.custom_tbtrans_contour is not None:
            f.write('%block TBT.Contour.line\n')
            f.write('  from '+str(object.custom_tbtrans_contour.real.min())+' eV to '+str(object.custom_tbtrans_contour.real.max())+' eV\n')
            f.write('   file my_energies\n')
            f.write('%endblock \n')
            f.write('TBT.Contours.Eta  '+str(0.0)+' eV')
            with open(object.dir+'/my_energies','w') as q:
                for z in object.custom_tbtrans_contour:
                    q.write(str(z.real)+' ' + str(z.imag) + ' 1.0 eV\n')
                
            
        elif len(object.elecs)==0 or n_ooc == 0:
            f.write('%block TBT.Contour.line\n')
            f.write('  from '+str(object.trans_emin)+' eV to '+str(object.trans_emax)+' eV\n')
            f.write('   delta '+str(object.trans_delta)+' eV\n')
            f.write('    method mid-rule\n')
            f.write('%endblock \n')
            f.write('TBT.Contours.Eta  '+str(eta)+' eV')
        elif n_ooc>0:
            ooc_idx = [e.elec_RSSE for e in object.elecs].index(True)
            f.write('%block TBT.Contour.line\n')
            f.write('  from '+str(object.elecs[ooc_idx].RSSE_Energy_from_to[0])+' eV to '+str(object.elecs[ooc_idx].RSSE_Energy_from_to[1])+' eV\n')
            f.write('   file ../' +object.elecs[ooc_idx].dir + '/' + 'contour.E\n')
            f.write('%endblock \n')
            f.write('TBT.Contours.Eta  '+str(eta)+' eV')
            
def write_relax_fdf(object, Constraints, force_tol = 0.01, max_it = 1000,
                    variable_cell = False):
    ua = unique_list(object.s)
    with open(object.dir+'/RUN.fdf','w') as f:    
        f.write('%include STRUCT.fdf\n')
        f.write('%include KP.fdf\n')
        f.write('%include DEFAULT.fdf\n')
        f.write('%include MD.fdf\n')
        f.write('\n')
    f.close()
    
    with open(object.dir+'/'+'DEFAULT'+'.fdf','w') as f:
        f.write('SystemName          ' + object.sm + '\n')
        f.write('SystemLabel         ' + object.sl + '\n')
        f.write('Spin      '+ object.spin_pol+ '\n')
        f.write('PAO.BasisSize      '+ object.basis+    '\n')
        if object.write_matrices==True:
            f.write('TS.HS.Save T\n')
        f.write('SaveElectrostaticPotential T\n')
        f.write('SaveTotalPotential True\n')
        f.write('SCF.Mixer.Method '+object.mixer+'\n')
        f.write('SCF.Mixer.History '+str(object.n_hist)+'\n')
        
        if object.energy_shift == None:
            pass
        else:
            f.write('PAO.EnergyShift	'+ str(object.energy_shift) +'  meV\n')
        if object.mesh_cutoff ==None:
            pass
        else:
            f.write('MeshCutoff            ' + str(object.mesh_cutoff)+' Ry \n')
        f.write('MaxSCFIterations      ' + str(object.max_scf_it)+'\n')
        if object.dm_mixing_weight == None:
            pass
        else:
            f.write('DM.MixingWeight       ' + str(object.dm_mixing_weight)+'\n')
        f.write('DM.Tolerance          ' + str(object.dm_tol)+'\n')
        if object.kgridcutoff==None:
            pass
        else:
            f.write('kgridcutoff           ' + str(object.kgridcutoff)+' Ang \n')
        f.write('SolutionMethod        ' + str(object.solution_method)+'\n')
        f.write('ElectronicTemperature ' + str(object.electronic_temperature_mev)+' meV \n')
        #f.write('MullikenInSCF T\nWriteMullikenPop 1\n')
        
        if object.reuse_dm==True:
            f.write('DM.UseSaveDM true\n')
        if isinstance(object.set_pdos,list)==True:
            f.write('%block ProjectedDensityOfStates\n   ')
            f.write('  '+str(object.set_pdos[0])+'.0  '+str(object.set_pdos[1])+'.0'+' '+str(object.set_pdos[2])+' '+str(object.set_pdos[3])+' ')
            f.write('eV\n')
            f.write('%endblock ProjectedDensityOfStates\n')
    f.close()
    
    
    with open(object.dir+'/STRUCT.fdf','w') as f:
        f.write('NumberOfAtoms       ' + str(len(object.s)) + '\n')
        f.write('Number of Species   ' + str(len(ua)) + '\n')
        it=1
        atomnr_to_label={}
        
        f.write('%block ChemicalSpeciesLabel \n')
        for atomnr in ua:
            f.write(' '+str(it) + '  ' + str(int(atomnr)) + '  '+Num2Sym[atomnr]+'\n')
            atomnr_to_label.update({str(ua[it-1]):str(it)})
            it+=1
        f.write('%endblock ChemicalSpeciesLabel \n \n')
        f.write('LatticeConstant	1.00 Ang\n')
        f.write('%block LatticeVectors\n')
        for i in range(3):
            f.write('  ')
            for j in range(3):
                f.write('  '+str(np.round(object.lat[i,j],5)))
            f.write('\n')
        f.write('%endblock LatticeVectors\n \n')
        
        
        f.write('AtomicCoordinatesFormat Ang\n')
        f.write('%block AtomicCoordinatesAndAtomicSpecies \n')
        for i in range(len(object.pos_real_space[:,0])):
            f.write('   '+str(R(object.pos_real_space[i,0]))+'   '+str(R(object.pos_real_space[i,1]))+'   '+str(R(object.pos_real_space[i,2]))+'  '+str(atomnr_to_label[str(object.s[i])]) +'\n')
        f.write('%endblock AtomicCoordinatesAndAtomicSpecies\n')
    f.close()
    with open(object.dir+'/KP.fdf','w') as f:
        if object.kp is not None:
            f.write('\n')
            f.write('%block kgrid.MonkhorstPack\n')
            for i in range(3):
                f.write('  ')
                for j in range(3):
                    if i==j:
                        f.write('  '+str(object.kp[i]))
                    else:
                        f.write('  '+str(0))
                    if j==2:
                        f.write('  '+str(object.k_shift))
                f.write('\n')
            f.write('%endblock kgrid.MonkhorstPack\n')
        bz_path_names=[]
        bz_path = []
        bz_path_num=[1]
        if object.calc_bands==True:
            for i in object.k_path:
                kp = object.sym_dic[i[0]]
                if object.TwoDim == True:
                    if kp[2]<1e-10:
                        bz_path_names+=[i[0]]
                        bz_path+=[object.sym_dic[i[0]]]
                        bz_path_num +=[100]
                if object.TwoDim == False:
                    bz_path_names+=[i[0]]
                    bz_path+=[i]
                    bz_path_num +=[50]
        
            f.write('\nBandLinesScale   ReciprocalLatticeVectors\n')
            f.write('%block BandLines\n')
            it=0
            for i in bz_path_names:
                f.write(str(bz_path_num[it])+ ' ' + str(bz_path[it][0]) + '  ' + str(bz_path[it][1]) + '  ' +str(bz_path[it][2]) + '  ' + symbol_dic(bz_path_names[it])+'\n')
                it+=1
            f.write('%endblock BandLines\n')
            object._BANDPATH = bz_path
    
    force_tol = str(force_tol)
    with open(object.dir + '/MD.fdf','w') as f:
        f.write('MD.TypeOfRun          	CG  \n')
        f.write('MD.NumCGsteps     	'+str(max_it)+ '\n')
        f.write('MD.MaxForceTol 	        '+ force_tol +' eV/Ang \n')
        if variable_cell:
            f.write('MD.VariableCell      	T\n')
        else:
            f.write('MD.VariableCell      	F\n')
        # f.write('MD.MaxStressTol         0.0010 eV/Ang\n')
        f.write('WriteMDHistory      	T \n')
        f.write('WriteMDXMol  		T\n')
        f.write('%block Geometry.Constraints\n')
        for line in Constraints:
            f.write('    ' + line + ' \n')
        f.write('%endblock\n')

def write_gin(object,cell_opti=[],fix=[],relax_only = False, phonons=False):
    with open(object.dir+'/Gulp.gin','w') as f:
        ##Fra Mads##
        if relax_only == True:
            f.write('opti dist full nosymmetry\n')
        elif phonons==True:
            f.write('nosymmetry phon eigenvectors dynamical_matrix\n')
        else:
            f.write('opti dist full nosymmetry phon eigenvectors dynamical_matrix\n')
        
        f.write('cutd 5.0\n')
        f.write('vectors\n')
        for i in range(3):
            f.write('  ')
            for j in range(3):
                f.write('  '+str(np.round(object.lat[i,j],5)))
            f.write('\n')
        for i in cell_opti:
            f.write(str(i)+' ')
        f.write('\n')
        f.write('cartesian 360\n')
        for i in range(len(object.pos_real_space)):
            atom=Num2Sym[object.s[i]]
            p=object.pos_real_space[i]
            if len(fix)>0:
                constraints=fix[i]
            else:
                constraints=[0,0,0]
            f.write(atom+'  '+'core'+'  '+str(R(p[0]))+'  '+str(R(p[1]))+'   '+str(R(p[2]))+'     0.0   1.0   0.0  '+str(constraints[0])+' '+str(constraints[1])+'  '+str(constraints[2])+'\n')
        
        f.write('output xyz Gulp_relaxed.xyz\n')
        f.write('brenner\n')
        f.write('output phon gulp\n')
        f.write('dump Gulp.res\n')

def read_total_energy(object):
    Res=None
    with open(object.dir+'/'+'RUN.out','r') as f:
        for l in f:
            if 'siesta: FreeEng' in l:
                Res=float(l[20:])
                break
    return Res
def read_fermi_level(object):
    Res=None
    with open(object.dir+'/'+'RUN.out','r') as f:
        for l in f:
            if 'siesta:' in l and 'Fermi =' in l:
                Res=float(l[25:])
                break
    return Res

def hist_distances(pos,cutoff=3):
    n=pos.shape[0]
    d=[]
    for i in range(n):
        for j in range(n):
            dij=np.linalg.norm(pos[i]-pos[j])
            if dij<cutoff and i!=j:
                d+=[dij]
    plt.hist(d)

def read_gulp_results(object):
    with open(object.dir+'/Gulp_relaxed.xyz','r') as f:
        count=0
        pos=[]
        atoms=[]    
        for l in f:
            if count==0:
                num_at=int(l[0:len(l)-2])
            if count==1:
                E=float(l[10:len(l)-2])
            if count>=2:
                atoms+=[Sym2Num[l[0:5].strip()]] 
                pos+=[np.fromstring(l[5:],sep=' ')]
            count+=1
    f.close()
    with open(object.dir+'/Gulp.gout','r') as f:
        read=False
        count=0
        A=np.zeros((3,3))
        for l in f:
            if count>3:
                break
            if count>0:
                A[count-1,:]=np.fromstring(l,sep=' ')
            if read==True:
                count+=1
            if 'Final Cartesian lattice vectors (Angstroms)' in l:
                read=True
    if (A==np.zeros((3,3))).all():
        A=object.lat.copy()
    
    for p in pos:
        if p[0]<0:
            p[0]+=A[0,0]
        if p[1]<0:
            p[1]+=A[1,1]
    
    
    return np.array(pos),A,num_at,E,
    

def read_analyze(object):
    L=[]
    opts = os.listdir(object.dir)
    for o in opts:
        if 'Analyze.out' in o:
            file_name=o
            break

    with open(object.dir+'/'+file_name,'r') as f:
        Methods = []
        for line in f:
            L+=[line]
    n=0
    for line in L:
        n+=1
        if 'Minimum memory required pivoting scheme:' in line:
            WRITE = L[n][2:]#line[2:]
            break
    
    return WRITE


def FORMAT(a):
    return "%.7f" % a


def struct_fdf_to_xsf(fdf):
    p, A = read_geom_from_fdf(fdf)
    n=p.shape[0]
    with open(fdf[0:len(fdf)-4]+'.xsf','w') as f:
        f.write('PRIMVEC\n')
        for i in range(A.shape[0]):
            f.write('   '+FORMAT(A[i,0])+'   '+FORMAT(A[i,1])+'   '+FORMAT(A[i,2])+'\n')
        f.write('\n\n')
        
        f.write('PRIMCOORD\n')
        f.write(str(n)+' 1\n')
        for i in range(n):
            f.write('  '+str(int(p[i,3]))+' '+
                     FORMAT(p[i,0])+' '+
                     FORMAT(p[i,1])+' '+
                     FORMAT(p[i,2])+' \n')
        f.write('\n\n')
        f.close()

def aimsgb_to_numpy(is_dot_init):
    L = is_dot_init.__str__()
    L=L.splitlines()
    s1 = L[5][9:len(L[5])]
    s2 = L[6][9:len(L[6])]
    s3 = L[7][9:len(L[7])]
    A=np.fromstring(s1,sep=' ')
    B=np.fromstring(s2,sep=' ')
    C=np.fromstring(s3,sep=' ')
    return A,B,C

def read_xsf(file):
    lat_vecs = np.zeros((3,3))
    with open(file,'r') as f:
        read_prim_coord = 0
        read_prim_vec = 0
        it_lat = 0
        it_at = 0
        pos=[]
        for l in f:
            if it_lat ==3: read_prim_vec= 0
                
            
            if read_prim_coord == 1:
                if it_at>0:
                    temp = np.fromstring(l,sep=' ')
                    temp=np.array([temp[i] for i in range(-3,1)])
                    pos+=[temp]
                else:
                    temp=np.fromstring(l,sep=' ')
                    nat = int(temp[0])
                it_at +=1
                
            if read_prim_vec == 1:
                lat_vecs[it_lat,:] = np.fromstring(l,sep=' ')
                it_lat += 1
            
            
            if 'PRIMVEC' in l: read_prim_vec = 1
            if 'PRIMCOORD' in l: read_prim_coord = 1
            
    return np.array(pos),lat_vecs,nat

def read_pdos_file(fname):
    PDOS=[]
    with open(fname,'r') as f:
        for l in f:
            if '#' not in l:
                PDOS+=[np.fromstring(l,sep=' ')]
    return np.array(PDOS)

def writerun_denchar(object):
    with open(object.dir + '/run_denchar.fdf', 'w') as f:
        f.write('SystemLabel ' + object.sl + '\n' )
        f.write('%include STRUCT.fdf\n' )
        f.write('Denchar.PlotCharge .TRUE.\n')
        f.write('Denchar.CoorUnits Ang\n')
        f.write('Denchar.MinX ' +str(object.lat[0,0]) + ' Ang\n')
        f.write('Denchar.MaxX ' +str(object.lat[0,1]) + ' Ang\n')
        f.write('Denchar.MinY ' +str(object.lat[1,0]) + ' Ang\n')
        f.write('Denchar.MaxY ' +str(object.lat[1,1]) + ' Ang\n')
        
        
        
    os.chdir(object.dir)
    os.system('denchar run_denchar.fdf')
    os.chdir('..')
    

def read_siesta_relax(fname):
    pos = []
    lattice = []
    read_species = False
    read_lattice = False
    read_pos     = False
    species_dict = {}
    
    if '.XV' in fname:
        #import sisl
        g = sisl.io.siesta.xvSileSiesta(fname).read_geometry()
        xyz = g.xyz
        t   = g.atoms.Z
        return np.hstack((xyz, t[:,np.newaxis])), g.cell
    
    with open(fname,'r') as f:
        for l in f:
            if '%endblock ChemicalSpeciesLabel' in l:
                read_species = False
            if 'outcell: Unit cell vectors (Ang)' in l:
                read_pos = False
            if 'outcell: Cell vector modules (Ang)' in l:
                read_lattice = False
                
            if read_species:
                r = l.split()[0:2]
                species_dict.update({int(r[0]): int(r[1])})
            if read_pos:
                r = l.split()
                #print(r)
                if len(r)>0:
                    p = np.array([float(r[0]), float(r[1]), float(r[2]), species_dict[int(r[3])]])
                    pos+=[p]
                else:
                    read_pos = False
            if read_lattice:
                r = l.split()
                #print(r)
                if len(r)>0:
                    lattice+=[[float(e) for e in r]]
            
            if '%block ChemicalSpeciesLabel' in l:
                read_species = True
            if 'outcoor: Relaxed atomic coordinates (Ang)' in l:
                read_pos     = True
            if 'outcell: Unit cell vectors (Ang)' in l:
                read_lattice = True
                lattice = []
                
    pos = np.array(pos)
    lattice = np.array(lattice)
    return pos, lattice

def numpy_inds_to_string(i_in):
    out = []
    inds = sorted(list(i_in.copy()))
    inds+= [max(inds) + 10,max(inds) + 11,max(inds) + 12 ]
    in_range = []
    def is_connected(i,j):
        for k in range(i,j-1):
            #print(k)
            idx_0, idx_1 = inds[k], inds[k+1]
            if idx_1-idx_0>1:
                return False
        return True
    for i in range(len(inds)):
        
        j=0
        j += i+1
        while is_connected(i,j) and j<=len(inds)-1:
            j+=1
        
        
        # print(i,j)
        temp = [i,j-1]
        IN = False
        for o in out:
            if o[1]==temp[1] and o[0]<temp[0]:
                IN=True
        if IN == False:
            out+=[temp]
    
    for i in range(len(out)):
        
        if out[i][0] == out[i][1]-1:
            out[i] = out[i][0]
    out = out[:-1]
        
    inds_out = []
    for o in out:
        if isinstance(o,int):
            inds_out+=[inds[o]]
        else:
            inds_out+=[ [inds[o[0]], inds[o[1]-1] ] ]
    
    return inds_out

def listinds_to_string(l):
    s = '['
    for i in l:
        if isinstance(i,np.int64):
            s+=str(i)+  ', '
        else:
            if i[0]==i[1]-1:
                s+=str(i[0])+', ' + str(i[1])+', '
            else:
                s+=str(i[0])+' -- '+str(i[1])+ ', '
    s = s[:-2]
    s+=']'
    return s

def Make_GB_Zmatrix(pos, types, idx_m1, idx_gb, idx_m2):
    #STR ='ZM.UnitsLength Ang\n'
    Ang2Bohr = 1.8897259886
    STR= '%block Zmatrix\n'
    STR+=' cartesian Ang\n'
    for i in idx_m1:
        t = types[i]
        p = Ang2Bohr*pos[i]
        STR+= '   '+str(int(t))+' '+ str(R(p[0])) + ' '+ str(R(p[1])) + ' ' + str(R(p[2])) + ' 0 0 0\n'
    STR+=' cartesian Ang\n'
    for i in idx_gb:
        t = types[i]
        p = Ang2Bohr*pos[i]
        STR+= '   '+str(int(t))+' '+ str(R(p[0])) + ' ' + str(R(p[1])) + ' ' + str(R(p[2])) + ' 1 1 1\n'
    STR+=' cartesian Ang\n'
    for i in idx_m2:
        t = types[i]
        p = Ang2Bohr*pos[i]
        STR+= '   '+str(int(t))+' '+ 'x' + str(i) + ' ' + str(R(p[1])) + ' ' + str(R(p[2])) + ' 1 1 1\n'
    # for i in idx_m2:
        # STR+=' x'+str(i)+' ' + str(R(pos[i,0])) +'\n'
    
    i0 = idx_m2[0]
    STR+=' variables\n'
    STR+='  x'+str(i0) +' '+ str(Ang2Bohr*pos[i0,0]) + '\n'
    STR+= ' constraints\n'
    for i in idx_m2[1:]:
        STR+='   x'+str(i) +' '+'x'+str(i0)+' 1.0 '+str(str(R(Ang2Bohr*pos[i,0]-Ang2Bohr*pos[i0,0])))+'\n'
    STR+= '%endblock Zmatrix\n'
    return STR

def get_random_from_LM2(done_list, return_name = False):
    f = None
    while f is None:
        angles = os.listdir('LM2')
        r1 = np.random.randint(len(angles))
        angle = angles[r1]
        cells = os.listdir('LM2/'+angle)
        f = cells[np.random.randint(len(cells))]
        if f in done_list:
            f = None
    
    tal = f.replace(',', ' ').replace('[' ,' ').replace(']' ,' ').replace('_', ' ').replace('(', ' ').replace(')', ' ').split()
    
    num = (int(tal[2]),int(tal[3]),int(tal[4]),int(tal[5]),int(tal[6]),int(tal[7]),int(tal[8]),int(tal[9]))
    v   = [(float(tal[10]), float(tal[11]))]
    if return_name:
        return num, v, f
    return num, v

def get_random_from_name(name,done_list, return_name = False):
    f = None
    while f is None:
        angles = os.listdir(name)
        if len(angles)>0:
            r1 = np.random.randint(len(angles))
            angle = angles[r1]
            cells = os.listdir(name + '/'+angle)
            if len(cells)>0:
                f = cells[np.random.randint(len(cells))]
                if f in done_list:
                    f = None
    
    tal = f.replace(',', ' ').replace('[' ,' ').replace(']' ,' ').replace('_', ' ').replace('(', ' ').replace(')', ' ').split()
    
    num = (int(tal[2]),int(tal[3]),int(tal[4]),int(tal[5]),int(tal[6]),int(tal[7]),int(tal[8]),int(tal[9]))
    v   = [(float(tal[10]), float(tal[11]))]
    if return_name:
        return num, v, f
    return num, v


def plotamok(Directory):
    #import sisl
    from sisl.viz.plotly import Plot
    try:
        t  = sisl.get_sile('siesta.TBT.nc')
        hs = sisl.get_sile('RUN.fdf').read_hamiltonian()
        ES = sisl.get_sile('ElectrostaticPotential.grid.nc').read_grid()
        V  = sisl.get_sile('TotalPotential.grid.nc').read_grid()
        z_avg  = np.average(t.xyz[:,2])
        plane_es = ES[:,:, ES.index(z_avg)]
        plane_vs =  V[:,:, V.index(z_avg)]
    
    except:
        pass

def read_buffer_atoms(file):
    with open(file, 'r') as f:
        for l in f:
            if 'TS.Atoms.Buffer' in l:
                return l

def read_electrode(file):
    L = []
    with open(file, 'r') as f:
        for l in f:
            if 'HS ..' in l:
                L+=[l.split(sep = '/')[1]]
        f.close()
    return L

def recreate_old_calculation(Dir,electrode = False):
    p, l = read_geom_from_fdf(Dir + '/STRUCT.fdf')
    string = read_buffer_atoms(Dir + '/TS_TBT.fdf')
    e = read_electrode(Dir + '/TS_TBT.fdf')
    t = p[:,3]
    p = p[:,0:3]
    device = [p,l,t,string]
    elecs = [read_geom_from_fdf(e[i] + '/STRUCT.fdf') for i in range(len(e))]
    
    return device,elecs



    
    




def SC(p):
    plt.scatter(p[:,0], p[:,1])
    
def is_string_int(string):
    if '.' in string or is_string_char(string)==True:
        return False
    return True
def is_string_char(string):
    return string.lower().islower()

def read_mulliken(file):
    Vals = []
    Q_tot= []
    with open(file,'r') as f:
        S = False
        it = 0
        for l in f:
            
            if 'mulliken: Qtot =' in l:
                S = False
                Q_tot += [l.split()[3]]
                it=0
                Vals += [LINES]
            
            if S:
                v = l.split()
                if len(v)>0:
                    LINES += [[v]]
                
            if 'mulliken: Atomic and Orbital Populations:' in l:
                S = True
                LINES = []
    
    return Vals, Q_tot


def sort_mulliken(v, Q_tot):
    
    idx = []
    Vals = []
    Switch = False 
    it = -1
    for v1 in v:
        for v2 in v1:
            for v3 in v2:
                for v4 in v3:
                    if is_string_char(v4) and Switch == True:
                        Switch = False
                    
                    if Switch and is_string_int(v4) == False:
                        Vals[it] += [float(v4)]
                    
                    if is_string_int(v4):
                        Switch = True
                        Vals += [[]]
                        idx+=[int(v4)]
                        it+=1
    
    lengths = [len(v) for v in Vals]
    max_length = max(lengths)
    Arr = np.zeros((len(Vals), max_length))
    for i, v in enumerate(Vals):
        Arr[i,0:len(v)] = v
    
    return np.array(idx)-1, Arr

def Mulliken(file):
    v,q  = read_mulliken(file)
    return sort_mulliken(v,q)

def Plot_mulliken(file, Step, atoms, xyz, perspective = 111, adjust_size = 1, which = 0, MAP = None):
    if len(atoms) != len(xyz):
        print('you need the same amount of atomic indecies and positions')
        return
    from mpl_toolkits.mplot3d import Axes3D
    idx, C = Mulliken(file)
    Vals = [C[np.where(idx == a),which][0,Step] for a in atoms]
    fig = plt.figure()
    ax = fig.add_subplot(perspective, projection='3d')
    Vals = np.array(Vals)
    Vals = Vals/Vals.max()
    if MAP is not None:
        Vals = MAP(Vals)
    
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=adjust_size * np.pi*Vals**2*100, c='blue', alpha=0.75)
    plt.show()


def read_contour_from_failed_RSSE_calc(file):
    Vals = []
    with open(file, 'r') as f:
        Switch = False
        for l in f:
            if len(l)>5:
                l.replace('\n', '')
            if l == '\n' and Switch == True:
                Switch = False
            if len(l.split())>4 and Switch == True:
                Switch = False
            if Switch: 
                Vals+= [l]
            if 'TS E.real [eV]' in l:
                Switch = True
    Arr = np.array([np.fromstring(li,sep = ' ') for li in Vals] )
    return Arr

def Identity(x):
    return x

def reciprocal_lattice(lattice):
    a1 = lattice[0,:]
    a2 = lattice[1,:]
    a3 = lattice[2,:]
    
    V = np.cross(a1,a2).dot(a3)
    
    k1 = 2 * np.pi * np.cross(a2,a3) / V
    k2 = 2 * np.pi * np.cross(a3,a1) / V
    k3 = 2 * np.pi * np.cross(a1,a2) / V
    return np.array([k1,k2,k3])

def d_proj_v(A, v):
    return (A*(v/np.linalg.norm(v))).sum(axis=1)

def Electrode_bands(sip_electrode):
    from sisl import BandStructure
    sym_dic = sip_electrode.sym_dic
    bz_path_names = []
    bz_path = []
    z_idx = np.where
    for i in sip_electrode.k_path:
        kp = sip_electrode.sym_dic[i[0]]
        if sip_electrode.TwoDim == True:
            if kp[2]==0:
                bz_path_names+=[i[0]]
                bz_path+=[sip_electrode.sym_dic[i[0]]]
    return bz_path, bz_path_names

#######   Lua  #######

def lua_format(v1):
    return '{:.10E}'.format(v1)

def write_3array_lua(np_arr, name):
    n = len(np_arr) 
    s = '     '
    with open(name, 'w') as f:
        f.write(s + str(n) + '\n')
        for i in range(n):
            if i+1==10:
                s = s[:-1]
            if i+1==100:
                s = s[:-1]
            if i+1==1000:
                s = s[:-1]
            if i+1==10000:
                s = s[:-1]
            
            v1,v2,v3 = np_arr[i]
            s1,s2,s3 = '   ', '   ', '   '
            if v1<0.0:
                s1 = s1[:-1]
            if v2<0.0:
                s2 = s2[:-1]
            if v3<0.0:
                s3 = s3[:-1]
                
            f.write(s + str(i+1) + s1 + lua_format(v1) + s2 + lua_format(v2) + s3 + lua_format(v3) + '\n')
    with open(name + '_history', 'a') as f:
        f.write(s + str(n) + '\n')
        for i in range(n):
            if i+1==10:
                s = s[:-1]
            if i+1==100:
                s = s[:-1]
            if i+1==1000:
                s = s[:-1]
            if i+1==10000:
                s = s[:-1]
            
            v1,v2,v3 = np_arr[i]
            s1,s2,s3 = '   ', '   ', '   '
            if v1<0.0:
                s1 = s1[:-1]
            if v2<0.0:
                s2 = s2[:-1]
            if v3<0.0:
                s3 = s3[:-1]
                
            f.write(s + str(i+1) + s1 + lua_format(v1) + s2 + lua_format(v2) + s3 + lua_format(v3) + '\n')
        f.write('\n ')

def doped_graphene_EF(n):
    return 4.2537 * np.sign(n) * np.sqrt(np.abs(n))

def barebones_RUN(object,eta = 1e-2):
    with open(object.dir + '/RUN.fdf', 'w') as f:
        f.write('%include TS_TBT.fdf')
    
    with open(object.dir + '/TS_TBT.fdf', 'w') as f:
        f.write('TBT.k '+ str(object.kp_tbtrans).replace(',', '') +'\n')
        f.write('TBT.HS ' + object.sl + '.TSHS\n')
        f.write('%block TBT.Contour.line\n')
        f.write('    from '+ str(object.trans_emin)+' eV to '+ str(object.trans_emax) +' eV\n')
        f.write('         delta ' + str(object.trans_delta) + ' eV\n')
        f.write('              method mid-rule\n')
        f.write('%endblock\n')
        f.write('TBT.Contours.Eta ' + str(eta) +' eV \n' )
        dummy_names = ['Left', 'Right']
        for i,e in enumerate(object.elecs):
            f.write('%block TBT.Elec.' + dummy_names[i] + '\n')
            f.write('  HS ../'+e.dir + '/'+ e.sl + '.TSHS \n')
            f.write('  semi-inf-direction ' + e.semi_inf + '\n')
            if e.elec_bloch is not None:
                f.write('  bloch ' + str(e.elec_bloch).replace(',', '').replace('[', '').replace(']', '') + '\n')
            f.write('   electrode-position '+str(object.elec_inds[i][0]+1)+'\n')
            f.write('%endblock TBT.Elec.'+ dummy_names[i] + '\n')
            

def diff_central(x, y):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    f = (x2 - x1)/(x2 - x0)
    return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)


def calc_current(tbt, i,j, mui, muj,idx, kTi = 0.025, kTj = 0.025):
    from scipy.integrate import simpson
    E = tbt.E[idx]
    tij = tbt.transmission(i,j)[idx]
    fi = 1/(1+np.exp((E - mui)/kTi))
    fj = 1/(1+np.exp((E - muj)/kTj))
    integrand = tij * (fj - fi)
    return simpson(integrand, E)

def write_mpr(object, proj):
    with open(object.dir + '/' + object.sl + '.mpr', 'w') as f:
        f.write(object.sl + '\n')
        f.write('DOS\n')
        groups = [i for i in proj.keys()]
        for g in groups:
            f.write(g + '\n')
            f.write(str(proj[g]).replace(',', '').replace('[','').replace(']','') + '\n')

def subgroups(proj):
    names     = [i for i in proj.keys()]
    new_proj  = {}
    for name in names:
        numbs   =  proj[name]
        length  =  len(numbs)
        newl    =  np.array_split(np.array(numbs), length // 10)
        newl    =  [list(newl[i]) for i in range(len(newl))]
        new_name = [name + 'sub_' + str(k) for k in range(len(newl))]
        for i,n in enumerate(new_name):
            l = newl[i]
            new_proj.update({n:l})
    return new_proj


def read_fatfile(file):
    v = np.genfromtxt(file)
    with open(file, 'r') as f:
        count = 0
        for l in f:
            if count == 1:
                ne,ns,nk = l.split()[4:]
                ne = int(ne)
                nk = int(nk)
                ns = int(ns)
                
                break
            count += 1
            
    return v, ne,ns, nk

def read_fat(D):
    files = os.listdir(D)
    files = [f for f in files if '.dat' in f]
    #print(files)
    blocks = [f.split('sub')[0] for f in files]
    blocks = unique_list(blocks)
    blocks_inds = [[] for i in blocks]
    for i,f in enumerate(files):
        p = f.split('sub')[0]
        blocks_inds[blocks.index(p)] += [i]
    vals = [read_fatfile( D + '/' + f )[0] for f in files]
    Res = []
    #print(blocks, blocks_inds)
    for ib in blocks_inds:
        res = sum([vals[i][:,2] for i in ib])
        Res += [[vals[ib[0]][:,0:2], res]]
    return blocks,Res

def interpolate_and_fft(x,y,N):
    from scipy.interpolate import interp1d
    xx = np.linspace(x.min(), x.max(), N)
    f  = interp1d(x,y)
    yy = f(xx)
    yy = np.pad(yy,(N//2, N//2))
    dx = xx[1] - xx[0]
    return np.fft.rfft(yy), np.fft.rfftfreq(2*N, dx)

def pyinds2siesta(idx):
    return listinds_to_string(numpy_inds_to_string(idx+1))

def get_btd_partition_of_matrix(_csr, start):
    from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm
    from scipy import sparse as sp
    assert _csr.shape[0] == _csr.shape[1]
    p = rcm(_csr)
    no = _csr.shape[0]
    csr = _csr[p,:][:,p]
    Part = [0, start]
    x0   = Part[-1]
    while x0<no:
        #print(x0)
        sl = slice(Part[-2], Part[-1])
        coup = csr[sl,sl.stop:]
        i,j,v = sp.find(coup)
        if len(i)==0:
            break
        else:
            x0 = np.max(j)+1
            Part += [Part[-1] + x0]
    btd = [Part[i+1] - Part[i] for i in range(len(Part)-1)]
    return p, btd, Part

def zigzag_g(bond = 1.42):
    #import sisl
    
    g = sisl.geom.graphene(orthogonal = True, bond=bond).rotate(90,[0,0,1])
    cell= g.cell.copy()
    g.cell[0] = cell[1]
    g.cell[1] = cell[0]
    g.cell[0]  *= -1
    g.xyz[:,0] *= -1
    return g

def num_neq_contour_points(Chem_Pot, kT):
    emin = min(Chem_Pot) - 5 * kT
    emax = max(Chem_Pot) + 5 * kT
    return np.arange(emin, emax, 0.01).shape[0]-1

def readH(file):
    return sisl.get_sile(file).read_hamiltonian()

def check_distances(r, cell = None, tol = 0.3):
    overlapping_atoms = False
    if cell is None:
        dij = np.linalg.norm(r[:,None,:] - r[None,:,:],axis=2)
        if (np.sum(dij<tol, axis=1)>1).any():
            overlapping_atoms = True
    else:
        r33 = np.zeros((len(r)*9, 3))
        count = 0
        na = len(r)
        for i in range(-1,2):
            for j in range(-1,2):
                r33[count * na : (count + 1)*na, :] = r + cell[0]*i +cell[1]*j
        dij = np.linalg.norm(r[:,None,:] - r[None,:,:],axis=2)
        if (np.sum(dij<tol, axis=1)>1).any():
            overlapping_atoms = True
    return overlapping_atoms

def get_twist(geom,s1,s2):
    v1 = geom.xyz[s1[1]]- geom.xyz[s1[0]]
    v2 = geom.xyz[s1[1]]- geom.xyz[s1[0]]
    v1 *= 1/np.linalg.norm(v1)
    v2 *= 1/np.linalg.norm(v2)
    return np.arccos(v1.dot(v2))




def write_dftb_hsd(object, Scc = 'Yes', write_cell = True, angmom_dic = {},
                   ReadInitialCharges = 'No', WriteChargesAsText = 'No', ReadChargesAsText = 'No',
                   WriteRealHS='No', CalculateForces='No', SkipChargeTest='No',
                   ThirdOrderFull = 'No', HubDeriv=None,DampCor=None):
    
    ua = unique_list(object.s)
    na = str(len(object.s))
    una= str(len(ua))
    bohr2A = 1.0#0.529177249
    A2bohr = 1.0#1 / bohr2A
    atomnr_to_label = {}
    it = 0
    for atomnr in ua:
        atomnr_to_label.update({str(ua[it]):str(it+1)})
        it+=1
    
    s = '  '
    if 'dftb_in.hsd' in os.listdir(object.dir):
        os.system('mv ' + object.dir+'/dftb_in.hsd '+object.dir+'/dftb_in.old')
    
    with open(object.dir+'/dftb_in.hsd','w') as f:
        GeomBlock  = 'Geometry = GenFormat{ \n  '+na+' S\n'
        for i in range(len(ua)):
            GeomBlock += s + Num2Sym[ua[i]]+' ' 
        GeomBlock += '\n'
        for i in range(len(object.s)):
            ri = np.round(object.pos_real_space[i] * A2bohr, 7)
            GeomBlock += s+str(i+1)+' ' + atomnr_to_label[str(object.s[i])]
            GeomBlock += ' '+str(ri[0]) + ' ' + str(ri[1]) + ' ' + str(ri[2])
            GeomBlock += '\n'
        if write_cell:
            GeomBlock += s+'0.0          0.0            0.0 \n'
            for i in range(3):
                GeomBlock+='  '
                for j in range(3):
                    GeomBlock+='  '+str(np.round(object.lat[i,j]*A2bohr,5))
                GeomBlock += '\n'
        GeomBlock+='}\n\n'
        
        HamBlock  = 'Hamiltonian = DFTB{\n'
        HamBlock += s+'Scc = ' +Scc +    '\n'
        HamBlock += s+'MaxSccIterations = '+str(object.max_scf_it) + '\n'
        HamBlock += s+'SccTolerance = '+str(object.dm_tol) + '\n'
        HamBlock += s+'SlaterKosterFiles =  Type2FileNames {\n'
        HamBlock += s+'Prefix = '+object.pp_path + '\n'
        HamBlock += s+'Separator = \"-\"\n'
        HamBlock += s+'suffix = .skf\n'
        HamBlock += s+'}\n'
        HamBlock += s+'MaxAngularMomentum {'
        for ia in ua:
            sym  = Num2Sym[ia]
            HamBlock += s+'  '+sym+' = \"'+angmom_dic[sym]+'\"\n'
        HamBlock +=s+'}\n\n'
        if object.kp is not None:
            HamBlock +=s+'KPointsAndWeights = SuperCellFolding {\n'
            HamBlock += s + '   '+str(object.kp[0])+' 0 0 \n'
            HamBlock += s + '   0 '+str(object.kp[1])+' 0 \n'
            HamBlock += s + '   0 0 ' + str(object.kp[2]) + '\n'
            HamBlock += s + '   0.5 0.5 0.5'
            HamBlock += s + '}\n'
        HamBlock += s + 'Filling = Fermi {\n'
        HamBlock += s + 'Temperature[Kelvin] = '+str(np.round(object.electronic_temperature_mev /(1000*8.6173 * 10**-5))) + '}\n\n'
        HamBlock += s + 'ReadInitialCharges = ' + ReadInitialCharges + '\n'
        if ThirdOrderFull=='Yes':
            HamBlock += s + 'ThirdOrderFull = Yes\n'
        if DampCor is not None:
            HamBlock += s + 'HCorrection = Damping {Exponent = '+str(DampCor)+' }\n'
        if HubDeriv is not None:
            HubBlock = s+'HubbardDerivs {\n'
            for k in HubDeriv.keys():
                HubBlock += 2*s + k + ' = ' + HubDeriv[k] + '\n'
            HubBlock += s+'}\n'
            HamBlock += HubBlock
        HamBlock += '}'
        
        OptBlock  = '\n\nOptions{\n'
        if WriteRealHS != 'No':
            OptBlock += s + 'WriteRealHS        = ' + WriteRealHS        + '\n'
        if WriteChargesAsText != 'No':
            OptBlock += s + 'WriteChargesAsText = ' + WriteChargesAsText + '\n'
        if ReadChargesAsText  != 'No':
            OptBlock += s + 'ReadChargesAsText  = ' + ReadChargesAsText  + '\n'
        if SkipChargeTest != 'No':
            OptBlock += s + 'SkipChargeTest     = ' + SkipChargeTest     + '\n'
        OptBlock += '}\n'
        
        f.write(GeomBlock + HamBlock + OptBlock)
        
        
        
        
        
        
        
    
        
        
        
    
