#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:59:30 2020

@author: aleks
"""


import numpy as np
import os
import matplotlib.pyplot as plt

ls = os.listdir

Num2Sym={1:'H', 2: 'He', 3:'Li',
         4:'Be',5: 'B', 6:'C',
         7:'N', 8:'O', 9:'F',
         10:'Ne', 
         14:'Si',
         16:'S',
         42:'Mo', 
         79:'Au'}

Sym2Num = {v: k for k, v in Num2Sym.items()}

def symbol_dic(x):
    if x=='GAMMA':
        return '\GAMMA'
    else:
        return x


def read_geom_from_fdf(file):
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

def R(x,n=6):
    return str(np.round(x,n))[:n+2]

def write_fdf(object,eta = 1e-2):
    ua = unique_list(object.s)
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
        f.write('SpinPolarized      '+ object.spin_pol+ '\n')
        f.write('PAO.BasisSize      '+ object.basis+    '\n')
        if object.write_matrices==True:
            f.write('TS.HS.Save True\n')
        f.write('SaveElectrostaticPotential T\n')
        f.write('SaveTotalPotential True\n')
        f.write('SCF.Mixer.Method '+object.mixer+'\n')
        f.write('SCF.Mixer.History '+str(object.n_hist)+'\n')
        f.write('SCF.Mix ' + str(object.mix_what) + '\n')
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
        f.write('MullikenInSCF T\nWriteMullikenPop 1\n')
        
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
                    if kp[2]==0:
                        bz_path_names+=[i[0]]
                        bz_path+=[object.sym_dic[i[0]]]
                        bz_path_num +=[50]
                if object.TwoDim == False:
                    bz_path_names+=[i[0]]
                    bz_path+=[i]
                    bz_path_num +=[200]
        
        if object.calc_bands ==True:
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
    
    with open(object.dir+'/TS_TBT.fdf','w') as f:
        if len(object.buffer_atoms)==0:
            pass
        elif isinstance(object.buffer_atoms, np.ndarray) or isinstance(object.buffer_atoms, list):
            f.write('TS.Atoms.Buffer ' +  listinds_to_string(numpy_inds_to_string(np.array(object.buffer_atoms) + 1)) +'\n')
        elif isinstance(object.buffer_atoms,str):
            f.write(object.buffer_atoms)
        
        # elif len(object.buffer_atoms)==2:
        #     f.write('TS.Atoms.Buffer ['+str(object.buffer_atoms[0])+' -- '+str(object.buffer_atoms[1])+']\n')
        
        if len(object.elecs)>0:
            
            f.write('TBT.T.Bulk True\n')
            f.write('TBT.DOS.Elecs True\n')
            f.write('TBT.DOS.A.All True\n')            
            #f.write('TBT.DM.Gf True\n')
            #f.write('TBT.DM.A True\n')
            
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
            for e in object.elecs:
                f.write('  '+xyz[it]+'\n')
                it+=1
            f.write('%endblock TS.ChemPots\n')
            
            it=0
            for e in object.elecs:
                let=xyz[it]
                block='TS.ChemPot.'+ let
                f.write('%block '+block+'\n')
                f.write('  mu '+str(object.Chem_Pot[it])+' eV\n')
                f.write('  contour.eq\n')
                f.write('    begin\n')
                f.write('      C-'+let+'\n')
                f.write('      T-'+let+'\n')
                f.write('    end\n')
                f.write('%endblock '+block+'\n')
                it+=1
            
            it=0
            for e in object.elecs:
                let=xyz[it]
                f.write('%block TS.Contour.C-'+let+'\n')
                f.write('part circle\n')
                f.write('   from '+str(object.Chem_Pot[it])+' eV -40. eV to '+str(object.Chem_Pot[it])+' eV -10 kT \n')
                f.write('     points 25\n')
                f.write('      method g-legendre\n')
                f.write('%endblock\n')
                
                f.write('%block TS.Contour.T-'+let+'\n')
                f.write('part tail\n')
                f.write('   from prev to inf\n')
                f.write('     points 10\n')
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
                if e.elec_bloch is not None:
                    f.write('   ' + 'bloch ' +str(e.elec_bloch[0]) +' ' + str(e.elec_bloch[1]) +' ' + str(e.elec_bloch[2]) +' \n')
                f.write('   chemical-potential '+ xyz[it]+'\n')
                f.write('   semi-inf-direction '+e.semi_inf+'\n')
                if e.elec_RSSE == True:
                    f.write('   out-of-core True\n')
                    f.write('   Gf ../'+e.dir+'/'+e.sl+'.TBTGF\n')
                    f.write('   Gf.Reuse True\n')
                f.write('   electrode-position '+str(object.elec_inds[it][0]+1)+'\n')
                f.write('%endblock TS.Elec.'+e.sl+'\n')
                it+=1
        
        n_ooc = sum([e.elec_RSSE for e in object.elecs])
        f.write('\n')
        if len(object.elecs)==0 or n_ooc == 0:
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
            
def write_relax_fdf(object, Constraints, force_tol = 0.01, max_it = 1000):
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
        f.write('SpinPolarized      '+ object.spin_pol+ '\n')
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
                    if kp[2]==0:
                        bz_path_names+=[i[0]]
                        bz_path+=[object.sym_dic[i[0]]]
                        bz_path_num +=[50]
                if object.TwoDim == False:
                    bz_path_names+=[i[0]]
                    bz_path+=[i]
                    bz_path_num +=[50]
        
        if object.calc_bands ==True:
            f.write('\nBandLinesScale   ReciprocalLatticeVectors\n')
            f.write('%block BandLines\n')
            it=0
            for i in bz_path_names:
                f.write(str(bz_path_num[it])+ ' ' + str(bz_path[it][0]) + '  ' + str(bz_path[it][1]) + '  ' +str(bz_path[it][2]) + '  ' + symbol_dic(bz_path_names[it])+'\n')
                it+=1
            f.write('%endblock BandLines\n')
    force_tol = str(force_tol)
    with open(object.dir + '/MD.fdf','w') as f:
        f.write('MD.TypeOfRun          	CG  \n')
        f.write('MD.NumCGsteps     	'+str(max_it)+ '\n')
        f.write('MD.MaxForceTol 	        '+ force_tol +' eV/Ang \n') 
        f.write('MD.VariableCell      	F\n')
        # f.write('MD.MaxStressTol         0.0010 eV/Ang\n')
        f.write('WriteMDHistory      	T \n')
        f.write('WriteMDXMol  		T\n')
        f.write('%block Geometry.Constraints\n')
        for line in Constraints:
            f.write('    ' + line + ' \n')
        f.write('%endblock\n')
    



def write_gin(object,cell_opti=[],fix=[],relax_only = False):
    with open(object.dir+'/Gulp.gin','w') as f:
        ##Fra Mads##
        if relax_only == True:
            f.write('opti dist full nosymmetry\n')
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
        # if 'TS.BTD.Pivot' in line :
        #     temp=[]
        #     m=0
        #     for lines in L:
        #         if n<=m<n+10:
        #             temp+=[lines]
        #         m+=1
        #         if m==10:
        #             Methods+=[temp]
        n+=1
        if 'Minimum memory required pivoting scheme:' in line:
            WRITE = L[n][2:]#line[2:]
            break
    
    # RAM=np.full(len(Methods),np.nan)
    # l=[]
    # i=0
    # for m in Methods:
    #     for l in m:
    #         if 'Rough estimation of MEMORY' in l:
    #             temp=l[33:48]
    #             while 'G' in temp:
    #                 temp=temp[0:len(temp)-1]
    #             RAM[i] = float(temp)
    #     i+=1
    # i=np.where(RAM==RAM.min())[0][0]
    # WRITE=Methods[i]
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

def read_siesta_relax(fname):
    pos = []
    lattice = []
    read_species = False
    read_lattice = False
    read_pos     = False
    species_dict = {}
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
    
    # return out    
    
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




def plotamok(Directory):
    import sisl
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


def sort_mulliken(Vals, Q_tot, NA, BASIS):
    dic = {'SZ':   4,
           'SZP':  9,
           'DZP': 12}
    NB = [dic[b] for b in BASIS]
    
    
    
    
    
    
    
    
    


Ost = ['Havarti', 
       'Parmesan',
       'Hullet ost',
       'Gedeost',
       'Blåskimmelost',
       'Cheddar',
       'Emmentaler',
       'Feta',
       'Gouda',
       'Mozzarella',
       ]
