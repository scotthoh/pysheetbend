from __future__ import print_function
import numpy as np
from timeit import default_timer as timer
#'''
from TEMPy.Quaternion import Quaternion
from TEMPy.ProtRep_Biopy import BioPy_Structure
'''
from TEMPy.math.quaternion import Quaternion
from TEMPy.protein.prot_rep_biopy import BioPy_Structure
'''
def RTop_orth(tgtArr, srcArr, weighted=False):
    '''
    Implemented in Clipper->coords.cpp, recoded here as independent code.
    Construct the operator which give the least-squares fit of one set of
    coordinates onto another. The coordinates are stored in an array.
    The array must the same size. Each atoms in source and target must
    correspond to each other. Algorithm reference: Kearsley S.K. (1989)
    'On the orthogonal transformation used for structural comparisons'.
    Acta Cryst. A45, 208-210.
    Arguments:  
      *srcArr*
        source atomList (BioPy_Structure)
      *tgtArr*
        target atomList (BioPy_Structure)
      *weighted*
        boolean to apply weight (atom mass) to calculation of centre of mass
        (default=False)
    Return:
      Rotation (3x3 matrix) and translation (x,y,z vector) operators
    '''
    debug = False
    # check size
    if len(srcArr) != len(tgtArr):
        raise ValueError('RTop_orth: coordinate list size mismatch!')
    # srcArr = BioPy_Structure()
    # get centre of mass
    n = float(len(srcArr))
    src_cen = np.zeros(3)
    tgt_cen = np.zeros(3)
    src_coord = np.array(srcArr.get_pos_mass_list()[:, :3])
    tgt_coord = np.array(tgtArr.get_pos_mass_list()[:, :3])
    src_cen = src_cen + np.sum(src_coord, axis=0)
    tgt_cen = tgt_cen + np.sum(tgt_coord, axis=0)
    src_cen = src_cen/n
    tgt_cen = tgt_cen/n
    # print('src_cen', src_cen)
    # print('tgt_cen', tgt_cen)
    # prepare cross sums
    mat = np.zeros((4, 4))
    s = src_coord - src_cen
    t = tgt_coord - tgt_cen
    p = s + t
    m = s - t

    mat[0,  0] = np.sum(np.square(m))
    mat[1,  1] = np.sum(np.square(m[:, 0])) + np.sum(np.square(p[:, 1:]))
    mat[2,  2] = np.sum(np.square(m[:, 1])) + np.sum(np.square(p[:, 0::2]))
    mat[3,  3] = np.sum(np.square(m[:, 2])) + np.sum(np.square(p[:, :2]))
    mat[0,  1] = np.sum(m[:, 2]*p[:, 1] - m[:, 1]*p[:, 2])
    mat[1,  0] = mat[0, 1]
    mat[0,  2] = np.sum(m[:, 0]*p[:, 2] - m[:, 2]*p[:, 0])
    mat[2,  0] = mat[0, 2]
    mat[0,  3] = np.sum(m[:, 1]*p[:, 0] - m[:, 0]*p[:, 1])
    mat[3,  0] = mat[0, 3]
    mat[1,  2] = np.sum(m[:, 0]*m[:, 1] - p[:, 0]*p[:, 1])
    mat[2,  1] = mat[1, 2]
    mat[1,  3] = np.sum(m[:, 0]*m[:, 2] - p[:, 0]*p[:, 2])
    mat[3,  1] = mat[1, 3]
    mat[2,  3] = np.sum(m[:, 1]*m[:, 2] - p[:, 1]*p[:, 2])
    mat[3,  2] = mat[2, 3]

    # calculate eigenvalue
    eval, evec = np.linalg.eigh(mat)
    if debug:
        print('mat ', mat)
        print('eval ', eval)
        print('evev ', evec)
    rot_mat = Quaternion([evec[0, 0], evec[1, 0], evec[2, 0],
                         evec[3, 0]]).to_rotation_matrix()

    trans_op = tgt_cen - (rot_mat.dot(src_cen))   
    return rot_mat, trans_op


class Pseudoregularize:
    '''
    Pseudoregulariser
    '''
    def __init__(self, atomList):
        self.crad = 2.4
        self.atomList = atomList.copy()
        self.chainIDs = self.atomList.get_chain_list() # list of chain ID
        self.fragsList = [] # list of fragments of atoms, len(fragsList)=number of fragments
        #self.fragsResRange = [] # listjhq ws of [start,end] res no for each fragment
        #self.frag_ref_CA = []
        stime = timer()
        # make a list of fragments with connected residues/atoms
        for c in self.chainIDs:
            temp_chn = atomList.get_chain(c) # atoms in chain
            start_res = temp_chn[0].res_no
            end_res = temp_chn[-1].res_no
            frag_start = temp_chn[0].res_no
            #print(start_res, end_res)
            iscon = False
            for a in range(start_res, end_res+1): # loop through residues in chain
                frag0 = temp_chn.get_selection(a, a, chain=c)
                if not frag0:  # check for gaps in between residues
                    if a == (end_res - 1):
                        if temp_chn.get_residue(end_res):
                            frag_start = end_res
                            frag_end = end_res
                            iscon = False
                    else:
                        continue
                elif not iscon:
                    frag_start = a
                iscon = False
                a1 = a + 1
                if a1 <= end_res:
                    frag1 = temp_chn.get_selection(a1, a1, chain=c)
                    if frag1:
                        # check if residues a, a1 are connected
                        for atm in frag0:
                            for atm1 in frag1:
                                if atm.distance_from_atom(atm1) < self.crad:
                                    iscon = True
                                    frag_end = a1
                    if not iscon:
                        frag_end = a
                else:
                    frag_end = a
                if not iscon:  # add continuous fragment to fragList
                    #print(frag_start, frag_end)
                    self.fragsList.append(temp_chn.get_selection(frag_start,
                                                                 frag_end,
                                                                 chain=c))
                    #self.fragsResRange.append((frag_start, frag_end))
                    #self.frag_ref_CA.append(self.fragsList[-1].get_CAonly())
        #print(len(self.fragsList))
        #self.check_fraglist()
        endtime = timer()
        print('make frags time : {0}'.format(endtime-stime))
        #for f in self.fragsList:
        #   print(f)
        '''
            self.chainList.append(temp_chn)
            tmp_resList = []
            for x in temp_chn:
            if x.res_no not in tmp_resList:
                tmp_resList.append(x.res_no) # res num/ID in chain
            frag_resrange = []  #[start_no,end_no]
            for i in tmp_resList:
            frag0 = tmp_resList.get_selection(i, i, chain=c)
            if frag0:
                if len(frag_resrange) == 0:
                frag_resrange.append(i)
            self.resList.append(tmp_resList[:]) # res num/ID in every chain
        
        # separate disconnected fragments, < crad
        for c in range(len(self.chainList)):
            #cfragsList = []
            frag_reslist = []
            for i in self.resList[c]: # range(c[0].res_no, c[-1].res_no+1):
            frag0 = self.chainList[c].get_selection(i, i, chain=self.chainList[c][0].chain)
            if frag0:
                frag_reslist.append(i)
            iscon = False
            if i+1 in self.resList[c]: 
                frag1 = self.chainList[c].get_selection(i+1, i+1, chain=self.chainList[c][0].chain)
                if frag1:
                for atm in frag0:
                    for atm1 in frag1:
                    if atm.distance_from_atom(atm1) < self.crad:
                        iscon = True
            if not iscon:
                #cfragsList.append(frag_reslist[:])
                # list of fragments along with chain ID
                self.fragsList.append((frag_reslist[:], self.chainList[c][0].chain))
                del frag_reslist[:]
            #self.fragsList.append(cfragsList[:])
            #del cfragsList[:]
        '''

    def get_fraglist(self):
        return self.fragsList

    def check_fraglist(self):
        for x in self.fragsList:
            print(len(x))
            print(x)

    def regularize_frag(self, mol_work):
        new_molwork = []
        #self.check_fraglist()
        for x in self.fragsList: #range(len(self.fragsList)): # list of fragments of atoms
            # get fragments from work molecule and reference molecule
            frag_ref = x.copy()
            frag_work = mol_work.get_selection(x[0].res_no, x[-1].res_no, chain=x[0].chain)
            #print(f'frag_ref len : {len(frag_ref)}')
            #print(f'frag_work len : {len(frag_work)}')

            #frag_ref = self.atomList.get_selection(self.fragsList[x][0][0], self.fragsList[x][0][-1], chain=self.fragsList[x][1])

            # find key atoms coord, use to determine per atom weights
            #frag_ref_CA = []
            if frag_ref[0].record_name != 'HETATM':
                frag_ref_CA = frag_ref.get_CAonly()  # keys
            else:
                backbonelist = []
                for atm in frag_ref.atomList:
                    if atm.get_name() == 'C1':
                        print(atm)
                        backbonelist.append(atm.copy())
                if len(backbonelist) != 0:
                    frag_ref_CA = BioPy_Structure(backbonelist[:])
                else:
                    backbonelist.append(frag_ref.atomList[0].copy())
                    frag_ref_CA = BioPy_Structure(backbonelist[:])
            # get start/end atom index for each residue in fragment
            curr_res = frag_ref[0].res_no
            frag_ref_resIndex = [0]
            for i in range(len(frag_ref)):
                if frag_ref[i].res_no != curr_res:
                    frag_ref_resIndex.append(i)
                    curr_res = frag_ref[i].res_no
            frag_ref_resIndex.append(len(frag_ref)) # the last one to mark end of last residue
            #print(f'frag_ref_CA len : {len(frag_ref_CA)}')
            #print(f'{frag_ref_CA}')
            # make table of distances by atom
            w0 = []
            w1 = []
            w2 = []
            for r in range(len(frag_ref_CA)): # each key represent each residue
                d0 = 1.0e6
                d2 = 1.0e6
                if r-1 >= 0:
                    d0 = frag_ref_CA[r].distance_from_atom(frag_ref_CA[r-1])
                if r+1 < len(frag_ref_CA):
                    d2 = frag_ref_CA[r].distance_from_atom(frag_ref_CA[r+1])
                r_atms = frag_ref.get_selection(frag_ref_CA[r].res_no, frag_ref_CA[r].res_no)
                for a in r_atms:  # loop atoms in current residue
                    r0 = 1.0e6
                    r2 = 1.0e6
                    if r-1 >= 0:
                        r0 = a.distance_from_atom(frag_ref_CA[r-1])
                    if r+1 < len(frag_ref_CA):
                        r2 = a.distance_from_atom(frag_ref_CA[r+1])
                    w00 = min((1.0 - r0/d0), 0.5)
                    w02 = min(1.0 - r2/d2, 0.5)
                    w01 = 1.0 - (w00 + w02)
                    w0.append(w00)
                    w1.append(w01)
                    w2.append(w02)
            # end looping through residues
            # now superpose a list of trimonomer fragments (tri or penta?)
            f0 =  frag_work.copy()
            f1 =  frag_work.copy()
            f2 =  frag_work.copy()
            dr = 2
            #print('rotmat')
            for r in range(len(frag_ref_CA)):
                if r<2:
                    slwork = []
                    slref = []
                    for r1 in range(dr-r):
                        r0_no = frag_ref_CA[0].res_no
                        slwork.append(frag_work.get_selection(r0_no,
                                                              r0_no))
                        slref.append(frag_ref.get_selection(r0_no,
                                                            r0_no))
                    r2_end = max(min(r+dr, len(frag_ref_CA)-1), 0)
                    r2e_resno = frag_ref_CA[r2_end].res_no
                    slwork.append(frag_work.get_selection(0, r2e_resno))
                    slref.append(frag_ref.get_selection(0, r2e_resno))
                    co1 = frag_work.combine_SSE_structures(slwork)
                    co2 = frag_ref.combine_SSE_structures(slref)
                elif (r+dr) >= len(frag_ref_CA):
                    slwork = []
                    slref = []
                    r2_start = max(min(r-dr, len(frag_ref_CA)-1), 0)
                    r2_end = len(frag_ref_CA)-1
                    r2s_resno = frag_ref_CA[r2_start].res_no
                    r2e_resno = frag_ref_CA[r2_end].res_no
                    slwork.append(frag_work.get_selection(r2s_resno,
                                                          r2e_resno))
                    slref.append(frag_ref.get_selection(r2s_resno, r2e_resno))
                    for r1 in range(r+dr-r2_end):
                        slwork.append(frag_work.get_selection(r2e_resno,
                                                              r2e_resno))
                        slref.append(frag_ref.get_selection(r2e_resno,
                                                            r2e_resno))
                    co1 = frag_work.combine_SSE_structures(slwork)
                    co2 = frag_ref.combine_SSE_structures(slref)
                else:
                    r2_start = max(min(r-dr, len(frag_ref_CA)-1), 0)
                    r2_end = max(min(r+dr, len(frag_ref_CA)-1), 0)
                    r2s_resno = frag_ref_CA[r2_start].res_no
                    r2e_resno = frag_ref_CA[r2_end].res_no
                    co1 = frag_work.get_selection(r2s_resno, r2e_resno)
                    co2 = frag_ref.get_selection(r2s_resno, r2e_resno)

                rot, trans = RTop_orth(co1, co2)  #(tgt, src)
                #print(r, rot, trans)
                #print(co1)
                #print(co2)

                # rebuild
                r2 = max(r-1, 0)
                r0 = min(r+1, len(frag_ref_CA)-1)
                #r2_no = frag_ref_CA[r2].res_no
                #r_no = frag_ref_CA[r].res_no
                #r0_no = frag_ref_CA[r0].res_no
                for a in range(frag_ref_resIndex[r2], frag_ref_resIndex[r2+1]):
                    f2[a].set_x(frag_ref[a].get_x())
                    f2[a].set_y(frag_ref[a].get_y())
                    f2[a].set_z(frag_ref[a].get_z())
                    f2[a].matrix_transform(rot)
                    f2[a].translate(trans[0,0], trans[0,1], trans[0,2])
                    #tmp_co = np.array(frag_ref[a].get_pos_mass()[:3])
                    #tmp_co = rot.dot(tmp_co)
                    #tmp_co = tmp_co + trans #[tmp_co[0,0] + trans[0], tmp_co[0,1] + trans[1], tmp_co[0,2] + trans[2]]
                    #f2[a].set_x(tmp_co[0,0])
                    #f2[a].set_y(tmp_co[0,1])
                    #f2[a].set_z(tmp_co[0,2])
                    #frag_ref[a].translate(trans[0,0], trans[0,1], trans[0,2])
                    #f2[a].set_x(frag)
                for a in range(frag_ref_resIndex[r], frag_ref_resIndex[r+1]):
                    f1[a].set_x(frag_ref[a].get_x())
                    f1[a].set_y(frag_ref[a].get_y())
                    f1[a].set_z(frag_ref[a].get_z())
                    f1[a].matrix_transform(rot)
                    f1[a].translate(trans[0,0], trans[0,1], trans[0,2])
                    #tmp_co = np.array(frag_ref[a].get_pos_mass()[:3])
                    #tmp_co = rot.dot(tmp_co)
                    #tmp_co = tmp_co + trans #[tmp_co[0,0] + trans[0], tmp_co[0,1] + trans[1], tmp_co[0,2] + trans[2]]
                    #f1[a].set_x(tmp_co[0,0])
                    #f1[a].set_y(tmp_co[0,1])
                    #f1[a].set_z(tmp_co[0,2])

                for a in range(frag_ref_resIndex[r0], frag_ref_resIndex[r0+1]):
                    f0[a].set_x(frag_ref[a].get_x())
                    f0[a].set_y(frag_ref[a].get_y())
                    f0[a].set_z(frag_ref[a].get_z())
                    f0[a].matrix_transform(rot)
                    f0[a].translate(trans[0,0], trans[0,1], trans[0,2])
                    #tmp_co = np.array(frag_ref[a].get_pos_mass()[:3])
                    #tmp_co = rot.dot(tmp_co)
                    #tmp_co = tmp_co + trans #[tmp_co[0,0] + trans[0], tmp_co[0,1] + trans[1], tmp_co[0,2] + trans[2]]
                    #f2[a].set_x(tmp_co[0,0])
                    #f2[a].set_y(tmp_co[0,1])
                    #f2[a].set_z(tmp_co[0,2])

                #print(r)
                #print(co1)
                #print(co2)


            # make weighted combination
            #print(len(w0), len(w1), len(w2))
            #print(len(f0), len(f1), len(f2))
            #print(frag_ref)
            #print(frag_work)
            #print('debug')
            for a in range(len(frag_work)):
                newpos0 = w0[a]*np.array(f0[a].get_pos_mass()[:3]) #+ \
                newpos1 = w1[a]*np.array(f1[a].get_pos_mass()[:3]) #+ \
                newpos2 = w2[a]*np.array(f2[a].get_pos_mass()[:3])
                #print(f0[a], w0[a])
                #print(f1[a], w1[a])
                #print(f2[a], w2[a])
                #print(newpos0, newpos1, newpos2)
                newpos = newpos0 + newpos1 + newpos2
                frag_work[a].set_x(newpos[0])
                frag_work[a].set_y(newpos[1])
                frag_work[a].set_z(newpos[2])


                #rt = get_RTop() # need to code this based on clipper RTop_orth(v,v), least square with quarternion to get eigenval and eigenvecs
                # can utilise the quarternion class in TEMPy
                # use np.linalg.eig to solve eigeneq
            new_molwork = np.append(new_molwork, frag_work)

          # return
        return BioPy_Structure(new_molwork)


if __name__ == '__main__':
    from TEMPy.StructureParser import PDBParser
    ippdb = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1.ent'
    struct = PDBParser.read_PDB_file('5ni1', ippdb, hetatm=False, water=False)
    workpdb = '/home/swh514/Projects/testing_ground/shiftfield_python/testrun/check_FT/test_12Apr/testout_sheetbend1_withorthmat_final.pdb'
    workstruc = PDBParser.read_PDB_file('5ni1', workpdb, hetatm=False, water=False)

    pr = Pseudoregularize(struct)

    # test
    ref_A = struct.get_chain('A')
    work_A = workstruc.get_chain('A')

    rot, trans = RTop_orth(work_A, ref_A)


    #newmol = pr.regularize_frag(workstruc)

    #newmol.write_to_PDB('test_psedoreg_outfile_4.pdb')
    # 28 april working, but slight problem with eigenvec and quaternion arrangement from matrix
    # different from clipper
    # 30 april done. work fine same output as C++ shiftfield's pseudoregulariser
    # initial problem was applying rotation+translation to the wrong coord (suppose to be the initial(ori) coord)


    #fL = pr.get_fraglist()
    #print(fL)

    #for i in pr.fragsResRange:
    #  print(i)

    #pr.regularize_frag(struct)

    #PROBLEMS:
    #numbering not in the residues number also printed 142-200 - solved using get_selection
    #probably can use the atom.init_x,y,z to get do the weights and frags
    #need to think of a way to update the atoms position and also return the structure
    #-can just combine BioPyStruc of fragments, then combine, split chains, reorder




    #elastic neural network james krieger (2019) Neuroscie Lett 700 22-29
    # doruker et al Proteins 40 (2000)
    # atilgan et al Biophys J 80 (2001)
    # Bahar et al 2010, chem reviews 110 1463:1497

    #CryoDRGN NeRF, ECCV 2020
