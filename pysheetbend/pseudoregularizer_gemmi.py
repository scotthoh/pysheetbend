from __future__ import print_function
from typing import Union
import numpy as np
from timeit import default_timer as timer
from dataclasses import dataclass
import gemmi
from ccpem_pyutils.other.cluster import cluster_coord_features
from ccpem_pyutils.model.gemmi_model_utils import (
    GemmiModelUtils,
    get_residue_attribute,
    set_bfactor_attributes,
)


@dataclass
class ConnectedResidues:
    C: str  # chain name
    S: str  # seq id
    R: str  # res name


class Pseudoregularize:
    """
    Pseudoregularizer
    Pseudoregularize the target model against the reference/input model
    """

    def __init__(self, original_structure: gemmi.Structure, model_number=0, verbose=0):
        """
        make a list of fragments of contiguous residues from the input structure
        Arguments:
            original_structure: GEMMI structure instance
            model_number: model number in structure instance, default: 0, first model
            verbose: verbosity
        """
        start = timer()
        self.mol_ref = original_structure
        self.model_number = model_number
        self.gemmimodelutils = GemmiModelUtils(original_structure)
        self.crad = 2.4
        self.verbose = verbose
        # list of fragments of atoms, len(fragsList) = number of fragments
        self.fragsList = []
        self.frags_cluster = []
        fragments_obtained = self.get_fragment_by_chains()
        if not fragments_obtained:
            print("ERROR: List containing fragments is empty!")
            exit()

        endtime = timer()
        if self.verbose >= 10:
            print("Pseudoreg Init Time : {0:.4f}".format(endtime - start))

    def get_fragment_by_chains(self):
        """
        Get fragments from chains from given structure into a list
        """

        for c in range(len(self.mol_ref[self.model_number])):
            frag = []
            chain = self.mol_ref[self.model_number][c]
            for r in range(len(chain)):
                residue = chain[r]
                # add residue to fragsList
                frag.append(
                    ConnectedResidues(chain.name, str(residue.seqid), residue.name)
                )
                iscon = False
                # check if next residue is connected to current one
                if (r + 1) < len(chain):
                    next_res = chain[r + 1]
                    for atom1 in residue:
                        for atom2 in next_res:
                            if atom1.pos.dist(atom2.pos) < self.crad:
                                iscon = True
                if not iscon:
                    if len(frag) != 0:
                        self.fragsList.append(frag)
                    frag = []
        if len(self.fragsList) != 0:
            return True
        else:
            return False
        # return self.fragsList

    def check_fraglist(self, len_only=True):
        """
        For debugging purposes, print out the length of each fragment
        """
        for x in self.fragsList:
            print(len(x))
            if not len_only:
                print(x)

    @staticmethod
    def eigen_to_rotation_matrix(w, x, y, z):
        rot_m = np.zeros((3, 3))
        rot_m[0, 0] = w * w + x * x - y * y - z * z
        rot_m[0, 1] = 2 * x * y - 2 * w * z
        rot_m[0, 2] = 2 * w * y + 2 * x * z
        rot_m[1, 0] = 2 * x * y + 2 * w * z
        rot_m[1, 1] = w * w - x * x + y * y - z * z
        rot_m[1, 2] = 2 * y * z - 2 * w * x
        rot_m[2, 0] = 2 * x * z - 2 * w * y
        rot_m[2, 1] = 2 * w * x + 2 * y * z
        rot_m[2, 2] = w * w - x * x - y * y + z * z

        return rot_m

    @staticmethod
    def orthogonal_transformation(target, source, weighted=False):
        """
        Implemented in Clipper->coords.cpp, recoded here as independent code.
        Construct the operator which give the least-squares fit of one set of
        coordinates onto another. The coordinates are stored in an array.
        The array must the same size. Each atoms in source and target must
        correspond to each other. Algorithm reference: Kearsley S.K. (1989)
        'On the orthogonal transformation used for structural comparisons'.
        Acta Cryst. A45, 208-210.
        Arguments:
          *srcArr*
            source atoms positions
          *tgtArr*
            target atoms positions
          *weighted*
            boolean to apply weight (atom mass) to calculation of centre of mass
            (default=False)
        Return:
          Rotation (3x3 matrix) and translation (x,y,z vector) operators
        """
        debug = False
        # check size
        if len(target) != len(source):
            raise ValueError("ortho_transformation: coordinate list size mismatch!")

        n = float(len(source))
        src_cen = np.zeros(3)
        tgt_cen = np.zeros(3)
        src_cen = src_cen + np.sum(source, axis=0)
        tgt_cen = tgt_cen + np.sum(target, axis=0)
        src_cen = src_cen / n
        tgt_cen = tgt_cen / n
        mat = np.zeros((4, 4))
        s = source - src_cen
        t = target - tgt_cen
        p = s + t
        m = s - t

        mat[0, 0] = np.sum(np.square(m))
        mat[1, 1] = np.sum(np.square(m[:, 0])) + np.sum(np.square(p[:, 1:]))
        mat[2, 2] = np.sum(np.square(m[:, 1])) + np.sum(np.square(p[:, 0::2]))
        mat[3, 3] = np.sum(np.square(m[:, 2])) + np.sum(np.square(p[:, :2]))
        mat[0, 1] = np.sum(m[:, 2] * p[:, 1] - m[:, 1] * p[:, 2])
        mat[1, 0] = mat[0, 1]
        mat[0, 2] = np.sum(m[:, 0] * p[:, 2] - m[:, 2] * p[:, 0])
        mat[2, 0] = mat[0, 2]
        mat[0, 3] = np.sum(m[:, 1] * p[:, 0] - m[:, 0] * p[:, 1])
        mat[3, 0] = mat[0, 3]
        mat[1, 2] = np.sum(m[:, 0] * m[:, 1] - p[:, 0] * p[:, 1])
        mat[2, 1] = mat[1, 2]
        mat[1, 3] = np.sum(m[:, 0] * m[:, 2] - p[:, 0] * p[:, 2])
        mat[3, 1] = mat[1, 3]
        mat[2, 3] = np.sum(m[:, 1] * m[:, 2] - p[:, 1] * p[:, 2])
        mat[3, 2] = mat[2, 3]

        # calculate eigenvalue
        eval, evec = np.linalg.eigh(mat)
        if debug:
            print("mat ", mat)
            print("eval ", eval)
            print("evev ", evec)
        rot_mat = Pseudoregularize.eigen_to_rotation_matrix(
            evec[0, 0], evec[1, 0], evec[2, 0], evec[3, 0]
        )

        trans_op = tgt_cen - (rot_mat.dot(src_cen))
        tr = gemmi.Transform()
        tr.mat.fromlist(rot_mat)
        tr.vec.fromlist(trans_op)
        return tr

    def regularize_frag(self, mol_work, model_number=0, dbscan_cluster=False, cycle=0):
        # if dbscan_cluster:
        #    return self.regularize_frag_1(
        #        mol_work=mol_work,
        #        model_number=model_number,
        #        dbscan_cluster=dbscan_cluster,
        #        cycle=cycle,
        #    )
        # else:
        return self.regularize_frag_0(mol_work=mol_work, model_number=model_number)

    def regularize_frag_0(self, mol_work, model_number=0, dbscan_cluster=False):
        """
        Regularize target model against reference model
        Argument:
            mol_work: GEMMI structure, target work model to be regularized
            model_number: model number in structure instance, default: 0, first model
        """
        if dbscan_cluster:
            if len(self.frags_cluster) != 0:
                fraglist = self.frags_cluster
            else:
                return False
        else:
            fraglist = self.fragsList
        # loop of fragments and regularize
        for frag in fraglist:
            mp1 = []
            mp2 = []
            keys = []
            for CR in frag:
                # just to make sure the residues in mol_work is also in mol_ref
                try:
                    mp1.append(mol_work[model_number][CR.C][CR.S][CR.R])
                except IndexError:
                    continue
                mp2.append(self.mol_ref[model_number][CR.C][CR.S][CR.R])
                # find key atoms coord, use to determine per atom weights
                if mp2[-1].het_flag == "A":
                    atom = mp2[-1].find_atom("CA", "*")
                else:
                    atom = mp2[-1].find_atom("C1*", "*")
                if atom is None:
                    atom = mp2[-1][0]
                keys.append(atom.pos)
            if len(mp1) == 0:
                continue  # skip empty frag list
            if self.verbose >= 10:
                print(f"Keys len : {len(keys)}")
                print(f"mp1 len : {len(mp1)}")
                print(f"mp2 len : {len(mp2)}")
            # make table of distances by atom
            w0 = []
            w1 = []
            w2 = []
            # each key represent each residue
            # 6 OCT something wrong with the index out of range, is it use len(mp2)-1?
            for r in range(len(mp2)):
                d0 = 1.0e6
                d2 = 1.0e6
                if r - 1 >= 0:
                    d0 = keys[r].dist(keys[r - 1])
                if r + 1 < len(mp2):
                    d2 = keys[r].dist(keys[r + 1])
                w0tmp = []
                w1tmp = []
                w2tmp = []
                for atom in mp2[r]:
                    r0 = 1.0e6
                    r2 = 1.0e6
                    if r - 1 >= 0:
                        r0 = atom.pos.dist(keys[r - 1])
                    if r + 1 < len(mp2):
                        r2 = atom.pos.dist(keys[r + 1])
                    w00 = min((1.0 - r0 / d0), 0.5)
                    w02 = min((1.0 - r2 / d2), 0.5)
                    w01 = 1.0 - (w00 + w02)
                    w0tmp.append(w00)
                    w1tmp.append(w01)
                    w2tmp.append(w02)

                w0.append(w0tmp)
                w1.append(w1tmp)
                w2.append(w2tmp)

            # end looping through residues
            # now superpose a list of pentamer fragments (tri or penta?)
            f0 = mp1.copy()
            f1 = mp1.copy()
            f2 = mp1.copy()
            dr = 2
            for r in range(len(mp2)):
                co1 = []
                co2 = []
                for r1 in range(r - dr, r + dr + 1):
                    r2 = max(0, min(r1, len(mp2) - 1))
                    for a in range(len(mp1[r2])):
                        co1.append(mp1[r2][a].pos)  # .tolist())
                        co2.append(mp2[r2][a].pos)  # .tolist())
                # transform = Pseudoregularize.orthogonal_transformation(co1, co2)
                superpose = gemmi.superpose_positions(co1, co2)
                transform = superpose.transform
                # 5OCT continue from here
                # rebuild
                r2 = max(r - 1, 0)
                r0 = min(r + 1, len(mp2) - 1)

                for a in range(len(mp2[r2])):
                    f2[r2][a].pos = gemmi.Position(transform.apply(mp2[r2][a].pos))
                for a in range(len(mp2[r])):
                    f1[r][a].pos = gemmi.Position(transform.apply(mp2[r][a].pos))
                for a in range(len(mp2[r0])):
                    f0[r0][a].pos = gemmi.Position(transform.apply(mp2[r0][a].pos))
            # make weighted combination
            # print(len(w0), len(w1), len(w2))
            # print(len(f0), len(f1), len(f2))
            # print(frag_ref)
            # print(frag_work)
            # print('debug')
            for r in range(len(mp1)):
                for a in range(len(mp1[r])):
                    mp1[r][a].pos = (
                        w0[r][a] * f0[r][a].pos
                        + w1[r][a] * f1[r][a].pos
                        + w2[r][a] * f2[r][a].pos
                    )

            for r in range(len(frag)):
                CR = frag[r]
                # just to make sure the residues in mol_work is also in mol_ref
                try:
                    res = mol_work[model_number][CR.C][CR.S][CR.R]
                except IndexError:
                    continue
                for a in range(len(mp1[r])):
                    res[a].pos = mp1[r][a].pos
                    # mol_work[model_number][CR.C][CR.S][CR.R][a].pos = mp1[r][a].pos
            # new_molwork = np.append(new_molwork, frag_work)
        return True

    def regularize_frag_1(
        self, mol_work, model_number=0, dbscan_cluster=False, cycle=0
    ):
        # loop of fragments and regularize
        # weight using a spherical distance weight
        mol_copy = mol_work.clone()
        count = 0
        if len(self.frags_cluster) != 0:
            fraglist = self.frags_cluster
        else:
            fraglist = self.fragsList
        for frag in fraglist:
            mp1 = []
            mp2 = []
            keys = []
            com = gemmi.Position(0.0, 0.0, 0.0)
            for CR in frag:
                # just to make sure the residues in mol_work is also in mol_ref
                try:
                    mp1.append(mol_work[model_number][CR.C][CR.S][CR.R])
                except IndexError:
                    continue
                mp2.append(self.mol_ref[model_number][CR.C][CR.S][CR.R])
                # find key atoms coord, use to determine per atom weights
                if mp2[-1].het_flag == "A":
                    atom = mp2[-1].find_atom("CA", "*")
                else:
                    atom = mp2[-1].find_atom("C1*", "*")
                if atom is None:
                    atom = mp2[-1][0]
                keys.append(atom.pos)
                com += atom.pos
                # continue from here 1 Feb
            com = com / len(keys)
            w0 = []
            # mid_r = (len(keys) // 2) - 1
            # rn = abs(keys[mid_r].dist(keys[-1]))
            # r0 = abs(keys[mid_r].dist(keys[0]))
            # rn = abs(com.dist(keys[-1]))
            # r0 = abs(com.dist(keys[0]))
            # max_radius = max(r0, rn) + 20.0
            max_radius = 20.0

            for r in range(len(mp2)):
                w0tmp = []
                for atom in mp2[r]:
                    # dr = atom.pos.dist(keys[mid_r])
                    dr = atom.pos.dist(com)
                    w00 = (np.cos(np.pi * (dr / max_radius)) + 1.0) / 2.0
                    # w00 = pow(1.0 - dr / max_radius, 2)
                    w0tmp.append(w00)
                w0.append(w0tmp)
            f0 = mp1.copy()
            co1 = []
            co2 = []
            for r in range(len(mp2)):
                for a in range(len(mp1[r])):
                    co1.append(mp1[r][a].pos)  # .tolist())
                    co2.append(mp2[r][a].pos)  # .tolist())
                # transform = Pseudoregularize.orthogonal_transformation(co1, co2)
                superpose = gemmi.superpose_positions(co1, co2)
                transform = superpose.transform
                # rebuild
                for a in range(len(mp2[r])):
                    f0[r][a].pos = gemmi.Position(transform.apply(mp2[r][a].pos))
            for r in range(len(mp1)):
                for a in range(len(mp1[r])):
                    mp1[r][a].pos = w0[r][a] * f0[r][a].pos
            for r in range(len(frag)):
                CR = frag[r]
                # just to make sure the residues in mol_work is also in mol_ref
                try:
                    res = mol_work[model_number][CR.C][CR.S][CR.R]
                    res_copy = mol_copy[model_number][CR.C][CR.S][CR.R]
                except IndexError:
                    continue
                for a in range(len(mp1[r])):
                    res[a].pos = mp1[r][a].pos
                    res_copy[a].b_iso = w0[r][a]
                    res_copy[a].occ = float(count)
                    # mol_work[model_number][CR.C][CR.S][CR.R][a].pos = mp1[r][a].pos
            count += 1
        mol_copy.write_minimal_pdb(f"test_out_biso_weight_{cycle}.pdb")
        return True

    def get_frags_clusters(
        self,
        atom_selection: Union[str, list] = "all",
        dbscan_eps: float = 2.3,
        attr_name: str = "cluster",
        outfile_suffix: str = None,
    ):
        """
        Use ccpem_pyutils cluster function to get labels for residues clusters

        Arguments:
            original_structure: gemmi.Structure
                input structure as reference for pseudoregularisation
            atom_selection: Union[str, list], optional
                atom selection for coordinate retrieval
                Input a list of atom names or any of the following keywords:
                    "all": all atoms in the model
                    "backbone": only model backbone atoms, all atoms for non polymers
                    "one_per_residue": representative atoms, e.g. CA for amino acids
                    "centre": central atom of the residue based on atom sequence

        Return:
            Boolean if succeeds or failed

        """
        print("IN GET FRAG CLUSTERS")
        list_ids, list_coords = self.gemmimodelutils.get_coordinates(
            return_list=True, atom_selection=atom_selection
        )

        # ids_arr = np.array(list_ids)
        cluster_labels = cluster_coord_features(
            np.array(list_coords), dbscan_eps=dbscan_eps
        )
        # for i in range(0, len(cluster_labels)):
        #    if cluster_labels[i] == -1:  # group labels -1 to adjacent clusters
        #        if i > 0:
        #            cluster_labels[i] = cluster_labels[i - 1]
        #        else:
        #            cluster_labels[i] = cluster_labels[i + 1]
        list_res_ids, list_res_attr = get_residue_attribute(list_ids, cluster_labels)
        print(list_res_ids[0], list_res_ids[-1])
        print(list_res_attr)
        dict_attr = {}
        # to put separated clusters in frags not according to same cluster number but
        # according to separation, a separate fragment if discontinued
        # get fragments from cluster labels
        frag_start_ind = np.where(np.diff(cluster_labels, prepend=np.nan))[0]
        for n in range(len(list_res_attr)):
            dict_attr[list_res_ids[n]] = list_res_attr[n]
        set_bfactor_attributes(
            self.gemmimodelutils.structure,
            dict_attr,
            skip_non_poly=False,
            attr_name=attr_name,
            outfile_suffix=outfile_suffix,
        )
        # get unique labels
        ##num_clusters = list(dict.fromkeys(cluster_labels))
        ##self.frags_cluster = [None] * len(num_clusters)
        self.frags_cluster = [None] * len(frag_start_ind)
        # setting the fragments from cluster labels
        # print(num_clusters)
        # clusters = [None] * len(num_clusters)
        # print(cluster_labels)
        # put clusters into frags_cluster list
        # print(len(cluster_labels))
        for n in range(len(frag_start_ind)):
            if n != len((frag_start_ind)) - 1:
                frags = list_ids[frag_start_ind[n] - 1 : frag_start_ind[n + 1] + 1]
            else:
                frags = list_ids[frag_start_ind[n] - 1 :]
            frags_dataclass = self.list_to_dataclass(frags)
            self.frags_cluster[n] = frags_dataclass

        # for i in range(len(num_clusters)):
        #    frags = ids_arr[cluster_labels == num_clusters[i]]
        #    # print(f"DEBUG: frag_index = {frags}")
        #    # frags = [list_ids[j] for j in frag_index]
        #    frag_dataclass = self.list_to_dataclass(frags)
        #    self.frags_cluster[i] = frag_dataclass

        if len(self.frags_cluster) != 0 and frags[0] is not None:
            return True
        else:
            return False

    # def get_frags_from_cluster_labels(self, cluster_labels, list_res_ids):

    def list_to_dataclass(self, array_ids: Union[str, np.ndarray] = None):
        if isinstance(array_ids, str):
            res_id = array_ids.split("_")
            return ConnectedResidues(C=str(res_id[1]), S=str(res_id[2]), R=res_id[3])
        else:
            frag = []
            for id in array_ids:
                res_id = id.split("_")
                frag.append(
                    ConnectedResidues(C=str(res_id[1]), S=str(res_id[2]), R=res_id[3])
                )
            return frag


if __name__ == "__main__":
    from pysheetbend.utils import fileio

    ippdb = "/home/swh514/Projects/work_and_examples/shiftfield/example4/data/test.pdb"
    struct, hetatms = fileio.get_structure(ippdb, keep_waters=True)
    workpdb = "/home/swh514/Projects/work_and_examples/shiftfield/example4/run6/sheetbend_pdbout_result_sheetbendfinal.pdb"  # noqa E501
    workstruc, hetatm_w = fileio.get_structure(workpdb, keep_waters=True)
    verbose = 10
    pr = Pseudoregularize(struct, model_number=0, verbose=verbose)
    # pr.check_fraglist()

    # test
    ref_A = struct[0]["A"]
    work_A = workstruc[0]["A"].clone()
    start = timer()
    pr.regularize_frag(workstruc, model_number=0)
    end = timer()
    # frag_len = pr.check_fraglist(len_only=True)
    count = 0
    sumerr = 0
    for i in range(len(work_A)):
        for j in range(len(work_A[i])):
            err = work_A[i][j].pos.dist(workstruc[0]["A"][i][j].pos)
            sumerr += err * err
            count += 1
    rmsd = np.sqrt(sumerr / count)
    print(f"rough RMSD {rmsd:.6f}")
    print("time for regularisation : {0:.4f}".format(end - start))
    workstruc.write_minimal_pdb("test_out_pseudoreg_gemmipy.pdb")

    # elastic neural network james krieger (2019) Neuroscie Lett 700 22-29
    # doruker et al Proteins 40 (2000)
    # atilgan et al Biophys J 80 (2001)
    # Bahar et al 2010, chem reviews 110 1463:1497

    # CryoDRGN NeRF, ECCV 2020
