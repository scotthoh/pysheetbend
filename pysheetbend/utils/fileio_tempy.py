# File handling for shiftfield code
# S.W.Hoh, University of York, 2020

import os
import sys
import numpy as np
from TEMPy.Vector import Vector as Vector
from TEMPy.EMMap import Map
from TEMPy.StructureParser import mmCIFParser as cifp
from TEMPy.StructureParser import PDBParser as pdbp
from TEMPy.ProtRep_Biopy import BioPy_Structure as BPS
from TEMPy.MapParser import MapParser as mp
from logger import log2file

mrcfile_import = False
try:
    import mrcfile

    mrcfile_import = True
except ImportError:
    mrfile_import = False


def get_structure(ipmodel, hetatom=True, verbose=0):
    """
    Read structure from file and return BioPy_Structure object
    and boolean if hetatoms is present
    Arguments:
        ippdb: coordinates file path (pdb/mmcif)
        hetatom: boolean to include hetatoms or not
        verbose: verbosity
    Return:
        TEMPy Biopy_Structure, boolean if hetatom is present
    """
    if ipmodel is None:
        print("Please specify path/file for input coordinates.\n")
        print("Exiting...\n")
        sys.exit()

    struc_id, file_ext = os.path.basename(ipmodel).split(".")
    if file_ext == "pdb" or file_ext == "ent":
        structure = pdbp.read_PDB_file(struc_id, ipmodel, hetatm=hetatom, water=False)
    else:
        structure = cifp.read_mmCIF_file(struc_id, ipmodel, hetatm=hetatom, water=False)
    if verbose > 1:
        print(f"{structure}")
    # reordering residues to make sure residues from same chain are grouped together
    new_reordered_struct = []
    chainlist = structure.split_into_chains()
    for c in chainlist:
        c.reorder_residues()
        new_reordered_struct = np.append(new_reordered_struct, c)
    struc = BPS(new_reordered_struct)
    # to make sure if hetatm is present and flag accordingly
    hetatm_present = False
    for atm in struc.atomList:
        if atm.record_name == "HETATM":
            hetatm_present = True
            break

    return struc, hetatm_present


def write_map_as_MRC(mapobj, outpath="mapout.mrc", verbose=0):
    """
    Write out map as MRC, for different map objects
    Arguments:
        mapobj: map object to be written out
        outpath: output map path. Default: "mapout.mrc"
        verbose: verbosity
    Return:
        boolean: True if file is written succesfully else otherwise
    """
    # TEMPy map
    if not mrcfile_import and mapobj.__class__.__name__ == "Map":
        if verbose > 1:
            print(f"Writing TEMPy map to {outpath}")
        mapobj.write_to_MRC_file(outpath)
    # mrcfile map
    elif mrcfile_import:
        if verbose > 1:
            print(f"Writing mrcfile map to {outpath}")
        newmrcobj = mrcfile.new(outpath, overwrite=True)
        mapobj.set_newmap_data_header(newmrcobj)
        newmrcobj.close()
    # tempy mapprocess map
    else:
        if verbose > 1:
            print(f"Writing TEMPy map_process map to {outpath}")
        newmrcobj = Map(
            np.zeros(mapobj.fullMap.shape), list(mapobj.origin), mapobj.apix, "mapname"
        )
        mapobj.set_newmap_data_header(newmrcobj)
        newmrcobj.update_header()
        newmrcobj.write_to_MRC_file(outpath)

    if os.path.exists(outpath):
        return True
    else:
        return False


def write_xmlout(ResultsByCycle, xmloutpath, ipmodel):

    f = open(xmloutpath, "w")
    for i in range(0, len(ResultsByCycle)):
        if i == 0:
            ResultsByCycle[i].write_xml_results_start(f, xmloutpath, ipmodel)
        ResultsByCycle[i].write_xml_results_cyc(f)
        if i == len(ResultsByCycle) - 1:
            ResultsByCycle[i].write_xml_results_end(f)
    f.close()
