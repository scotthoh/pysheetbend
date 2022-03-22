# File handling for shiftfield code
# S.W.Hoh, University of York, 2020

from __future__ import print_function
import os
import sys
import numpy as np
import gemmi

mrcfile_import = False
try:
    import mrcfile

    mrcfile_import = True
except ImportError:
    mrfile_import = False


def get_structure(ipmodel, verbose=0):
    """
    Read structure from file and return Gemmi structure object
    and boolean if hetatoms is present
    Arguments:
        ippdb: coordinates file path (pdb/mmcif)
        hetatom: boolean to include hetatoms or not
        verbose: verbosity
    Return:
        Gemmi Structure, boolean if hetatom is present
    """
    if ipmodel is None:
        print("Please specify path/file for input coordinates.\n")
        print("Exiting...\n")
        sys.exit()

    structure = gemmi.read_structure(ipmodel)
    structure.setup_entities()
    structure.assign_label_seq_id()
    hetatm_present = False
    for sc in structure[0].subchains():
        for r in sc:
            if r.het_flag == "H":
                hetatm_present = True
                break
    if verbose > 1:
        print(f"{structure}")

    return structure, hetatm_present


def write_map_as_MRC(
    grid_data,
    unitcell,
    spacegroup="P1",
    outpath="mapout.mrc",
    verbose=0,
):
    """
    Write out map as MRC, for different map objects
    Arguments:
        grid_data: data (numpy 3D array)
        unitcell: unit cell parameters 1D tuple or 1D np.array (a, b, c, alpha, beta, gamma)
        spacegroup: spacegroup for data; default: P1
        outpath: output map path. Default: "mapout.mrc"
        verbose: verbosity
    Return:
        boolean: True if file is written succesfully else otherwise
    """
    mrcout = gemmi.Ccp4Map()
    # mrcout.grid = gemmi.FloatGrid(np.zeros((grid_shape), dtype=np.float32))
    mrcout.grid = grid_data
    if len(unitcell) != 6:
        raise TypeError(f"Unit cell has size {len(unitcell)}, expecting 6.")
    mrcout.grid.unit_cell.set(
        unitcell[0],
        unitcell[1],
        unitcell[2],
        unitcell[3],
        unitcell[4],
        unitcell[5],
    )
    mrcout.grid.spacegroup = gemmi.SpaceGroup(spacegroup)
    mrcout.update_ccp4_header()
    if verbose > 1:
        print(f"Writing map data to {outpath}")
    mrcout.write_ccp4_map(outpath)

    if os.path.exists(outpath):
        return True
    else:
        return False


def write_xmlout(ResultsByCycle, xmloutpath, ipmodel):

    f = open(xmloutpath, "w")
    print(f"Writing XML summary file: {xmloutpath}")
    for i in range(0, len(ResultsByCycle)):
        if i == 0:
            ResultsByCycle[i].write_xml_results_start(f, xmloutpath, ipmodel)
        ResultsByCycle[i].write_xml_results_cyc(f)
        if i == len(ResultsByCycle) - 1:
            ResultsByCycle[i].write_xml_results_end(f)
    f.close()
