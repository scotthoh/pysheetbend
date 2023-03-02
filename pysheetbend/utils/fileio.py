# File handling for shiftfield code
# S.W.Hoh, University of York, 2020

from __future__ import print_function
import os
import sys
import numpy as np
import gemmi
from pysheetbend.utils import cell


mrcfile_import = False
try:
    import mrcfile

    mrcfile_import = True
except ImportError:
    mrfile_import = False


def get_structure(
    ipmodel,
    keep_waters=False,
    keep_hetatom=True,
    keep_hydrogen=False,
    remove_empty_chains=True,
    verbose=0,
):
    """
    Read structure from file and return Gemmi structure object
    and boolean if hetatoms is present
    Arguments:
        ippdb: coordinates file path (pdb/mmcif)
        waters: boolen to include waters or not, default=False
        hetatom: boolean to include hetatoms or not, default=True
        hydrogen: boolean to include hydrogens or not, default=False
        remove_empty_chains: boolean to remove empty chains, default=True
        verbose: verbosity, default=0
    Return:
        Gemmi Structure, boolean if hetatom is present
    """
    if ipmodel is None:
        print("Please specify path/file for input coordinates.\n")
        print("Exiting...\n")
        sys.exit()
    print(f"Reading {ipmodel}.")
    # structure = gemmi.read_structure(ipmodel, merge_chain_parts=False)
    structure = gemmi.read_structure(ipmodel)
    structure.setup_entities()
    structure.assign_label_seq_id()

    if not keep_waters:
        structure.remove_waters()
    if not keep_hetatom:
        structure.remove_ligands_and_waters()
    if remove_empty_chains:
        structure.remove_empty_chains()
    if not keep_hydrogen:
        structure.remove_hydrogens()
    structure.assign_serial_numbers()
    hetatm_present = False
    if len(structure) > 1:
        print("There are >1 models in the file. Using first model in this job.")
    for sc in structure[0].subchains():
        for r in sc:
            if r.het_flag == "H":
                hetatm_present = True
                break
    if verbose > 1:
        print(f"{structure}")
        print(f"HETATM present: {hetatm_present}")

    return structure, hetatm_present


def read_map(mapin, verbose=1):
    """
    Read map file (CCP4/MRC format)
    Arguments:
        mapin: input map path (CCP4/MRC format)
        verbose: verbosity, default=1
    Return:
        GEMMI map object and GridInfo class
    """
    m = gemmi.read_ccp4_map(mapin)
    grid_shape = np.array([m.header_i32(x) for x in (1, 2, 3)])
    grid_start = np.array([m.header_i32(x) for x in (5, 6, 7)])
    grid_samp = np.array([m.header_i32(x) for x in (8, 9, 10)])
    voxel_size = np.array(
        [m.grid.unit_cell.parameters[i] / grid_samp[i] for i in (0, 1, 2)]
    )
    origin = np.array([m.header_float(x) for x in (50, 51, 52)])
    if verbose >= 1:
        print(f"Reading {mapin}")
        print(f"Grid shape : {grid_shape}")
        print(f"Grid start : {grid_start}")
        print("Origin : {:.6f} {:.6f} {:.6f}".format(*origin))
        print(f"Grid sampling : {grid_samp}")
        print(f"Axis order : {m.axis_positions()}")
        print("Voxel size : {:.6f} {:.6f} {:.6f}".format(*voxel_size))
        print("Cell : {} {} {} {} {} {}".format(*m.grid.unit_cell.parameters))
    grid_info = cell.GridInfo(grid_shape, grid_start, grid_samp, voxel_size, origin)
    return m, grid_info


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
        unitcell: unit cell parameters 1D tuple or 1D np.array (a, b, c, alpha, beta, gamma) # noqa E501
        spacegroup: spacegroup for data; default: P1
        outpath: output map path. Default: "mapout.mrc"
        verbose: verbosity, default=0
    Return:
        boolean: True if file is written succesfully else otherwise
    """
    mrcout = gemmi.Ccp4Map()
    mrcout.grid = gemmi.FloatGrid(grid_data)
    if not isinstance(unitcell, gemmi.UnitCell):
        if len(unitcell) != 6:
            raise TypeError("Expecting unit cell array with length 6")
        else:
            mrcout.grid.unit_cell.set(
                unitcell[0],
                unitcell[1],
                unitcell[2],
                unitcell[3],
                unitcell[4],
                unitcell[5],
            )
    else:
        mrcout.grid.unit_cell.set(
            unitcell.a,
            unitcell.b,
            unitcell.c,
            unitcell.alpha,
            unitcell.beta,
            unitcell.gamma,
        )

    mrcout.grid.spacegroup = gemmi.SpaceGroup(spacegroup)
    mrcout.update_ccp4_header()
    if verbose > 1:
        print(f"Writing map data to {outpath}")
    mrcout.write_ccp4_map(outpath)
    # just for checking if output is written
    if os.path.exists(outpath):
        return True
    else:
        return False


def write_mask_as_MRC(
    maskin,
    unitcell,
    spacegroup="P1",
    outpath="maskout.mrc",
    verbose=0,
):
    """
    Write out mask as MRC
    Arguments:
        maskin: numpy 3D mask array of boolean type ((np.ma.mask)
        unitcell: unit cell parameters 1D tuple or 1D np.array (a, b, c, alpha, beta, gamma) # noqa E501
        spacegroup: spacegroup for data; default: P1
        outpath: output map path. Default: "mapout.mrc"
        verbose: verbosity, default=0
    Return:
        boolean: True if file is written succesfully else otherwise
    """
    mrcout = gemmi.Ccp4Map()
    mask = np.zeros(maskin.shape, dtype=np.float32)
    mask[~maskin] = 1.0
    mrcout.grid = gemmi.FloatGrid(mask)
    if not isinstance(unitcell, gemmi.UnitCell):
        if len(unitcell) != 6:
            raise TypeError("Expecting unit cell array with length 6")
        else:
            mrcout.grid.unit_cell.set(
                unitcell[0],
                unitcell[1],
                unitcell[2],
                unitcell[3],
                unitcell[4],
                unitcell[5],
            )
    else:
        mrcout.grid.unit_cell.set(
            unitcell.a,
            unitcell.b,
            unitcell.c,
            unitcell.alpha,
            unitcell.beta,
            unitcell.gamma,
        )

    mrcout.grid.spacegroup = gemmi.SpaceGroup(spacegroup)
    mrcout.update_ccp4_header()
    if verbose > 1:
        print(f"Writing mask data to {outpath}")
    mrcout.write_ccp4_map(outpath)
    # just for checking if output is written
    if os.path.exists(outpath):
        return True
    else:
        return False


def write_xmlout(ResultsByCycle, xmloutpath, ipmodel, pdboutname, ncycrr, ncyc):
    """
    Write xml file for results
    Arguments:
        ResultsByCycle: list of ResultsByCycle class
        xmloutpath: path and filename for output xml file
        ipmodel: input model filename/path
        pdboutname: final output model filename
        ncycrr: total number of macro cycles (refine regularize cycles)
        ncyc: total number of refine cycles
    """

    f = open(xmloutpath, "w")
    for m in range(0, ncycrr):
        for i in range(0, ncyc):
            if m == 0 and i == 0:
                ResultsByCycle[m][i].write_xml_results_header(f, pdboutname, ipmodel)
            if m > 0 and i == 0:
                ResultsByCycle[m][i].write_xml_results_start(f)
            ResultsByCycle[m][i].write_xml_results_cyc(f)
            if i == ncyc - 1:
                ResultsByCycle[m][i].write_xml_results_end_macrocyc(f)
        if m == ncycrr - 1:
            ResultsByCycle[m][i].write_xml_results_final(f)
    f.close()
