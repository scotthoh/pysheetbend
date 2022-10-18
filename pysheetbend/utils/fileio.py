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
        waters: boolen to include waters or not
        hetatom: boolean to include hetatoms or not
        hydrogen: boolean to include hydrogens or not
        verbose: verbosity
    Return:
        Gemmi Structure, boolean if hetatom is present
    """
    if ipmodel is None:
        print("Please specify path/file for input coordinates.\n")
        print("Exiting...\n")
        sys.exit()
    print(f"Reading {ipmodel}.")
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

    return structure, hetatm_present


def read_map(mapin, verbose=0):
    m = gemmi.read_ccp4_map(mapin)
    grid_shape = np.array([m.header_i32(x) for x in (1, 2, 3)])
    grid_start = np.array([m.header_i32(x) for x in (5, 6, 7)])
    grid_samp = np.array([m.header_i32(x) for x in (8, 9, 10)])
    voxel_size = np.array(
        [m.grid.unit_cell.parameters[i] / grid_samp[i] for i in (0, 1, 2)]
    )
    origin = np.array([m.header_float(x) for x in (50, 51, 52)])
    print(f"Reading {mapin}")
    print(f"Grid shape : {grid_shape}")
    print(f"Grid start : {grid_start}")
    print("Origin : {:.6f} {:.6f} {:.6f}".format(*origin))
    print(f"Grid sampling : {grid_samp}")
    print(f"Axis order : {m.axis_positions()}")
    print("Voxel size : {:.6f} {:.6f} {:.6f}".format(*voxel_size))
    print("Cell : {} {} {} {} {} {}".format(*m.grid.unit_cell.parameters))
    # print("map datatype {0}".format(m.grid.dtype))
    grid_info = cell.GridInfo(grid_shape, grid_start, grid_samp, voxel_size, origin)
    # grid_cell
    # grid_info.grid_shape = grid_shape
    # grid_info.grid_start = grid_start
    # grid_info.voxel_size = voxel_size
    return m, grid_info


def write_map_as_MRC(
    grid_data,
    unitcell,
    spacegroup='P1',
    outpath='mapout.mrc',
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
    mrcout.grid = gemmi.FloatGrid(grid_data)
    if not isinstance(unitcell, gemmi.UnitCell):
        if len(unitcell) != 6:
            raise TypeError(f"Expecting unit cell array with length 6")
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
