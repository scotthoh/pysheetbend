import re
import gemmi
from pysheetbend.utils.map_funcs import resample_data_by_boxsize


def calculate_density(structure, reso, rate=1.5):
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = reso
    dencalc.rate = rate
    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])

    return dencalc.grid


def calculate_density_with_boxsize(
    structure,
    reso,
    rate=1.5,
    grid_shape=(100, 100, 100),
    # origin=None,
):
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = reso
    dencalc.rate = rate
    # struc_copy = structure
    # if origin is not None:
    #    tr = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*origin))
    #    struc_copy[0].transform_pos_and_adp(tr)
    #
    #    dencalc.set_grid_cell_and_spacegroup(struc_copy)
    #    dencalc.put_model_density_on_grid(struc_copy[0])
    #    resample_grid = resample_data_by_boxsize(dencalc.grid, grid_shape)
    # else:
    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])
    resample_grid = resample_data_by_boxsize(dencalc.grid, grid_shape)

    return resample_grid


if __name__ == "__main__":
    import sys
    from pysheetbend.utils import fileio
    import numpy as np

    mapin = sys.argv[1]
    pdbin = sys.argv[2]
    reso = float(sys.argv[3])

    m, gridinfo = fileio.read_map(mapin)
    s, hetatm_present = fileio.get_structure(pdbin)
    min_d = np.amax(gridinfo.voxel_size)
    samp_rate = reso / (2 * min_d)
    print(samp_rate)
    print(min_d)
    calc_map = calculate_density_with_boxsize(
        s, reso, rate=samp_rate, grid_shape=gridinfo.grid_shape
    )
    fileio.write_map_as_MRC(calc_map, m.grid.unit_cell, outpath="calc_map.mrc")
