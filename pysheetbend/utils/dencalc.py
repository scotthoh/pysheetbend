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
    struc_copy = structure
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
