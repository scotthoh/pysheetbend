from __future__ import absolute_import, print_function
from logging import raiseExceptions
import gemmi
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.signal import resample
from scipy.ndimage import measurements
from typing import Union

# from pysheetbend.shiftfield import effective_radius, fltr
import pysheetbend.shiftfield_util as sf_util
from math import fabs
from pysheetbend.utils.cell import GridInfo
from gemmi import Position as GPosition, FloatGrid, SpaceGroup


def map_grid_position_array(grid_info, structure, index=False):
    xyz_coordinates = []
    origin = np.array((grid_info.origin[0], grid_info.origin[1], grid_info.origin[2]))
    apix = np.array(
        (grid_info.voxel_size[0], grid_info.voxel_size[1], grid_info.voxel_size[2])
    )
    # print(origin, apix)
    # out = np.zeros(3, dtype=np.int16)
    for cra in structure[0].all():
        xyz = np.array((cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z))
        if index:
            xyz = np.round((xyz - origin) / apix, 0)
            # xyz = (np.round(((xyz - origin) / apix, 0))).astype(int)
        else:
            xyz = (xyz - origin) / apix
        # check if coordinates are within boundary
        if (xyz > grid_info.grid_shape[0] - 1).any():
            xyz = np.round(xyz, 0)
            # check again if over then apply periodic boundary
            if (xyz > grid_info.grid_shape[0] - 1).any():
                for i in xyz:
                    if i > grid_info.grid_shape[0]:
                        i -= grid_info.grid_shape[0]
                # raise ValueError('Atom position in ')
        xyz_coordinates.append([xyz[0], xyz[1], xyz[2]])
    # print(xyz_coordinates)
    return xyz_coordinates


def numpy_to_gemmi_grid(numpy_array, unitcell, spacegroup="P1"):
    grid = FloatGrid(numpy_array)
    if isinstance(unitcell, list) or isinstance(unitcell, np.ndarray):
        grid.unit_cell.set(
            unitcell[0], unitcell[1], unitcell[2], unitcell[3], unitcell[4], unitcell[5]
        )
    else:
        grid.unit_cell.set(
            unitcell.a,
            unitcell.b,
            unitcell.c,
            unitcell.alpha,
            unitcell.beta,
            unitcell.gamma,
        )
    grid.spacegroup = SpaceGroup(spacegroup)

    return grid


def update_atoms_position(
    grid_dx,
    grid_dy,
    grid_dz,
    structure,
    mode="linear",
    scale=1.0,
):
    """
    Interpolate values of atom positions on the grids and update positions.
    """
    modes = ["linear", "tricubic"]
    if mode not in modes:
        raise ValueError(f"Invalid mode. Expected one of {modes}")
    if mode == "linear":
        for cra in structure[0].all():
            dx = scale * grid_dx.interpolate_value(cra.atom.pos) * grid_dx.unit_cell.a
            dy = scale * grid_dy.interpolate_value(cra.atom.pos) * grid_dy.unit_cell.b
            dz = scale * grid_dz.interpolate_value(cra.atom.pos) * grid_dz.unit_cell.c
            cra.atom.pos += GPosition(dx, dy, dz)

    if mode == "tricubic":
        for cra in structure[0].all():
            dx = (
                scale
                * grid_dx.tricubic_interpolation(cra.atom.pos)
                * grid_dx.unit_cell.a
            )
            dy = (
                scale
                * grid_dy.tricubic_interpolation(cra.atom.pos)
                * grid_dy.unit_cell.b
            )
            dz = (
                scale
                * grid_dz.tricubic_interpolation(cra.atom.pos)
                * grid_dz.unit_cell.c
            )
            cra.atom.pos += GPosition(dx, dy, dz)


def update_uiso_values(
    grid_du,
    structure,
    biso_range,
    mode="linear",
    scale=1.0,
    cycle=0,
    verbose=0,
    ucyc=-1,
):
    modes = ["linear", "tricubic"]
    if mode not in modes:
        raise ValueError(f"Invalid mode. Expected one of {modes}")
    if verbose >= 2:
        shift_u = []
    if mode == "linear":
        for cra in structure[0].all():
            du = scale * grid_du.interpolate_value(cra.atom.pos)
            db = sf_util.u2b(du)
            cra.atom.b_iso = sf_util.limit_biso(
                (cra.atom.b_iso - db), biso_range[0], biso_range[1]
            )
    if mode == "tricubic":
        for cra in structure[0].all():
            du = scale * grid_du.tricubic_interpolation(cra.atom.pos)
            db = sf_util.u2b(du)
            cra.atom.b_iso = sf_util.limit_biso(
                (cra.atom.b_iso - db), biso_range[0], biso_range[1]
            )
            if verbose >= 2:
                shift_u.append(db)
    if verbose >= 2:
        if len(shift_u) != 0:
            outusiocsv = f"shiftuiso_u2b_{cycle}"
            if ucyc >= 0:
                outusiocsv += f"_{ucyc}"
            outusiocsv += ".csv"
            fuiso = open(outusiocsv, "w")
            for j in range(0, len(shift_u)):
                fuiso.write("{0}, {1}\n".format(j, shift_u[j]))
            fuiso.close()


def grid_coord_to_frac(densmap=None, grid_info=None, tempy_flag=False):
    """
    convert grid coordinates to fractional coordinates at the given grid_shape
    TEMPy em map box_size is ZYX format
    GEMMI map grid is XYZ format
    """
    if tempy_flag:
        zpos, ypos, xpos = np.mgrid[
            0 : densmap.z_size(), 0 : densmap.y_size(), 0 : densmap.x_size()
        ]
        ind_pos = np.vstack([zpos.ravel(), ypos.ravel(), xpos.ravel()]).T
        # get coord position
        # zyx_pos_new = (zyx_pos * densmap.apix) + np.array([z0, y0, x0])
        frac_coord = ind_pos[:] / np.array(densmap.box_size())
    else:
        xpos, ypos, zpos = np.mgrid[
            0 : grid_info.grid_shape[0],
            0 : grid_info.grid_shape[1],
            0 : grid_info.grid_shape[2],
        ]
        ind_pos = np.vstack([xpos.ravel(), ypos.ravel(), zpos.ravel()]).T
        frac_coord = ind_pos[:] / grid_info.grid_shape

    return frac_coord, ind_pos


def fltr(r, radius, function="quadratic"):
    """
    Radial map filter, returns value from filter function,
    outside radius=0.0

    Arguments
    ---------
        r: float
            distance
        radius: float
            radius cutoff
        function: {"step", "linear", "quadratic"}
            function type used to calculate spherical radius

    Return
    ------
        Radius value from filter function
    """
    if r < radius:
        if function == "quadratic":
            return pow(1.0 - r / radius, 2)
        elif function == "linear":
            return 1.0 - r / radius
        elif function == "step":
            return 1.0
    else:
        return 0.0


def effective_radius(radius, function="quadratic"):
    """
    Calculates effective radius of the radial function from function and radius given

    Arguments
    ---------
        radius: float
            radius cutoff
        func: {"step", "linear", "quadratic"}
            function type used to effective radius of the radial function. Default to "quadratic"

    Return
    ------
        effective radius value
    """
    nrad = 1000
    drad = 0.25

    r = np.arange(0, nrad)
    r = drad * (r + 0.5)
    r_bool = r < radius
    if function == "quadratic":
        rf = r[r_bool] * r[r_bool] * pow(1.0 - r[r_bool] / radius, 2)
    elif function == "linear":
        rf = r[r_bool] * r[r_bool] * (1.0 - r[r_bool] / radius)
    elif function == "step":
        rf = r[r_bool] * r[r_bool]

    r = np.zeros(nrad)
    r[r_bool] = rf[:]
    sum_r = np.cumsum(r)
    i = np.argmax(sum_r > (0.99 * sum_r[-1]))

    return drad * (float(i) + 1.0)


def prepare_mask_filter_1(
    grid_info,
    filter_radius=15.0,
    function="quadratic",
):
    eff_rad = effective_radius(filter_radius, function)
    fltr_data_r = np.zeros(grid_info.grid_shape, dtype=np.float32)
    # x,y,z convention
    nx, ny, nz = np.indices(grid_info.grid_shape)
    indi = np.vstack([nx.ravel(), ny.ravel(), nz.ravel()]).T
    c = indi + grid_info.grid_half  # self.gridshape.g_half
    c = np.fmod(c, grid_info.grid_shape)
    c_bool = c < 0
    c[c_bool[:, 0], 0] += grid_info.grid_shape[0]
    c[c_bool[:, 1], 1] += grid_info.grid_shape[1]
    c[c_bool[:, 2], 2] += grid_info.grid_shape[2]
    c -= grid_info.grid_half
    # at the start the origin are corrected to 0 so no need offset with origin
    pos = c[:] * grid_info.voxel_size
    dist = np.sqrt(np.sum(np.square(pos), axis=1)).reshape(grid_info.grid_shape)
    dist_bool = np.logical_and((dist < eff_rad), (dist < filter_radius))

    f000 = 0.0
    if function == "quadratic":
        rf = pow(1.0 - dist[dist_bool] / filter_radius, 2)
    elif function == "linear":
        rf = 1.0 - dist[dist_bool] / filter_radius
    elif function == "step":
        rf = 1.0
    else:
        raise ValueError('Specify function={"quadratic", "linear", "step"}.')
    f000 = np.sum(rf)
    fltr_data_r[dist_bool] = rf[:]
    # fltr_data_shift = np.fft.fftshift(fltr_data)
    # return fltr_data_shift, f000
    del c, c_bool, dist, dist_bool, indi
    return fltr_data_r, f000


def prepare_mask_filter(
    apix: np.ndarray, fltr_radius=15.0, pad=5, function="quadratic"
):
    """
    Prepare filter kernel to be used with scipy.fftconvolve.
    Peak at the center
    Arguments
        apix: pixel size [x,y,z]
        fltr_radius: radius for filter
        pad: padding for filter kernel
    Return
        Filter 3D array
    """
    rad = effective_radius(fltr_radius, function)
    win_points = int(fltr_radius * 2) + 1 + (pad * 2)
    start = (fltr_radius + pad) * -1
    end = fltr_radius + pad
    rad_x = np.linspace(start, end, num=win_points)
    rad_y = np.linspace(start, end, num=win_points)
    rad_z = np.linspace(start, end, num=win_points)
    rad_x = rad_x * apix[0]
    rad_y = rad_y * apix[1]
    rad_z = rad_z * apix[2]
    rad_x = rad_x ** 2
    rad_y = rad_y ** 2
    rad_z = rad_z ** 2
    dist = np.sqrt(rad_x[:, None, None] + rad_y[:, None] + rad_z)
    # dist_ind = zip(*np.nonzero(dist < rad))
    dist_bool = np.logical_and((dist < rad), (dist < fltr_radius))
    fltr_data = np.zeros(dist.shape, dtype=np.float32)
    # count = 0
    # f000 = 0.0
    # fill the radial function map
    if function == "quadratic":
        rf = pow(1.0 - dist[dist_bool] / fltr_radius, 2)
    elif function == "linear":
        rf = 1.0 - dist[dist_bool] / fltr_radius
    elif function == "step":
        rf = 1.0
    else:
        raise ValueError("Choose from function mode 2:Quadratic, 1:Linear, 0:Step.")
    # for i in dist_ind:
    #    rf = fltr(dist[i], fltr_radius, 2)
    #    count += 1
    #    fltr_data[i] = rf
    f000 = np.sum(rf)
    fltr_data[dist_bool] = rf[:]
    return fltr_data, f000


def make_filter_edge_centered(
    grid_info, filter_radius=15.0, function="quadratic", verbose=0
):
    """
    Prepare filter kernel for use with fft_convolution_filter.
    Peak at the edge.

    Arguments
    ---------
        grid_info: GridInfo
            pysheetbend.utils.cell.GridInfo object
        filter_radius: float
            radius for filter
        function: {"step","linear","quadratic"}
            function type to use for calculating spherical filter radius
        verbose:
            verbosity

    Return
    ------
        Filter 3D array, sum of values
    """
    eff_rad = effective_radius(filter_radius, function)
    fltr_data_r = np.zeros(grid_info.grid_shape, dtype=np.float32)
    x, y, z = grid_info.grid_shape
    if grid_info.grid_shape[0] % 2:  # odd
        # to make sure dist 0.0 is at the start after calculation
        rad_x = np.arange(-np.ceil(x / 2.0), np.floor(x / 2.0))
        rad_y = np.arange(-np.ceil(y / 2.0), np.floor(y / 2.0))
        rad_z = np.arange(-np.ceil(z / 2.0), np.floor(z / 2.0))
    else:
        rad_x = np.arange(-np.floor(x / 2.0), np.ceil(x / 2.0))
        rad_y = np.arange(-np.floor(y / 2.0), np.ceil(y / 2.0))
        rad_z = np.arange(-np.floor(z / 2.0), np.ceil(z / 2.0))

    rad_x[rad_x < 0] += x
    rad_y[rad_y < 0] += y
    rad_z[rad_z < 0] += z
    rad_x -= grid_info.grid_half[0]
    rad_y -= grid_info.grid_half[1]
    rad_z -= grid_info.grid_half[2]
    rad_x *= grid_info.voxel_size[0] + grid_info.grid_start[0]
    rad_y *= grid_info.voxel_size[1] + grid_info.grid_start[1]
    rad_z *= grid_info.voxel_size[2] + grid_info.grid_start[2]
    rad_x = rad_x ** 2
    rad_y = rad_y ** 2
    rad_z = rad_z ** 2
    dist = np.sqrt(rad_x[:, None, None] + rad_y[:, None] + rad_z)
    dist_bool = np.logical_and((dist < eff_rad), (dist < filter_radius))

    if function == "quadratic":
        rf = pow(1.0 - dist[dist_bool] / filter_radius, 2)
    elif function == "linear":
        rf = 1.0 - dist[dist_bool] / filter_radius
    elif function == "step":
        rf = np.full_like(dist, fill_value=1.0)
    f000 = np.sum(rf)
    fltr_data_r[dist_bool] = rf[:]
    if verbose >= 1:
        print(f"f000 = {f000:.4f}")

    return fltr_data_r, f000


def fft_convolution_filter(
    data_arr: np.ndarray,
    filter: np.ndarray,
    scale,
    grid_info: GridInfo,
    fft_obj=None,
    ifft_obj=None,
):
    """
    Returns filtered/covolved data
    Arguments:
        data_arr: numpy array of data
        filter: numpy array of filter
        scale: scale value for the fft convolution
        fft_obj: planned fft object
        ifft_obj: planned ifft object
        grid_info: GridInfo class containing grid info
    """
    if fft_obj is None:
        fft_obj, ifft_obj = sf_util.plan_fft_ifft(gridinfo=grid_info.grid_shape)

    data_r = np.zeros(grid_info.grid_shape, dtype=np.float32)
    data_r = data_arr.copy()
    fltr_input = np.zeros(grid_info.grid_shape, dtype=np.float32)
    fltr_input = filter.copy()

    fltr_data_c = np.zeros(grid_info.grid_reci, dtype=np.complex64)
    data_c = np.zeros(grid_info.grid_reci, dtype=np.complex64)
    fltr_data_c = fft_obj(fltr_input, fltr_data_c)
    fltr_data_c = fltr_data_c.conjugate().copy()

    data_c = fft_obj(data_r, data_c)
    data_c = data_c.conjugate().copy()

    data_c[:, :, :] = scale * data_c[:, :, :] * fltr_data_c[:, :, :]
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c, data_r)

    return data_r


def find_background_peak(map_array, iter=2):
    """
    rewritten from ccpem TEMPy (from Agnel)
    Arguments:
        mapdata: data array
        iter: iteration cycles (default=2)
    Return:
        peak, average
    """
    lbin = np.amin(map_array)
    rbin = np.amax(map_array)
    avg = np.nanmean(map_array)  # .mean()
    sigma = np.nanstd(map_array)  # .std()
    for i in range(iter):
        if i == 0:
            data = map_array
        else:
            data = map_array[(map_array >= lbin) & (map_array <= rbin)]

        freq, bins = np.histogram(data, 100)
        ind = np.nonzero(freq == np.amax(freq))[0]
        peak = None
        for j in ind:
            val = (bins[j] + bins[j + 1]) / 2.0
            if val < float(avg) + float(sigma):
                peak = val
                lbin = bins[j]
                rbin = bins[j + 1]
            if peak is None:
                break

    return peak, avg


def get_peak_and_sigma(map_array):
    peak, avg = find_background_peak(map_array)

    if peak is None:
        peak = avg
    sigma = None
    if peak is not None:
        mask_array = map_array[map_array > peak]
        sigma = np.sqrt(np.mean(np.square(mask_array - peak)))

    return peak, avg, sigma


def calculate_map_threshold(map_data, sigma_factor: float = 2.0):
    if isinstance(map_data, gemmi.FloatGrid):
        maparray = np.array(map_data, copy=False, dtype=np.float32)
    else:
        maparray = map_data
    try:
        peak, avg, sigma = get_peak_and_sigma(maparray)
        vol_threshold = float(avg) + (sigma_factor * float(sigma))
    except:
        try:
            mean = np.nanmean(maparray)
            sigma = np.nanstd(maparray)
            vol_threshold = float(mean) + (1.5 * float(sigma))
        # peak, avg, sigma = get_peak_and_sigma(map_data)
        except:
            vol_threshold = 0.0

    return vol_threshold


def calculate_map_contour(map_data, sigma_factor: float = 1.5):
    if isinstance(map_data, gemmi.FloatGrid):
        maparray = np.array(map_data, copy=False, dtype=np.float32)
        peak, avg, sigma = get_peak_and_sigma(maparray)
    else:
        peak, avg, sigma = get_peak_and_sigma(map_data)
    if peak is not None:
        contour = float(peak) + (sigma_factor * float(sigma))
    else:
        contour = 0.0

    return contour


def cor_mod_array(a, b):
    """
    Returns corrected remainder of division. If remainder <0,
    then adds value b to remainder.
    Arguments
        a: array of Dividend (x,y,z indices)
        b: array of Divisor (x,y,z indices)
    """
    c = np.fmod(a, b)
    d = np.transpose(np.nonzero(c < 0))
    # d, e = np.nonzero(c<0)
    for i in d:  # range(len(d)):
        c[i[0], i[1]] += b[i[1]]
        # c[i, j] += b[i]
    return c


def calculate_overlap_scores(
    map_data,
    map_data2,
    map_threshold,
    map_threshold2,
):
    if isinstance(map_data, gemmi.FloatGrid):
        map_data = np.array(map_data, copy=False, dtype=np.float32)
    if isinstance(map_data2, gemmi.FloatGrid):
        map_data2 = np.array(map_data2, copy=False, dtype=np.float32)
    binmap1 = map_data > float(map_threshold)
    binmap2 = map_data2 > float(map_threshold2)
    mask_array = (binmap1 * binmap2) > 0

    size1 = np.sum(binmap1)
    size2 = np.sum(binmap2)
    return float(np.sum(mask_array)) / size1, float(np.sum(mask_array)) / size2


def calculate_pixel_size(cell_size, grid_shape):

    x = cell_size.a / float(grid_shape[0])  # x
    y = cell_size.b / float(grid_shape[1])  # y
    z = cell_size.c / float(grid_shape[2])  # z

    return np.array((x, y, z), dtype=np.float32)


def resample_data_by_boxsize(map_data, new_shape):
    if isinstance(map_data, gemmi.FloatGrid):
        grid_data = np.array(map_data, copy=False, dtype=np.float32)
        grid_data = resample(grid_data, new_shape[0], axis=0)
        grid_data = resample(grid_data, new_shape[1], axis=1)
        grid_data = resample(grid_data, new_shape[2], axis=2)
        # map_data = gemmi.FloatGrid(grid_data)
        return grid_data
    else:
        if isinstance(map_data, np.ndarray):
            map_data = resample(map_data, new_shape[0], axis=0)
            map_data = resample(map_data, new_shape[1], axis=1)
            map_data = resample(map_data, new_shape[2], axis=2)
            return map_data
        else:
            return map_data


def compare_tuple(tuple1, tuple2):
    return bool(np.asarray(tuple1 == tuple2).all())


def make_mask_from_maps(
    maps,
    grid_info,
    resolution=2.0,
    func="quadratic",
    lpfilt_pre=False,
    lpfilt_post=False,
    ref_scale=False,
    radcyc=15.0,
    verbose=0,
):
    # amplitude match
    fft_obj, ifft_obj = sf_util.plan_fft_ifft(gridinfo=grid_info)
    # fft_obj = sf_util.plan_fft(grid_info, input_dtype=np.float32)
    # ifft_obj = sf_util.plan_ifft(grid_info, input_dtype=np.complex64)
    # make sure maps are of same grid shape/size
    if not np.all(np.array(maps[0].shape) == grid_info.grid_shape):
        maps[0] = resample_data_by_boxsize(maps[0], grid_info.grid_shape)
    if not np.all(np.array(maps[1].shape) == grid_info.grid_shape):
        maps[1] = resample_data_by_boxsize(maps[1], grid_info.grid_shape)

    scaled_maps = amplitude_match(
        maps[0],
        maps[1],
        grid_info,
        grid_info,
        reso=resolution,
        lpfilt_pre=lpfilt_pre,
        lpfilt_post=lpfilt_post,
        ref_scale=ref_scale,
        fft_obj=fft_obj,
        ifft_obj=ifft_obj,
    )

    combined_map = scaled_maps[0] + scaled_maps[1]
    filt_data_r, f000 = make_filter_edge_centered(
        grid_info, filter_radius=radcyc, function=func
    )
    mmap = fft_convolution_filter(
        combined_map,
        filt_data_r,
        1.0 / f000,
        grid_info,
        fft_obj,
        ifft_obj,
    )
    if verbose >= 5:
        print(f"{mmap.min()}, {mmap.max()}, {mmap.mean()}, {mmap.std()}")
    mmap_ma = np.ma.masked_less(mmap, mmap.mean())

    return mmap, mmap_ma


def downsample_mask(mask_array, downsamp_shape):
    """
    Downsample mask array of bool type. Input mask array as int type
    Args:
        mask_array : Mask array in int type
        downsamp_shape: shape to be resampled into

    Returns:
        np.array: downsampled array in bool type
    """
    mask_array = resample(mask_array, downsamp_shape[0], axis=0)
    mask_array = resample(mask_array, downsamp_shape[1], axis=1)
    mask_array = resample(mask_array, downsamp_shape[2], axis=2)

    return np.ma.masked_less(mask_array, mask_array == 0).mask


def make_fourier_shell(map_shape, keep_shape=False, fftshift=True, normalise=False):
    """
    Make grid sampling frequencies as distance from centre for a given grid
    Return:
        Grid with sampling frequencies
    """
    if keep_shape:
        rad_0 = np.arange(
            np.floor(map_shape[0] / 2.0) * -1, np.ceil(map_shape[0] / 2.0)
        ) / float(np.floor(map_shape[0]))
        rad_1 = np.arange(
            np.floor(map_shape[1] / 2.0) * -1, np.ceil(map_shape[1] / 2.0)
        ) / float(np.floor(map_shape[1]))
        rad_2 = np.arange(
            np.floor(map_shape[2] / 2.0) * -1, np.ceil(map_shape[2] / 2.0)
        ) / float(np.floor(map_shape[2]))
        if not fftshift:
            rad_2 = ifftshift(rad_2)
    else:
        rad_0 = np.arange(
            np.floor(map_shape[0] / 2.0) * -1, np.ceil(map_shape[0] / 2.0)
        ) / float(np.floor(map_shape[0]))
        rad_1 = np.arange(
            np.floor(map_shape[1] / 2.0) * -1, np.ceil(map_shape[1] / 2.0)
        ) / float(np.floor(map_shape[1]))
        rad_2 = np.arange(0, np.floor(map_shape[2] / 2.0) + 1) / float(
            np.floor(map_shape[2])
        )
    if not fftshift:
        rad_0 = fftshift(rad_0)
        rad_1 = fftshift(rad_1)
    if normalise:
        rad_0 = rad_0 / float(np.floor(map_shape[0]))
        rad_1 = rad_1 / float(np.floor(map_shape[1]))
        rad_2 = rad_2 / float(np.floor(map_shape[2]))
    rad_2 = rad_2 ** 2
    rad_1 = rad_1 ** 2
    rad_0 = rad_0 ** 2
    dist = np.sqrt(rad_0[:, None, None] + rad_1[:, None] + rad_2)
    return dist


def tanh_lowpass(map_shape, cutoff, fall=0.3, keep_shape=False):
    if fall == 0.0:
        fall = 0.01
    drop = np.pi / (2 * float(cutoff) * float(fall))
    cutoff = min(float(cutoff), 0.5)
    dist = make_fourier_shell(map_shape, keep_shape=keep_shape)
    dist1 = dist + cutoff
    dist1[:] = drop * dist1
    dist1[:] = np.tanh(dist1)
    dist[:] = dist - cutoff
    dist[:] = drop * dist
    dist[:] = np.tanh(dist)
    dist[:] = dist1 - dist
    dist = 0.5 * dist
    del dist1
    return dist


def amplitude_match(
    emmap1,
    emmap2,
    grid1_info,
    grid2_info,
    step=0.0005,
    reso=None,
    lpfilt_pre=False,
    lpfilt_post=False,
    ref_scale=False,
    fft_obj=None,
    ifft_obj=None,
):
    if fft_obj is None:
        fft_obj, ifft_obj = sf_util.plan_fft_ifft(gridinfo=grid1_info)
    # Taken from Agnel's code in TEMPy library / CCPEM
    # assuming maps used are always cubic (same dimension x,y,z)
    ft1 = fft_obj(emmap1)
    ft1 = fftshift(ft1, axes=(0, 1))
    ft2 = fft_obj(emmap2)
    ft2 = fftshift(ft2, axes=(0, 1))
    # ft1 = fftshift(fftn(emmap1))
    # ft2 = fftshift(fftn(emmap2))
    # pre scaling lowpass filter
    if reso is not None:
        cutoff1 = grid1_info.voxel_size[0] / float(reso)
        cutoff2 = grid2_info.voxel_size[0] / float(reso)
        if lpfilt_pre and not lpfilt_post:
            ftfltr1 = tanh_lowpass(
                grid1_info.grid_shape, cutoff1, fall=0.2, keep_shape=False
            )
            ftfltr2 = tanh_lowpass(
                grid2_info.grid_shape, cutoff2, fall=0.2, keep_shape=False
            )
            ft1[:] = ft1 * ftfltr1
            ft2[:] = ft2 * ftfltr2
            del ftfltr1, ftfltr2
    # min dimension
    size1 = np.min(grid1_info.grid_shape)
    size2 = np.min(grid2_info.grid_shape)
    if step is None:
        step = 1.0 / min(
            size1 * grid1_info.voxel_size[0], size2 * grid2_info.voxel_size[0]
        )
    else:
        step = max(
            1.0
            / min(
                size1 * grid1_info.voxel_size[0],
                size2 * grid2_info.voxel_size[0],
            ),
            step,
        )
    dist1 = (
        make_fourier_shell(grid1_info.grid_shape, fftshift=True)
        / grid1_info.voxel_size[0]
    )
    dist2 = (
        make_fourier_shell(grid2_info.grid_shape, fftshift=True)
        / grid2_info.voxel_size[0]
    )
    print(dist1.shape, dist2.shape)
    # ft1_avg = []
    # ft2_avg = []
    # ft1_avg_new = []
    lfreq = []
    # select max spatial frequency to iterate to low resolution map
    maxlvl = 0.5 / min(grid1_info.voxel_size[0], grid2_info.voxel_size[0])
    # loop over freq shells, shellwidth = 0.005
    # scale amplitudes
    nc = 0
    x = 0.0
    highlvl = x + step
    while x < maxlvl:
        # indices between upper and lower shell bound
        fshells1 = (dist1 < min(maxlvl, highlvl)) & (dist1 >= x)
        # radial average
        shellvec1 = ft1[fshells1]
        # indices between upper and lower shell bound
        fshells2 = (dist2 < min(maxlvl, highlvl)) & (dist2 >= x)
        # radial average
        shellvec2 = ft2[fshells2]
        abs1 = abs(shellvec1)
        abs2 = abs(shellvec2)
        ns1 = len(np.nonzero(abs1)[0])
        ns2 = len(np.nonzero(abs2)[0])
        ###
        if (ns1 < 5 or ns2 < 5) and nc < 3:
            nc += 1
            highlvl = min(maxlvl, x + (nc + 1) * step)
            if highlvl < maxlvl:
                continue
            # x = max(0.0, x - nc * step)
            # continue
        else:
            nc = 0
        if len(abs1) != 0:
            ft1_m = np.nanmean(np.square(abs1))
        else:
            ft1_m = 1e-6
        if len(abs2) != 0:
            ft2_m = np.nanmean(np.square(abs2))
        else:
            ft2_m = 1e-6
        if ft1_m == 0.0 and ft2_m == 0.0:
            x = highlvl
            highlvl = x + step
            if highlvl < maxlvl:
                continue

        # sq of radial avg amplitude
        # ft1_avg.append(np.log10(np.mean(np.square(abs1))))
        # ft2_avg.append(np.log10(np.mean(np.square(abs2))))

        # scale to amplitudes of reference map, map1 if reference
        if ref_scale:
            if ft2_m == 0.0:
                continue
            ft2[fshells2] = shellvec2 * np.sqrt(ft1_m / ft2_m)
        else:
            # replace with avg amplitudes for the two maps
            ft1[fshells1] = shellvec1 * np.sqrt((ft2_m + ft1_m) / (2.0 * ft1_m))
            ft2[fshells2] = shellvec2 * np.sqrt((ft2_m + ft1_m) / (2.0 * ft2_m))
        lfreq.append(highlvl)
        x = highlvl
        highlvl = x + step

        # sampling_frq = highlvl
        # cutoff_freq = min((1.0 / reso) + 0.25, maxlvl)
        del fshells1, fshells2, shellvec1, shellvec2
        # scale the rest and break after relevant frequencies
        # if sampling_frq > cutoff_freq:
        #    fshells1 = dist1 >= highlvl
        #    shellvec1 = ft1[fshells1]
        #    ft1_m = np.mean(abs(shellvec1))
        #    fshells2 = dist2 >= highlvl
        #    shellvec2 = ft2[fshells2]
        #    ft2_m = np.mean(abs(shellvec2))
        #    if ft1_m == 0.0 and ft2_m == 0.0:
        #        break
        #    ft1_avg.append(np.log10(np.mean(np.square(abs(shellvec1)))))
        #    ft2_avg.append(np.log10(np.mean(np.square(abs(shellvec2)))))

        #   if ref_scale:
        #        if ft1_m == 0.0:
        #            break
        #        ft1[fshells1] = shellvec1 * (ft2_m / ft1_m)
        #    else:
        #        ft1[fshells1] = shellvec1 * (ft2_m + ft1_m) / (2 * ft1_m)
        #        ft2[fshells2] = shellvec2 * (ft2_m + ft1_m) / (2 * ft2_m)

        # ft1_m = np.mean(abs(ft1[fshells1]))
        # ft1_avg_new.
        #    lfreq.append((highlvl + step / 2))
        #    break
        # x = highlvl
        # highlvl = x + step

        # post scaling low pass filter
    if reso is not None:
        cutoff1 = grid1_info.voxel_size[0] / float(reso)
        cutoff2 = grid2_info.voxel_size[0] / float(reso)
        if lpfilt_post and not lpfilt_pre:
            ftfltr1 = tanh_lowpass(
                grid1_info.grid_shape, cutoff1, fall=0.2, keep_shape=False
            )
            ftfltr2 = tanh_lowpass(
                grid2_info.grid_shape, cutoff2, fall=0.2, keep_shape=False
            )
            ft1[:] = ft1 * ftfltr1
            ft2[:] = ft2 * ftfltr2
            del ftfltr1, ftfltr2
    # print('in amplitude scale {0}'.format(ft1.dtype))
    # print('in amplitude scale {0}'.format(ft2.dtype))
    map1_scaled = np.zeros(emmap1.shape, dtype=np.float32)
    map2_scaled = np.zeros(emmap2.shape, dtype=np.float32)
    map1_scaled = ifft_obj(ifftshift(ft1, axes=(0, 1)), map1_scaled)
    map2_scaled = ifft_obj(ifftshift(ft2, axes=(0, 1)), map2_scaled)
    # print('in amplitude scale {0}'.format(map1_scaled.dtype))
    # print('in amplitude scale {0}'.format(map2_scaled.dtype))
    return map1_scaled, map2_scaled


def grid_footprint():
    m = np.zeros((3, 3, 3))
    m[1, 1, 1] = 1
    m[0, 1, 1] = 1
    m[1, 0, 1] = 1
    m[1, 1, 0] = 1
    m[2, 1, 1] = 1
    m[1, 2, 1] = 1
    m[1, 1, 2] = 1
    return m


def label_patches(mapdata, map_threshold, prob=0.1, inplace=False):
    # Taken from Agnel's code in TEMPy library / CCPEM
    ftpt = grid_footprint
    binmap = mapdata > float(map_threshold)
    label_array, labels = measurements.label(mapdata * binmap, structure=ftpt)
    sizes = measurements.sum(binmap, label_array, range(labels + 1))

    if labels <= 10:
        m_array = sizes < 0.05 * sizes.max()
        ct_remove = np.sum(m_array)
        remove_points = m_array[label_array]
        label_array[remove_points] = 0
        if inplace:
            mapdata[:] = (label_array > 0) * (mapdata * binmap)
        else:
            newmap = mapdata.copy()
            newmap = (label_array > 0) * (mapdata * binmap)
            return newmap, labels - ct_remove
        return labels - ct_remove + 1

    means = measurements.mean(mapdata * binmap, label_array, range(labels + 1))
    freq, bins = np.histogram(sizes[1:], 20)

    m_array = np.zeros(len(sizes))
    ct_remove = 0
    for i in range(len(freq)):
        f = freq[i]
        s2 = bins[i + 1]
        s1 = bins[i]
        p_size = float(f) / float(np.sum(freq))
        if p_size > prob:
            m_array = m_array + (
                (sizes >= s1)
                & (sizes < s2)
                & (
                    means
                    < (
                        float(map_threshold)
                        + 0.35 * (np.amax(mapdata) - float(map_threshold))
                    )
                )
            )
            ct_remove += 1
    m_array = m_array > 0
    remove_points = m_array[label_array]

    label_array[remove_points] = 0
    if inplace:
        mapdata[:] = (label_array > 0) * (mapdata * binmap)
    else:
        newmap = mapdata.copy()
        newmap = (label_array > 0) * (mapdata * binmap)
        return newmap, labels - ct_remove
    return labels - ct_remove


def calc_diffmap(
    emmap1,
    emmap2,
    res1,
    res2,
    grid1_info,
    grid2_info,
    lpfilt_pre=False,
    lpfilt_post=False,
    refscl=False,
    randsize=0.1,
    flag_dust=False,
    cyc=0,
    verbose=0,
    fft_obj=None,
    ifft_obj=None,
):
    """
    Calculate difference map, rewritten from Agnel's difference map calculation in CCPEM TEMpy

    """
    t1 = 2.0 if res1 > 4.0 else 2.5
    t2 = 2.0 if res2 > 4.0 else 2.5
    # convert data to np.array if necesary
    # if isinstance(emmap_1, gemmi.FloatGrid):
    #    emmap1 = np.array(emmap_1, copy=False, dtype=np.float32)
    # if isinstance(emmap_2, gemmi.FloatGrid):
    #    emmap2 = np.array(emmap_2, copy=False, dtype=np.float32)

    # get contours
    c1 = calculate_map_contour(emmap1, sigma_factor=t1)
    c2 = calculate_map_contour(emmap2, sigma_factor=t2)
    samegrid = False
    # check same grid dimensions
    try:
        assert np.broadcast(emmap1.shape, emmap2.shape)
        assert compare_tuple(grid1_info.grid_start, grid2_info.grid_start)
        assert compare_tuple(grid1_info.voxel_size, grid2_info.voxel_size)
        samegrid = True
    except AssertionError:
        samegrid = False

    # check spacing along all axes
    if (
        grid1_info.voxel_size[0] != grid1_info.voxel_size[1]
        or grid1_info.voxel_size[1] != grid1_info.voxel_size[2]
    ):
        samegrid = False
    if (
        grid2_info.voxel_size[0] != grid2_info.voxel_size[1]
        or grid2_info.voxel_size[1] != grid2_info.voxel_size[2]
    ):
        samegrid = False

    # global scaling of amplitudes
    if samegrid:
        if verbose >= 2:
            print("Matching maps amplitude")
        diff1, diff2 = amplitude_match(
            emmap1,
            emmap2,
            grid1_info,
            grid2_info,
            reso=max(res1, res2),
            lpfilt_pre=lpfilt_pre,
            lpfilt_post=lpfilt_post,
            ref_scale=refscl,
            fft_obj=fft_obj,
            ifft_obj=ifft_obj,
        )
        # min of minimas of two scaled maps
        min1 = diff1.min()
        min2 = diff2.min()
        min_scaled_maps = min(min1, min2)
        # shift to positive values
        if min_scaled_maps < 0.0:
            # make values non zero
            min_scaled_maps = min_scaled_maps + 0.05 * min_scaled_maps
            diff1[:] = diff1 + float(-min_scaled_maps)
            diff2[:] = diff2 + float(-min_scaled_maps)

        # calculate difference map
        if verbose >= 2:
            print("Calculating difference map")
        # store scaled maps
        scaledmap1 = diff1.copy()
        scaledmap2 = diff2.copy()
        # difference map Fo - Fc (map1 - experiment, map2 - calculated from model)
        diff1 = diff1 - diff2
        diff2 = diff2 - diff1

        # dust filter
        if flag_dust:
            label_patches(diff1, 0.0, prob=randsize, inplace=True)
            label_patches(diff2, 0.0, prob=randsize, inplace=True)
            diff1[:] = diff1 * (diff1 > 0.0)
            diff2[:] = diff2 * (diff2 > 0.0)

        # shift-back
        # scaledmap1[:] = scaledmap1 - float(-min_scaled_maps)
        # scaledmap2[:] = scaledmap2 - float(-min_scaled_maps)
        return scaledmap1, scaledmap2, diff1


def global_scale_maps(
    ref_map,
    target_map,
    ref_grid_info,
    target_grid_info,
    ref_map_reso,
    target_map_reso,
    lpfilt_pre=False,
    lpfilt_post=False,
    ref_scale=False,
    fft_obj=None,
    ifft_obj=None,
):
    if fft_obj is None:
        fft_obj, ifft_obj = sf_util.plan_fft_ifft(gridinfo=ref_grid_info)

    scale_ref, scale_tgt, = amplitude_match(
        ref_map,
        target_map,
        ref_grid_info,
        target_grid_info,
        reso=max(ref_map_reso, target_map_reso),
        lpfilt_pre=lpfilt_pre,
        lpfilt_post=lpfilt_post,
        ref_scale=ref_scale,
        fft_obj=fft_obj,
        ifft_obj=ifft_obj,
    )
    # min of minimas of two scaled maps
    min1 = scale_ref.min()
    min2 = scale_tgt.min()
    min_scaled_maps = min(min1, min2)
    # shift to positive values
    if min_scaled_maps < 0.0:
        # make values non zero
        min_scaled_maps = min_scaled_maps + 0.05 * min_scaled_maps
        scale_ref[:] = scale_ref + float(-min_scaled_maps)
        scale_tgt[:] = scale_tgt + float(-min_scaled_maps)

    return scale_ref, scale_tgt


def make_map_cubic(mapin, grid_info):
    max_len = -1
    max_ind = -1
    for i in range(0, 3):
        if grid_info.grid_shape[i] > max_len:
            max_len = grid_info.grid_shape[i]
            max_ind = i

    max_celldim = mapin.grid.unit_cell.parameters[max_ind]
    new_shape = np.array([max_len, max_len, max_len])
    map_data = resample_data_by_boxsize(mapin.grid, [max_len, max_len, max_len])
    apix = float(max_celldim / max_len)
    new_voxel_size = np.array([apix, apix, apix])
    new_gridinfo = GridInfo(
        new_shape, grid_info.grid_start, new_shape, new_voxel_size, grid_info.origin
    )
    newmap = gemmi.Ccp4Map()
    newmap.grid = gemmi.FloatGrid(map_data)
    newmap.grid.unit_cell.set(
        max_celldim,
        max_celldim,
        max_celldim,
        mapin.grid.unit_cell.alpha,
        mapin.grid.unit_cell.beta,
        mapin.grid.unit_cell.gamma,
    )
    newmap.grid.spacegroup = gemmi.SpaceGroup("P1")
    return newmap, new_gridinfo


if __name__ == "__main__":
    from pysheetbend.utils import fileio
    from pysheetbend.utils.dencalc import calculate_density_with_boxsize

    mapin = "/home/swh514/Projects/data/EMD-3488/map/emd_3488.map"
    # pdbin = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1_cryst1.ent'
    pdbin = "/home/swh514/Projects/work_and_examples/shiftfield/EMD-3488/4angs_lowres/translate_4angxyz.pdb"
    m, gridinfo = fileio.read_map(mapin)
    model, hetatm = fileio.get_structure(pdbin)
    apix0 = gridinfo.voxel_size
    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)
    res = 3.2
    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)  # so that the grid size matches original
    else:
        samp_rate = 1.5  # default; CLIPPER's oversampling parameter
    cmap = calculate_density_with_boxsize(model, res, samp_rate, gridinfo.grid_shape)
    fftobj = sf_util.plan_fft(gridinfo)
    ifftobj = sf_util.plan_ifft(gridinfo)

    scm1, scm2, dmap = calc_diffmap(
        m.grid,
        cmap,
        res,
        res,
        gridinfo,
        gridinfo,
        lpfilt_pre=True,
        lpfilt_post=False,
        refscl=False,
        randsize=0.1,
        flag_dust=False,
        verbose=6,
        fft_obj=fftobj,
        ifft_obj=ifftobj,
    )

    fileio.write_map_as_MRC(
        grid_data=scm1,
        unitcell=m.grid.unit_cell.parameters,
        outpath="scale_emd3488.mrc",
    )
    fileio.write_map_as_MRC(
        grid_data=scm2,
        unitcell=m.grid.unit_cell.parameters,
        outpath="scale_calcmap.mrc",
    )
    fileio.write_map_as_MRC(
        grid_data=dmap,
        unitcell=m.grid.unit_cell.parameters,
        outpath="diffmap.mrc",
    )
