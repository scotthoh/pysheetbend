# Python implementation of shift field refinement
# for coordinates and U-Iso values
# S.W.Hoh, University of York, 2020

import numpy as np
from numba import njit
from math import fabs
from pysheetbend.shiftfield_util import plan_fft_ifft
from pysheetbend.utils import map_funcs

# from memory_profiler import profile


def prepare_filter(radcyc, function, gridinfo, verbose):
    """
    Sets the radius and function to use
    Arguments
        radcyc: radius of the filter
        function: 0 = step, 1 = linear, 2 = quadratic
        densmap: reference density map
        verbose: verbosity
    Return
        N-D filter data array, weight
    """
    g_shape = gridinfo.grid_shape
    # determine effective radius of radial function
    rad = map_funcs.effective_radius(radcyc, function)
    fltr_data_r = np.zeros(g_shape, dtype=np.float32)
    # x,y,z convention
    nx, ny, nz = g_shape
    rad_x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    rad_y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    rad_z = np.arange(-np.floor(nz / 2.0), np.ceil(nz / 2.0))
    # shift negative indices to positive, periodic space
    rad_x[rad_x < 0] += g_shape[0]
    rad_y[rad_y < 0] += g_shape[1]
    rad_z[rad_z < 0] += g_shape[2]
    # shift center (array value 0.0) to edge
    rad_x -= g_shape[0] // 2
    rad_y -= g_shape[1] // 2
    rad_z -= g_shape[2] // 2
    rad_x *= gridinfo.voxel_size[0] + gridinfo.grid_start[0]
    rad_y *= gridinfo.voxel_size[1] + gridinfo.grid_start[1]
    rad_z *= gridinfo.voxel_size[2] + gridinfo.grid_start[2]
    xi, yi, zi = np.meshgrid(
        rad_x, rad_y, rad_z, sparse=True, indexing="ij", copy=False
    )
    r = np.sqrt(xi**2 + yi**2 + zi**2)
    r_bool = np.logical_and((r < rad), (r < radcyc))
    if function == "quadratic":
        rf = pow(1.0 - r[r_bool] / radcyc, 2)
    elif function == "linear":
        rf = 1.0 - r[r_bool] / radcyc
    elif function == "step":
        rf = np.full_like(r, fill_value=1.0)
    fltr_data_r[r_bool] = rf[:]
    f000 = np.sum(rf)
    # calc scale factor
    # scale = 1.0 / f000
    if verbose >= 2:
        print(" f000, ", f000)
    del r, r_bool
    return fltr_data_r, f000


# @profile
def cor_mod1(a, b):
    """
    Returns corrected remainder of division. If remainder <0,
    then adds value b to remainder.
    Arguments
    *a*
      array of Dividend (z,y,x indices)
    *b*
      array of Divisor (z,y,x indices)
    """
    c = np.fmod(a, b)
    d = np.transpose(np.nonzero(c < 0))
    # d, e = np.nonzero(c<0)
    for i in d:  # range(len(d)):
        c[i[0], i[1]] += b[i[1]]
        # c[i, j] += b[i]
    return c


# @profile
def mapfilter(data_arr, fltr_data_r, scale, fft_obj, ifft_obj, g_sam, g_reci):
    """
    Returns filtered data
    Argument
    *data_arr*
      array of data to be filtered
    *fft_obj*
      fft object
    *ifft_obj*
      ifft object
    """
    # copy map data and filter data
    # data_r = np.zeros(g_sam, dtype=np.float64)
    # print(data_arr.dtype)
    # print(fltr_data_r.dtype)
    # print(fft_obj)
    data_r = np.zeros(g_sam, dtype=np.float32)
    data_r = data_arr.copy()
    # fltr_input = np.zeros(g_sam, dtype=np.float64)
    fltr_input = np.zeros(g_sam, dtype=np.float32)
    fltr_input = fltr_data_r.copy()

    # if self.verbose >= 1:
    #    start = timer()
    # create complex data array
    # fltr_data_c = np.zeros(g_reci, dtype=np.complex128)
    # data_c = np.zeros(g_reci, dtype=np.complex128)
    fltr_data_c = np.zeros(g_reci, dtype=np.complex64)
    data_c = np.zeros(g_reci, dtype=np.complex64)
    # fourier transform of filter data
    fltr_data_c = fft_obj(fltr_input, fltr_data_c)
    fltr_data_c = fltr_data_c.conjugate().copy()
    # if self.verbose >= 1:
    #    end = timer()
    #    print('fft fltr_data : {0}s'.format(end-start))
    # if self.verbose >= 1:
    #    start = timer()
    # fourier transform of map data
    data_c = fft_obj(data_r, data_c)
    data_c = data_c.conjugate().copy()
    # if self.verbose >= 1:
    #    end = timer()
    #    print('fft data : {0}s'.format(end-start))
    # apply filter
    # if self.verbose >= 1:
    #    start = timer()
    data_c[:, :, :] = scale * data_c[:, :, :] * fltr_data_c[:, :, :]
    # if self.verbose >= 1:
    #    end = timer()
    #    print('Convolution : {0}s'.format(end-start))
    # inverse fft
    # if self.verbose >= 1:
    #    start = timer()
    # print(data_r.dtype, data_c.dtype)
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c, data_r)
    # if self.verbose >= 1:
    #    end = timer()
    #    print('ifft : {0}s'.format(end-start))
    return data_r


'''
def get_indices_zyx(origin, apix, array_shape):
    """
    Return gridtree and indices (z,y,x) convention
    Argument
    *densmap*
      Input density map
    """
    nz, ny, nx = array_shape

    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    zgc = zg * apix[2] + origin[2]
    ygc = yg * apix[1] + origin[1]
    xgc = xg * apix[0] + origin[0]

    indi = np.vstack([zgc.ravel(), ygc.ravel(), xgc.ravel()]).T

    # zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    indi = np.vstack([zg.ravel(), yg.ravel(), xg.ravel()]).T
    return gridtree, indi
'''


def gradient_map_calc(xdata_c, g_reci, g_real, g_half):
    ydata_c = np.zeros(xdata_c.shape, dtype=np.complex64)
    zdata_c = np.zeros(xdata_c.shape, dtype=np.complex64)
    i = complex(0.0, 1.0)
    if not g_reci[0] % 2:  # even
        xi = np.arange(-np.floor(g_reci[0] / 2.0), np.ceil(g_reci[0] / 2.0))
        yi = np.arange(-np.floor(g_reci[1] / 2.0), np.ceil(g_reci[1] / 2.0))
    else:  # odd
        xi = np.arange(-np.ceil(g_reci[0] / 2.0), np.floor(g_reci[0] / 2.0))
        yi = np.arange(-np.ceil(g_reci[1] / 2.0), np.floor(g_reci[1] / 2.0))
    zi = np.arange(0, g_reci[2])
    zi += g_half[2]
    zi = np.fmod(zi, g_real[2])
    xi[xi < 0] += g_real[0]
    yi[yi < 0] += g_real[1]
    zi[zi < 0] += g_real[2]
    xi -= g_half[0]
    yi -= g_half[1]
    zi -= g_half[2]
    cdata = i * xdata_c
    ih, ik, il = np.meshgrid(xi, yi, zi, indexing="ij", sparse=True, copy=False)
    xdata_c = ((2 * np.pi) * ih).astype(float) * cdata
    ydata_c = ((2 * np.pi) * ik).astype(float) * cdata
    zdata_c = ((2 * np.pi) * il).astype(float) * cdata

    return xdata_c, ydata_c, zdata_c


def gradient_map_iso_calc(xdata_c, g_real, g_half, cell):
    i = complex(0.0, 1.0)  # noqa E741
    g_reci = xdata_c.shape
    if not g_reci[0] % 2:  # even
        xi = np.arange(-np.floor(g_reci[0] / 2.0), np.ceil(g_reci[0] / 2.0))
        yi = np.arange(-np.floor(g_reci[1] / 2.0), np.ceil(g_reci[1] / 2.0))
    else:  # odd
        xi = np.arange(-np.ceil(g_reci[0] / 2.0), np.floor(g_reci[0] / 2.0))
        yi = np.arange(-np.ceil(g_reci[1] / 2.0), np.floor(g_reci[1] / 2.0))
    zi = np.arange(0, g_reci[2])
    zi += g_half[2]
    zi = np.fmod(zi, g_real[2])
    xi[xi < 0] += g_real[0]
    yi[yi < 0] += g_real[1]
    zi[zi < 0] += g_real[2]
    xi -= g_half[0]
    yi -= g_half[1]
    zi -= g_half[2]
    ih, ik, il = np.meshgrid(xi, yi, zi, indexing="ij", sparse=True, copy=False)
    scl = (
        ih * ih * (1.0 / (cell[0] * cell[0]))
        + ik * ik * (1.0 / (cell[1] * cell[1]))
        + il * il * (1.0 / (cell[2] * cell[2]))
    )
    scl = (2 * np.pi * np.pi) * scl
    xdata_c = scl * xdata_c

    return xdata_c


# @profile
# @njit()
# def gradient_map_calc_old(xdata_c, g_reci, g_real, ch):
#     # ydata_c = np.zeros(xdata_c.shape, dtype=np.complex128)
#     # zdata_c = np.zeros(xdata_c.shape, dtype=np.complex128)
#     ydata_c = np.zeros(xdata_c.shape, dtype=np.complex64)
#     zdata_c = np.zeros(xdata_c.shape, dtype=np.complex64)
#
#     i = complex(0.0, 1.0)
#     for cx in range(g_reci[0]):  # x
#         for cy in range(g_reci[1]):  # y
#             for cz in range(g_reci[2]):  # z
#                 hkl = hkl_c((cx, cy, cz), ch, g_real)  # returned index x,y,z
#                 cdata = i * xdata_c[cx, cy, cz]
#                 xdata_c[cx, cy, cz] = float((2 * np.pi) * hkl[0]) * cdata
#                 ydata_c[cx, cy, cz] = float((2 * np.pi) * hkl[1]) * cdata
#                 zdata_c[cx, cy, cz] = float((2 * np.pi) * hkl[2]) * cdata
#     return xdata_c, ydata_c, zdata_c
#
#
# @njit()
# def gradient_map_iso_calc_old(data_c, g_reci, g_real, ch, cell):
#     for cx in range(0, g_reci[0]):  # x
#         for cy in range(0, g_reci[1]):  # y
#             for cz in range(0, g_reci[2]):  # z
#                 hkl = hkl_c((cx, cy, cz), ch, g_real)
#                 scl = (2 * np.pi * np.pi) * metric_reci_lengthsq(
#                     hkl[0], hkl[1], hkl[2], cell
#                 )
#                 data_c[cx, cy, cz] = scl * data_c[cx, cy, cz]
#     return data_c


# @profile
def shift_field_coord(
    cmap,
    dmap,
    mmap,
    rad,
    function_type,
    gridinfo,
    fft_obj,
    ifft_obj,
    cyc,
    verbose=0,
    timelog=None,
):
    """
    Returns 3 map instances for shifts in x,y,z directions
    Performs shift field refinement on coordinates.
    Arguments:
    *cmap*
      Calculated map from input structure using esf_map_calc.py
    *dmap*
      Difference map between observed and calculated
    *mask*
      Mask map
    *rad*
      Radius
    *fltr*
      Filter type 2=quadratic, 1=linear, 0=step
    *fft_obj*
      Planned fft object
    *ifft_obj*
      Planned ifft object
    *verbose*
      verbose >= 1 to print out some time profiling
    """
    # tracemalloc.start()
    # set the numbers for the grids in list form
    g_reci = gridinfo.grid_reci
    g_shape = gridinfo.grid_shape
    g_half = gridinfo.grid_half

    # calculate map coefficients
    data_r = cmap.copy()
    timelog.start("FFTCalc")
    xdata_c = fft_obj(data_r)
    xdata_c = xdata_c.conjugate().copy()
    timelog.end("FFTCalc")
    # calculate gradient map coefficients
    xdata_c, ydata_c, zdata_c = gradient_map_calc(xdata_c, g_reci, g_shape, g_half)

    # calculate gradient maps
    # fft complex to real
    zdata_c = zdata_c.conjugate().copy()
    ydata_c = ydata_c.conjugate().copy()
    xdata_c = xdata_c.conjugate().copy()
    zdata_r = np.zeros(g_shape, dtype=np.float32)
    ydata_r = np.zeros(g_shape, dtype=np.float32)
    xdata_r = np.zeros(g_shape, dtype=np.float32)
    timelog.start("IFFTCalc")
    zdata_r = ifft_obj(zdata_c, zdata_r)
    ydata_r = ifft_obj(ydata_c, ydata_r)
    xdata_r = ifft_obj(xdata_c, xdata_r)
    timelog.end("IFFTCalc")
    # if verbose >= 10:
    #    writeMap(xdata_r, origin, apix, cmap.shape, f"gradx1map_{cyc}.map")
    #    writeMap(ydata_r, origin, apix, cmap.shape, f"gradx2map_{cyc}.map")
    #    writeMap(zdata_r, origin, apix, cmap.shape, f"gradx3map_{cyc}.map")

    # calculate XTY
    y1map = xdata_r * dmap
    y2map = ydata_r * dmap
    y3map = zdata_r * dmap
    # calculate XTX
    x11map = xdata_r * xdata_r
    x12map = xdata_r * ydata_r
    x13map = xdata_r * zdata_r
    x22map = ydata_r * ydata_r
    x23map = ydata_r * zdata_r
    x33map = zdata_r * zdata_r
    timelog.start("Filter")
    fltr_data_r, f000 = map_funcs.make_filter_edge_centered(
        grid_info=gridinfo, filter_radius=rad, function=function_type, verbose=verbose
    )
    # if verbose >= 3:
    #    print(f"ymap shape {ymap.shape}")
    #    filt_ymap = mapfilter(
    #        ymap, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci
    #    )
    #    writeMap(filt_ymap, origin, apix, cmap.shape, f"filt_ymap{cyc}.map")
    y1map = map_funcs.fft_convolution_filter(
        y1map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    y2map = mapfilter(
        y2map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    y3map = mapfilter(
        y3map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x11map = mapfilter(
        x11map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x12map = mapfilter(
        x12map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x13map = mapfilter(
        x13map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x22map = mapfilter(
        x22map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x23map = mapfilter(
        x23map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x33map = mapfilter(
        x33map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    timelog.end("Filter")
    """
    if verbose >= 6:
        writeMap(x11map, origin, apix, cmap.shape, f"filt_x11map_{cyc}.map")
        writeMap(x12map, origin, apix, cmap.shape, f"filt_x12map_{cyc}.map")
        writeMap(x13map, origin, apix, cmap.shape, f"filt_x13map_{cyc}.map")
        writeMap(x22map, origin, apix, cmap.shape, f"filt_x22map_{cyc}.map")
        writeMap(x23map, origin, apix, cmap.shape, f"filt_x23map_{cyc}.map")
        writeMap(x33map, origin, apix, cmap.shape, f"filt_x33map_{cyc}.map")
        writeMap(y1map, origin, apix, cmap.shape, f"filt_y1map_{cyc}.map")
        writeMap(y2map, origin, apix, cmap.shape, f"filt_y2map_{cyc}.map")
        writeMap(y3map, origin, apix, cmap.shape, f"filt_y3map_{cyc}.map")
    """
    # calculate U shifts
    mmap_f = np.ravel(mmap)
    mat_size = np.count_nonzero(~mmap_f)
    timelog.start("SetMatrix")
    # these use a lot of memory i think, see how to reduce
    m = np.zeros((mat_size, 3, 3), dtype=np.float32)
    v = np.zeros((mat_size, 3), dtype=np.float32)
    x11_f = np.ravel(x11map)[~mmap_f]
    x12_f = np.ravel(x12map)[~mmap_f]
    x13_f = np.ravel(x13map)[~mmap_f]
    x22_f = np.ravel(x22map)[~mmap_f]
    x23_f = np.ravel(x23map)[~mmap_f]
    x33_f = np.ravel(x33map)[~mmap_f]
    y1map_f = np.ravel(y1map)[~mmap_f]
    y2map_f = np.ravel(y2map)[~mmap_f]
    y3map_f = np.ravel(y3map)[~mmap_f]
    # flatten arrays and assign to matrix and vector
    v[:, 0] = y1map_f
    v[:, 1] = y2map_f
    v[:, 2] = y3map_f
    m[:, 0, 0] = x11_f
    m[:, 0, 1] = x12_f
    m[:, 1, 0] = x12_f
    m[:, 0, 2] = x13_f
    m[:, 2, 0] = x13_f
    m[:, 1, 1] = x22_f
    m[:, 1, 2] = x23_f
    m[:, 2, 1] = x23_f
    m[:, 2, 2] = x33_f
    timelog.end("SetMatrix")
    timelog.start("LINALG")
    try:
        v[:] = np.linalg.solve(m[:], v[:])
    except np.linalg.LinAlgError:
        for im in range(len(m)):
            v[im] = np.linalg.lstsq(m[im], v[im], rcond=None)[0]
    timelog.end("LINALG")
    x1map_f = np.zeros(cmap.size, dtype=np.float32)
    x2map_f = np.zeros(cmap.size, dtype=np.float32)
    x3map_f = np.zeros(cmap.size, dtype=np.float32)
    x1map_f[~mmap_f] = v[:, 0]
    x2map_f[~mmap_f] = v[:, 1]
    x3map_f[~mmap_f] = v[:, 2]
    # reassign values back to fullMaps
    x1map_r = x1map_f.reshape(cmap.shape)
    x2map_r = x2map_f.reshape(cmap.shape)
    x3map_r = x3map_f.reshape(cmap.shape)
    del m
    del v
    return x1map_r, x2map_r, x3map_r


@njit()
def metric_reci_lengthsq(x, y, z, celldim):
    """
    (x*(x*a'*a' + y*2.0*a'*b'*np.cos(gam') + z*2.0*a'*c'*np.cos(bet'))
     + y*(y*b'*b' + z*2.0*b'*c'*np.cos(alp'))
     + z*(z*c'*c'))
    = x*x*a'*a' + y*y*b'*b' + z*z*c'*c' when alp, bet, gam = 90; cos90 = 0
    """
    scl = (
        x * x * (1.0 / (celldim[0] * celldim[0]))
        + y * y * (1.0 / (celldim[1] * celldim[1]))
        + z * z * (1.0 / (celldim[2] * celldim[2]))
    )
    return scl


def shift_field_uiso(
    cmap,
    dmap,
    mmap,
    rad,
    function_type,
    gridinfo,
    unitcell,
    fft_obj=None,
    ifft_obj=None,
    verbose=0,
    timelog=None,
):
    """
    Performs shift field refinement on isotropic U values
    Arguments:
    *cmap*
      Calculated map from input structure using ESF
    *dmap*
      Difference map between observed and calculated
    *mmap*
      Mask map
    *rad*
      Radius
    *function_type*
      Filter type 2=quadratic, 1=linear, 0=step
    *gridinfo*
        GridInfo class containing grid info
    *fft_obj*
      Planned fft object
    *ifft_obj*
      Planned ifft object
    """
    g_shape = gridinfo.grid_shape
    g_half = gridinfo.grid_half
    if fft_obj is None:
        fft_obj, ifft_obj = plan_fft_ifft(gridinfo=gridinfo)
    # fullMap is numpy array
    data_r = cmap
    timelog.start("FFTCalc")
    data_c = fft_obj(data_r)
    data_c = data_c.conjugate().copy()
    timelog.end("FFTCalc")
    # calculate gradient map coefficients
    data_c = gradient_map_iso_calc(data_c, g_shape, g_half, unitcell)
    # calculate gradient maps
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c)
    y1map = data_r * dmap
    # calculate XTX
    x11map = data_r * data_r
    # filter maps
    fltr_data_r, f000 = map_funcs.make_filter_edge_centered(
        grid_info=gridinfo, filter_radius=rad, function="quadratic", verbose=verbose
    )
    y1map = map_funcs.fft_convolution_filter(
        y1map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    x11map = mapfilter(
        x11map, fltr_data_r, gridinfo, f000, fft_obj=fft_obj, ifft_obj=ifft_obj
    )
    # calculate U shifts
    data_r[~mmap] = y1map[~mmap] / x11map[~mmap]
    data_r[mmap] = 0.0
    return data_r


@njit
def solve(m, v):
    # l = len(m)  # no of points
    # check if matrix is square
    l, r, c = m.shape
    vl, vr = v.shape

    if r != c:
        # print('Matrix not square')
        raise TypeError("Matrix not square")
    if r != vr:
        raise TypeError("Matrix/Vector mismatch")

    a = m.copy()
    x = v.copy()
    for p in range(0, l):
        for i in range(0, r):
            # pick largest pivot
            j = i
            for k in range(i + 1, r):
                if fabs(a[p, k, i]) > fabs(a[p, j, i]):
                    j = k
            # swap rows
            for k in range(0, r):
                a[p, i, k], a[p, j, k] = a[p, j, k], a[p, i, k]
            x[p, i], x[p, j] = x[p, j], x[p, i]
            # perform elimination
            pivot = a[p, i, i]
            for j in range(0, r):
                if j != i:
                    s = a[p, j, i] / pivot
                    for k in range(i + 1, r):
                        a[p, j, k] = a[p, j, k] - s * a[p, i, k]
                    x[p, j] = x[p, j] - s * x[p, i]
        for i in range(0, r):
            x[p, i] /= a[p, i, i]
    return x
