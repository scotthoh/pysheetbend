# Python implementation of shift field refinement
# for coordinates and U-Iso values
# S.W.Hoh, University of York, 2020

# from TEMPy.protein.structure_blurrer import StructureBlurrer

# from TEMPy.maps.em_map import Map
# from TEMPy.math.vector import Vector
from logging import raiseExceptions
import numpy as np
from timeit import default_timer as timer
from numba import njit
from math import fabs

from TEMPy.EMMap import Map
from TEMPy.Vector import Vector

# from TEMPy.maps.em_map import Map
from numpy import linalg

# import tracemalloc

# from numpy.linalg.linalg import solve
import numpy.ma as ma

# using numba here


@njit()
def fltr(r, radius, function=2):
    """
    Returns radius value from filter function
    Arguments
    *r*
      radius
    """
    if r < radius:
        if function == 2:
            return pow(1.0 - r / radius, 2)
        elif function == 1:
            return 1.0 - r / radius
        elif function == 0:
            return 1.0
    else:
        return 0.0


# @profile
@njit()
def effective_radius(func, radius):
    nrad = 1000
    drad = 0.25
    sum_r = np.zeros(nrad)
    for i in range(0, nrad):
        r = drad * (float(i) + 0.5)
        sum_r[i] = r * r * fabs(fltr(r, radius, func))
        if i >= 1:
            sum_r[i] += sum_r[i - 1]

    # for i in range(1, nrad):
    #    sum_r[i] += sum_r[i-1]

    for i in range(0, nrad):
        if sum_r[i] > 0.99 * sum_r[nrad - 1]:
            break

    return drad * (float(i) + 1.0)


# @profile
@njit()
def radial_filter_func(indi, fltr_data_r, r, r_ind, fltr_rad, func):
    count = 0
    f000 = 0.0
    # fill the radial function map
    print("in radial")

    for i in r_ind[0]:
        rf = fltr(r[i], fltr_rad, func)
        f000 += rf
        # print(gt[1][i][0], gt[1][i][1], gt[1][i][2], rf)
        count += 1
        fltr_data_r[indi[i][0], indi[i][1], indi[i][2]] = rf  # [i]
    print("end radial")

    return fltr_data_r, f000


# @profile
def prepare_filter(radcyc, function, g_reci, g_sam, g_half, origin, apix, array_shape):
    """
    Sets the radius and function to use
    Arguments
    *radcyc*
      radius of the filter
    *function*
      0 = step, 1 = linear, 2 = quadratic
    *densmap*
      reference density map
    """
    verbose = 0
    fltr_rad = radcyc
    func = function
    # g_reci = np.array(g_reci)
    # g_sam = np.array(g_sam)
    # g_half = np.array(g_half)
    """if function == 'step':
        self.function = 0
    elif function == 'linear':
        self.function = 1
    elif function == 'quadratic':
        self.function = 2
    """
    # function = step, linear, quadratic
    # determine effective radius of radial function
    # self.gridshape = GridDimension(densmap)
    start = timer()
    rad = effective_radius(function, fltr_rad)
    end = timer()
    print("Effective radius : {0}s".format(end - start))
    fltr_data_r = np.zeros(g_sam, dtype="float64")
    # fltr_data_r = np.zeros(g_sam, dtype='float32')

    # z,y,x convention
    # origin = np.array((origin[2], origin[1], origin[0]))
    if isinstance(apix, tuple):
        apix = np.array([apix[2], apix[1], apix[0]])
    else:
        apix = np.array([apix, apix, apix])
    # g_half = (g_real[0]//2, g_real[1]//2, g_real[0]//2+1)
    # SB = StructureBlurrer()
    # gt = SB.maptree(densmap)
    print("1")
    nz, ny, nx = array_shape
    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    indi = np.vstack([zg.ravel(), yg.ravel(), xg.ravel()]).T
    g_sam = np.array(g_sam)
    g_half = np.array(g_half)
    print("2")

    c = indi + g_half  # self.gridshape.g_half
    print("2a")
    c1 = cor_mod1(c, g_sam) - g_half
    print("2b")

    pos = c1[:] * apix + origin
    print("2c")
    r = np.sqrt(np.sum(np.square(pos), axis=1))
    print("2d")
    r_ind = np.nonzero(r < rad)
    print("3")

    start = timer()
    fltr_data_r, f000 = radial_filter_func(indi, fltr_data_r, r, r_ind, fltr_rad, func)
    end = timer()
    print("Fltr data r : {0}s".format(end - start))

    # calc scale factor
    scale = 1.0 / f000
    if verbose >= 1:
        print("scale, ", scale, " f000, ", f000)
    del c1
    return fltr_data_r, scale


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
    data_r = np.zeros(g_sam, dtype="float64")
    # data_r = np.zeros(g_sam, dtype='float32')
    data_r = data_arr.copy()
    fltr_input = np.zeros(g_sam, dtype="float64")
    # fltr_input = np.zeros(g_sam, dtype='float32')
    fltr_input = fltr_data_r.copy()

    # if self.verbose >= 1:
    #    start = timer()
    # create complex data array
    fltr_data_c = np.zeros(g_reci, dtype="complex128")
    data_c = np.zeros(g_reci, dtype="complex128")
    # fltr_data_c = np.zeros(g_reci, dtype='complex64')
    # data_c = np.zeros(g_reci, dtype='complex64')
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


# @profile
@njit()
def gradient_map_calc(xdata_c, g_reci, g_real, ch):
    ydata_c = np.zeros(xdata_c.shape, dtype="complex128")
    zdata_c = np.zeros(xdata_c.shape, dtype="complex128")
    # ydata_c = np.zeros(xdata_c.shape, dtype='complex64')
    # zdata_c = np.zeros(xdata_c.shape, dtype='complex64')

    i = complex(0.0, 1.0)
    for cz in range(g_reci[0]):  # z
        for cy in range(g_reci[1]):  # y
            for cx in range(g_reci[2]):  # x
                hkl = hkl_c((cz, cy, cx), ch, g_real)
                cdata = i * xdata_c[cz, cy, cx]
                zdata_c[cz, cy, cx] = float((2 * np.pi) * hkl[0]) * cdata
                ydata_c[cz, cy, cx] = float((2 * np.pi) * hkl[1]) * cdata
                xdata_c[cz, cy, cx] = float((2 * np.pi) * hkl[2]) * cdata
    return zdata_c, ydata_c, xdata_c


@njit()
def gradient_map_iso_calc(data_c, g_reci, g_real, ch, cell):
    for cz in range(0, g_reci[0]):  # z
        for cy in range(0, g_reci[1]):  # y
            for cx in range(0, g_reci[2]):  # x
                hkl = hkl_c((cz, cy, cx), ch, g_real)
                scl = (2 * np.pi * np.pi) * metric_reci_lengthsq(
                    hkl[2], hkl[1], hkl[0], cell
                )
                data_c[cz, cy, cx] = scl * data_c[cz, cy, cx]
    return data_c


@njit()
def hkl_c(c, ch, g):
    """
    Returns the index (z,y,x)
    Arguments
    *c*
      Index h,k,l
    *ch*
      Half of the grid shape
    *g*
      Real space grid shape
    """
    # z, y, x convention
    cv = np.array((int(c[0]), int(c[1]), int(c[2])))
    chv = np.array((int(ch[0]), int(ch[1]), int(ch[2])))
    v1 = cv + chv
    m1 = np.array((cor_mod(v1[0], g[0]), cor_mod(v1[1], g[1]), cor_mod(v1[2], g[2])))

    return m1 - chv


@njit()
def cor_mod(a, b):
    """
    Returns corrected remainder of division. If remainder <0,
    then adds value b to remainder.
    Arguments
    *a*
      Dividend
    *b*
      Divisor
    """
    c = np.fmod(a, b)
    if c < 0:
        c += b
    return int(c)


# @njit()
# def solvelinalg(m, v):
#  for i in range(m.shape[0]):
#    v[i] = np.linalg.solve(m[i], v[i])

#  return v


def writeMap(data_array, origin, apix, shape, fname):
    mapobj = Map(
        np.zeros((shape[2], shape[1], shape[0])),
        origin,
        apix,
        "mapname",
    )
    mapobj.fullMap = data_array.copy()
    mapobj.update_header()
    mapobj.write_to_MRC_file(fname)


# @profile
def shift_field_coord(
    cmap, dmap, mmap, rad, fltr, origin, apix, fft_obj, ifft_obj, cyc, verbose=0
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
    g_reci = (cmap.shape[2], cmap.shape[1], cmap.shape[0] // 2 + 1)
    # g_real = (g_reci[0], g_reci[1], int(g_reci[2]-1)*2)
    g_real = (cmap.shape[2], cmap.shape[1], cmap.shape[0])
    ch = (g_real[0] // 2, g_real[1] // 2, g_real[2] // 2)

    # calculate map coefficients
    data_r = cmap.copy()
    if verbose >= 1:
        start = timer()
    xdata_c = fft_obj(data_r)
    xdata_c = xdata_c.conjugate().copy()
    if verbose >= 1:
        end = timer()
        print("FFT Calc Map : {0} s".format(end - start))

    # calculate gradient map coefficients
    zdata_c, ydata_c, xdata_c = gradient_map_calc(xdata_c, g_reci, g_real, ch)

    # calculate gradient maps
    # fft complex to real
    if verbose >= 1:
        start = timer()
    zdata_c = zdata_c.conjugate().copy()
    ydata_c = ydata_c.conjugate().copy()
    xdata_c = xdata_c.conjugate().copy()
    if verbose >= 1:
        end = timer()
        print("3x conjugate : {0} s".format(end - start))
    zdata_r = np.zeros(data_r.shape, dtype="float64")
    ydata_r = np.zeros(data_r.shape, dtype="float64")
    xdata_r = np.zeros(data_r.shape, dtype="float64")
    # zdata_r = np.zeros(data_r.shape, dtype='float32')
    # ydata_r = np.zeros(data_r.shape, dtype='float32')
    # xdata_r = np.zeros(data_r.shape, dtype='float32')

    if verbose >= 1:
        start = timer()
    zdata_r = ifft_obj(zdata_c, zdata_r)
    if verbose >= 1:
        end = timer()
        print("first ifft ", end - start)
    ydata_r = ifft_obj(ydata_c, ydata_r)
    xdata_r = ifft_obj(xdata_c, xdata_r)
    if verbose >= 6:
        writeMap(xdata_r, origin, apix, cmap.shape, f"gradx1map_{cyc}.map")
        writeMap(ydata_r, origin, apix, cmap.shape, f"gradx2map_{cyc}.map")
        writeMap(zdata_r, origin, apix, cmap.shape, f"gradx3map_{cyc}.map")

    # print(x1map.apix, x2map.apix, x3map.apix)
    # print(x1map.origin, x2map.origin, x3map.origin)
    # copy map
    x1map_r = xdata_r.copy()
    x2map_r = ydata_r.copy()
    x3map_r = zdata_r.copy()
    # end map preparation
    # dmap = dmap.astype('float64')
    # calculate XTY and apply mask
    # ymap=diffmap , mmap = mask
    # ymap = np.zeros(dmap.shape)
    # mmap = np.zeros(mask.shape)
    if verbose >= 3:
        writeMap(dmap, origin, apix, cmap.shape, f"ymap_{cyc}.map")

    ymap = dmap.copy()
    # mmap = mask.copy()
    ymap_m = ma.masked_array(ymap, mask=mmap).data
    x1map_m = ma.masked_array(x1map_r, mask=mmap).data
    # x1map_r*ymap*mmap
    x2map_m = ma.masked_array(x2map_r, mask=mmap).data
    x3map_m = ma.masked_array(x3map_r, mask=mmap).data

    y1map = x1map_m * ymap_m
    # x1map_r*ymap*mmap
    y2map = x2map_m * ymap_m
    y3map = x3map_m * ymap_m
    # y2map = x2map_r*ymap*mmap
    # y3map = x3map_r*ymap*mmap

    # print(np.count_nonzero(y1map==0.0), np.count_nonzero(y2map==0.0), np.count_nonzero(y3map==0.0))
    # calculate XTX  (removed multiply with mask)
    x11map = x1map_r * x1map_r
    # print(np.array_equal(x11map, x1map.fullMap))
    # was  x12map = x1map_r*x2map_r
    x12map = x1map_r * x2map_r
    x13map = x1map_r * x3map_r
    x22map = x2map_r * x2map_r
    x23map = x2map_r * x3map_r
    x33map = x3map_r * x3map_r
    if verbose >= 6:
        writeMap(x11map, origin, apix, cmap.shape, f"x11map_{cyc}.map")
        writeMap(x12map, origin, apix, cmap.shape, f"x12map_{cyc}.map")
        writeMap(x13map, origin, apix, cmap.shape, f"x13map_{cyc}.map")
        writeMap(x22map, origin, apix, cmap.shape, f"x22map_{cyc}.map")
        writeMap(x23map, origin, apix, cmap.shape, f"x23map_{cyc}.map")
        writeMap(x33map, origin, apix, cmap.shape, f"x33map_{cyc}.map")
        writeMap(y1map, origin, apix, cmap.shape, f"y1map_{cyc}.map")
        writeMap(y2map, origin, apix, cmap.shape, f"y2map_{cyc}.map")
        writeMap(y3map, origin, apix, cmap.shape, f"y3map_{cyc}.map")
    # y1map = y1map.astype('float32')
    # y2map = y2map.astype('float32')
    # y3map = y3map.astype('float32')

    # filter
    # x33map1 = Map(np.zeros(cmap.fullMap.shape),
    #              cmap.origin,
    #              cmap.apix[0],
    #              'mapname',)
    # x33map1.fullMap = x33map
    # x33map1.update_header()
    # x33map1.write_to_MRC_file('x33_map1.map')

    # dmap.set_apix_tempy()
    start = timer()
    fltr_data_r, scale = prepare_filter(
        rad, fltr, g_reci, g_real, ch, origin, apix, dmap.shape
    )
    # if verbose >= 3:
    #    print(f"ymap shape {ymap.shape}")
    #    filt_ymap = mapfilter(
    #        ymap, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci
    #    )
    #    writeMap(filt_ymap, origin, apix, cmap.shape, f"filt_ymap{cyc}.map")
    # dmap_gt = SB.maptree(dmap)
    y1map = mapfilter(y1map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    y2map = mapfilter(y2map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    y3map = mapfilter(y3map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x11map = mapfilter(x11map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x12map = mapfilter(x12map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x13map = mapfilter(x13map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x22map = mapfilter(x22map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x23map = mapfilter(x23map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x33map = mapfilter(x33map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    end = timer()
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
    print("Filter maps : {0} s".format(end - start))

    # x33map1.fullMap = x33map
    # x33map1.update_header()
    # x33map1.write_to_MRC_file('x33_map_filtered.map')
    # calculate U shifts

    start = timer()
    # these use a lot of memory. see how to reduce
    m = np.zeros((cmap.size, 3, 3))
    v = np.zeros((cmap.size, 3))
    x11_f = np.ravel(x11map)
    x12_f = np.ravel(x12map)
    x13_f = np.ravel(x13map)
    x22_f = np.ravel(x22map)
    x23_f = np.ravel(x23map)
    x33_f = np.ravel(x33map)
    y1map_f = np.ravel(y1map)
    y2map_f = np.ravel(y2map)
    y3map_f = np.ravel(y3map)
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
    print("check determinant")
    # det_m = np.linalg.det(m)
    # count_m0 = 0
    # for im in det_m:
    #  if im == 0:
    #    count_m0 += 1
    # print(count_m0)
    end = timer()
    print("Set matrix : {0} s".format(end - start))
    start = timer()
    v[:] = np.linalg.solve(m[:], v[:])
    # v = solve(m, v)
    # v[:], residuals, rank, s = np.linalg.lstsq(m[:], v[:])

    # v = solvelinalg(m[:], v[:])
    end = timer()
    print("Solve linalg : {0} s".format(end - start))
    start = timer()
    x1map_f = v[:, 0]
    x2map_f = v[:, 1]
    x3map_f = v[:, 2]
    # reassign values back to fullMaps
    x1map_r = x1map_f.reshape(cmap.shape)
    x2map_r = x2map_f.reshape(cmap.shape)
    x3map_r = x3map_f.reshape(cmap.shape)
    """for index in range(0, cmap.fullMap.size):
        pos = mf.gt[1][index]  # gt from maptree_zyx
        p_zyx = (pos[0], pos[1], pos[2])
        x1map.fullMap[p_zyx] = v[index][0]  # .x  # [0]
        x2map.fullMap[p_zyx] = v[index][1]  # .y  # [1]
        x3map.fullMap[p_zyx] = v[index][2]  # .z  # [2]
    """
    # end = timer()
    # print('Copy answer : {0} s'.format(end-start))
    """
    for index in range(0, cmap.fullMap.size):
        # vector v
        pos = mf.gt[1][index]  # gt from maptree_zyx
        p_zyx = (pos[0], pos[1], pos[2])
        #v[index][0] = np.array([y1map[p_zyx], y2map[p_zyx], y3map[p_zyx]])  # Vector(x, y, z) 
        v[index][0] = y1map[p_zyx]  # Vector(x, y, z) 
        v[index][1] = y2map[p_zyx]  # Vector(x, y, z) 
        v[index][2] = y3map[p_zyx]  # Vector(x, y, z) 
        # matrix(3,3)
        #m = np.zeros([3, 3])
        m[index][0, 0] = x11map[p_zyx]
        m[index][0, 1] = x12map[p_zyx]
        m[index][1, 0] = x12map[p_zyx]
        m[index][0, 2] = x13map[p_zyx]
        m[index][2, 0] = x13map[p_zyx]
        m[index][1, 1] = x22map[p_zyx]
        m[index][1, 2] = x23map[p_zyx]
        m[index][2, 1] = x23map[p_zyx]
        m[index][2, 2] = x33map[p_zyx]
        # solve matrix v c++ m.solve(v)
        # solve linear eq Ax=b for x
        # can use numpy.linalg.solve
        
        #try:
        #    v = np.linalg.solve(m, v)
        #except np.linalg.LinAlgError:
        #    count += 1
        #    pass  # print(v)
    """

    # 2 - z and x axis swap in np array becomes x,y,z
    # 3 - back using z,y,x convention by tempy map
    # 4 - fixed hkl cv - chv to cv + chv ... no change since return modulus
    # 5 - tanh_lowpass filter
    # 6 - init x1,x2,x3maps with lowpass map apix
    # make maps
    # print('linalg err num: {0}'.format(count))
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #    print(stat)
    del m
    del v
    return x1map_r, x2map_r, x3map_r


@njit
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
    cmap, dmap, mmap, rad, fltr, origin, apix, fft_obj, ifft_obj, g_cell, verbose=0
):
    """
    Performs shift field refinement on isotropic U values
    Arguments:
    *cmap*
      Calculated map from input structure using ESF
    *dmap*
      Difference map between observed and calculated
    *x1map*
      x1map from previous shift_field_coords calculation
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
    """
    g_reci = (cmap.shape[2], cmap.shape[1], cmap.shape[0] // 2 + 1)
    g_real = (cmap.shape[2], cmap.shape[1], cmap.shape[0])
    ch = (g_real[0] // 2, g_real[1] // 2, g_real[2] // 2)
    # fullMap is numpy array
    data_r = cmap
    if verbose >= 1:
        start = timer()
    data_c = fft_obj(data_r)
    data_c = data_c.conjugate().copy()
    if verbose >= 1:
        end = timer()
        print("first ft ", end - start)
    # calculate gradient map coefficients
    data_c = gradient_map_iso_calc(data_c, g_reci, g_real, ch, g_cell)
    """for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = sf_util.hkl_c((cz, cy, cx), ch, g_real_)
                scl = (2*np.pi*np.pi)*g_cell.metric_reci_lengthsq(hkl.x, hkl.y, hkl.z)
                data_c[cz, cy, cx] = scl * data_c[cz, cy, cx]"""

    # calculate gradient maps
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c)
    x1map = data_r.copy()

    # make xmap
    # ymap = np.zeros(dmap.fullMap.shape)
    # mmap = np.zeros(mask.fullMap.shape)
    # ymap = dmap.fullMap.copy()

    ymap_m = ma.masked_array(dmap, mask=mmap).data
    x1map_m = ma.masked_array(x1map, mask=mmap).data
    # calculate XTY apply mask
    y1map = x1map_m * ymap_m
    # y1map = x1map.fullMap*ymap_m
    # calculate XTX
    x11map = x1map * x1map

    # filter maps
    # dmap.set_apix_tempy()
    fltr_data_r, scale = prepare_filter(
        rad, fltr, g_reci, g_real, ch, origin, apix, dmap.shape
    )
    y1map = mapfilter(y1map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    x11map = mapfilter(x11map, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci)
    # mf = RadialFilter(rad, fltr, g_reci, g_real, ch,
    #                  dmap.origin, dmap.apix, dmap.fullMap.shape)
    # y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    # x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)

    # calculate U shifts
    x1map = y1map / x11map

    return x1map


@njit
def solve(m, v):
    # l = len(m)  # no of points
    # check if matrix is square
    l, r, c = m.shape
    vl, vr = v.shape

    if r != c:
        # print('Matrix not square')
        raise TypeError("Matrix not square")
        return -1
    if r != vr:
        raise TypeError("Matrix/Vector mismatch")
        return -1

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
