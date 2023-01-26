# Python implementation of shift field refinement
# for coordinates and U-Iso values
# S.W.Hoh, University of York, 2020

import numpy as np
from timeit import default_timer as timer
from numba import njit
from math import fabs
from pysheetbend.shiftfield_util import plan_fft_ifft
from pysheetbend.utils import map_funcs

# from memory_profiler import profile


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
    rad = effective_radius(function, radcyc)
    fltr_data_r = np.zeros(g_shape, dtype=np.float32)
    # x,y,z convention
    nx, ny, nz = np.indices(g_shape)
    indi = np.vstack([nx.ravel(), ny.ravel(), nz.ravel()]).T
    c = indi + gridinfo.grid_half  # self.gridshape.g_half
    c = np.fmod(c, g_shape)
    c_bool = c < 0
    c[c_bool[:, 0], 0] += g_shape[0]
    c[c_bool[:, 1], 1] += g_shape[1]
    c[c_bool[:, 2], 2] += g_shape[2]
    c -= gridinfo.grid_half
    # at the start the origin are corrected to 0 so no need offset with origin
    pos = c[:] * gridinfo.voxel_size
    r = np.sqrt(np.sum(np.square(pos), axis=1)).reshape(g_shape)
    r_bool = np.logical_and((r < rad), (r < radcyc))
    if function == 2:
        rf = pow(1.0 - r[r_bool] / radcyc, 2)
    elif function == 1:
        rf = 1.0 - r[r_bool] / radcyc
    elif function == 0:
        rf = 1.0
    fltr_data_r[r_bool] = rf[:]
    f000 = np.sum(rf)
    # calc scale factor
    # scale = 1.0 / f000
    if verbose >= 2:
        print(" f000, ", f000)
    del c, c_bool, r, r_bool, indi
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


# @profile
@njit()
def gradient_map_calc(xdata_c, g_reci, g_real, ch):
    # ydata_c = np.zeros(xdata_c.shape, dtype=np.complex128)
    # zdata_c = np.zeros(xdata_c.shape, dtype=np.complex128)
    ydata_c = np.zeros(xdata_c.shape, dtype=np.complex64)
    zdata_c = np.zeros(xdata_c.shape, dtype=np.complex64)

    i = complex(0.0, 1.0)
    for cx in range(g_reci[0]):  # x
        for cy in range(g_reci[1]):  # y
            for cz in range(g_reci[2]):  # z
                hkl = hkl_c((cx, cy, cz), ch, g_real)  # returned index x,y,z
                cdata = i * xdata_c[cx, cy, cz]
                xdata_c[cx, cy, cz] = float((2 * np.pi) * hkl[0]) * cdata
                ydata_c[cx, cy, cz] = float((2 * np.pi) * hkl[1]) * cdata
                zdata_c[cx, cy, cz] = float((2 * np.pi) * hkl[2]) * cdata
    return xdata_c, ydata_c, zdata_c


@njit()
def gradient_map_iso_calc(data_c, g_reci, g_real, ch, cell):
    for cx in range(0, g_reci[0]):  # x
        for cy in range(0, g_reci[1]):  # y
            for cz in range(0, g_reci[2]):  # z
                hkl = hkl_c((cx, cy, cz), ch, g_real)
                scl = (2 * np.pi * np.pi) * metric_reci_lengthsq(
                    hkl[0], hkl[1], hkl[2], cell
                )
                data_c[cx, cy, cz] = scl * data_c[cx, cy, cz]
    return data_c


@njit()
def hkl_c(c, ch, g):
    """
    Returns the index (x,y,z)
    Arguments
    *c*
      Index h,k,l
    *ch*
      Half of the grid shape
    *g*
      Real space grid shape
    """
    # x,y,z convention
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


# def writeMap(data_array, origin, apix, shape, fname):
#    mapobj = Map(
#        np.zeros((shape[2], shape[1], shape[0])),
#        origin,
#        apix,
#        "mapname",
#    )
#    mapobj.fullMap = data_array.copy()
#    mapobj.update_header()
#    mapobj.write_to_MRC_file(fname)


# @profile
def shift_field_coord(
    cmap,
    dmap,
    mmap,
    rad,
    fltr,
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
    # print(data_r.dtype)
    # print(cmap.dtype)
    # print(xdata_c.dtype)
    # calculate gradient map coefficients
    xdata_c, ydata_c, zdata_c = gradient_map_calc(xdata_c, g_reci, g_shape, g_half)

    # calculate gradient maps
    # fft complex to real
    zdata_c = zdata_c.conjugate().copy()
    # print("zdata_c {0}".format(zdata_c.shape))
    ydata_c = ydata_c.conjugate().copy()
    xdata_c = xdata_c.conjugate().copy()
    # print("data_r shape {0}".format(data_r.shape))
    # zdata_r = np.zeros(data_r.shape, dtype=np.float64)
    # ydata_r = np.zeros(data_r.shape, dtype=np.float64)
    # xdata_r = np.zeros(data_r.shape, dtype=np.float64)
    zdata_r = np.zeros(g_shape, dtype=np.float32)
    ydata_r = np.zeros(g_shape, dtype=np.float32)
    xdata_r = np.zeros(g_shape, dtype=np.float32)
    # print("zdata_r {0}".format(zdata_r.shape))
    # print("ifftw")
    # print("output shape")
    # print(ifft_obj.output_shape)
    # print(ifft_obj.output_dtype)
    # print(zdata_r.dtype)
    timelog.start("IFFTCalc")
    zdata_r = ifft_obj(zdata_c, zdata_r)
    ydata_r = ifft_obj(ydata_c, ydata_r)
    xdata_r = ifft_obj(xdata_c, xdata_r)
    timelog.end("IFFTCalc")
    # if verbose >= 6:
    #    writeMap(xdata_r, origin, apix, cmap.shape, f"gradx1map_{cyc}.map")
    #    writeMap(ydata_r, origin, apix, cmap.shape, f"gradx2map_{cyc}.map")
    #    writeMap(zdata_r, origin, apix, cmap.shape, f"gradx3map_{cyc}.map")

    # print(x1map.apix, x2map.apix, x3map.apix)
    # print(x1map.origin, x2map.origin, x3map.origin)
    # copy map
    # x1map_r = xdata_r.copy()
    # x2map_r = ydata_r.copy()
    # x3map_r = zdata_r.copy()
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
    # fltr_data_r, scale = prepare_filter(rad, fltr, gridinfo, verbose)
    fltr_data_r, scale = map_funcs.make_filter_edge_centered(
        grid_info=gridinfo, filter_radius=rad, function="quadratic", verbose=verbose
    )
    # if verbose >= 3:
    #    print(f"ymap shape {ymap.shape}")
    #    filt_ymap = mapfilter(
    #        ymap, fltr_data_r, scale, fft_obj, ifft_obj, g_real, g_reci
    #    )
    #    writeMap(filt_ymap, origin, apix, cmap.shape, f"filt_ymap{cyc}.map")
    # dmap_gt = SB.maptree(dmap)
    y1map = mapfilter(
        y1map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    y2map = mapfilter(
        y2map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    y3map = mapfilter(
        y3map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x11map = mapfilter(
        x11map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x12map = mapfilter(
        x12map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x13map = mapfilter(
        x13map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x22map = mapfilter(
        x22map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x23map = mapfilter(
        x23map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x33map = mapfilter(
        x33map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
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

    # x33map1.fullMap = x33map
    # x33map1.update_header()
    # x33map1.write_to_MRC_file('x33_map_filtered.map')
    # calculate U shifts
    mmap_f = np.ravel(mmap)

    mat_size = np.count_nonzero(~mmap_f)
    # print(cmap.size, np.count_nonzero(mmap), np.count_nonzero(mmap_f), mat_size)
    timelog.start("SetMatrix")
    # these use a lot of memory. see how to reduce
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
    # print(len(x11_f[x11_f == 0]))
    # print(len(x12_f[x12_f == 0]))
    # print(len(x13_f[x13_f == 0]))
    # print(len(x22_f[x22_f == 0]))
    # print(len(x23_f[x23_f == 0]))
    # print(len(x33_f[x33_f == 0]))
    # print(len(y1map_f[y1map_f == 0]))
    # print(len(y2map_f[y2map_f == 0]))
    # print(len(y3map_f[y3map_f == 0]))
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
    # print("check determinant")
    # det_m = np.linalg.det(m)
    # count_m0 = 0
    # for im in det_m:
    #  if im == 0:
    #    count_m0 += 1
    # print(count_m0)
    timelog.end("SetMatrix")
    timelog.start("LINALG")
    try:
        v[:] = np.linalg.solve(m[:], v[:])
        # v[:] = linalg.solve(m[:], v[:])
        # v = linalg.solve(m, v)
    except np.linalg.LinAlgError:
        for l in range(len(m)):
            v[l] = np.linalg.lstsq(m[l], v[l], rcond=None)[0]
        # v[:] = np.linalg.lstsq(m[:], v[:], rcond=None)
        # v = solve(m, v)
    # v = solve(m, v)
    # v[:], residuals, rank, s = np.linalg.lstsq(m[:], v[:])

    # v = solvelinalg(m[:], v[:])
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
    fltr,
    gridinfo,
    unitcell,
    fft_obj=None,
    ifft_obj=None,
    verbose=0,
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
    g_reci = gridinfo.grid_reci
    g_shape = gridinfo.grid_shape
    g_half = gridinfo.grid_half
    if fft_obj is None:
        fft_obj, ifft_obj = plan_fft_ifft(gridinfo=gridinfo)
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
    data_c = gradient_map_iso_calc(data_c, g_reci, g_shape, g_half, unitcell)
    """for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = sf_util.hkl_c((cz, cy, cx), ch, g_real_)
                scl = (2*np.pi*np.pi)*g_cell.metric_reci_lengthsq(hkl.x, hkl.y, hkl.z)
                data_c[cz, cy, cx] = scl * data_c[cz, cy, cx]"""

    # calculate gradient maps
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c)
    # x1map = data_r.copy()

    # make xmap
    # ymap = np.zeros(dmap.fullMap.shape)
    # mmap = np.zeros(mask.fullMap.shape)
    # ymap = dmap.fullMap.copy()

    # ymap_m = ma.masked_array(dmap, mask=mmap).data
    # x1map_m = ma.masked_array(x1map, mask=mmap).data
    # calculate XTY apply mask
    # y1map = x1map_m * ymap_m
    y1map = data_r * dmap
    # y1map = x1map.fullMap*ymap_m
    # calculate XTX
    x11map = data_r * data_r

    # filter maps
    # dmap.set_apix_tempy()
    fltr_data_r, scale = prepare_filter(rad, fltr, gridinfo, verbose)

    y1map = mapfilter(
        y1map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    x11map = mapfilter(
        x11map, fltr_data_r, 1.0 / scale, fft_obj, ifft_obj, g_shape, g_reci
    )
    # mf = RadialFilter(rad, fltr, g_reci, g_real, ch,
    #                  dmap.origin, dmap.apix, dmap.fullMap.shape)
    # y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    # x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)

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
