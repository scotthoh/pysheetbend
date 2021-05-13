# Python implementation of shift field refinement
# for coordinates and U-Iso values
# S.W.Hoh, University of York, 2020

'''from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.maps.em_map import Map
from TEMPy.math.vector import Vector
'''
from TEMPy.EMMap import Map

import numpy as np
from timeit import default_timer as timer
import shiftfield_util as sf_util


def shift_field_coord(cmap, dmap, mask, x1map, x2map, x3map, rad, fltr,
                      fft_obj, ifft_obj, verbose=0):
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
    """
    g_reci_ = (cmap.z_size(), cmap.y_size(), cmap.x_size()//2+1)
    g_real_ = (g_reci_[0], g_reci_[1], int(g_reci_[2]-1)*2)
    data_r = cmap.fullMap
    if verbose >= 1:
        start = timer()
    xdata_c = fft_obj(data_r)
    # xdata_c[:, :, :] = (xdata_c[:, :, :]).conjugate()
    xdata_c = xdata_c.conjugate().copy()
    if verbose >= 1:
        end = timer()
        print('FFT Calc Map : {0} s'.format(end-start))
    ydata_c = np.zeros(xdata_c.shape, dtype='complex128')
    zdata_c = np.zeros(xdata_c.shape, dtype='complex128')
    ch = (g_real_[0]//2, g_real_[1]//2, g_real_[2]//2+1)
    i = complex(0.0, 1.0)
    for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = sf_util.hkl_c((cz, cy, cx), ch, g_real_)
                cdata = i * xdata_c[cz, cy, cx]
                zdata_c[cz, cy, cx] = float((2*np.pi)*hkl.z) * cdata
                ydata_c[cz, cy, cx] = float((2*np.pi)*hkl.y) * cdata
                xdata_c[cz, cy, cx] = float((2*np.pi)*hkl.x) * cdata

    # calculate gradient maps
    # fft complex to real
    if verbose >= 1:
        start = timer()
    zdata_c = zdata_c.conjugate().copy()
    ydata_c = ydata_c.conjugate().copy()
    xdata_c = xdata_c.conjugate().copy()
    if verbose >= 1:
        end = timer()
        print('3x conjugate : {0} s'.format(end-start))
    zdata_r = np.zeros(data_r.shape, dtype='float64')
    ydata_r = np.zeros(data_r.shape, dtype='float64')
    xdata_r = np.zeros(data_r.shape, dtype='float64')
    if verbose >= 1:
        start = timer()
    zdata_r = ifft_obj(zdata_c, zdata_r)
    if verbose >= 1:
        end = timer()
        print('first ifft ', end-start)
    ydata_r = ifft_obj(ydata_c, ydata_r)
    xdata_r = ifft_obj(xdata_c, xdata_r)    
    '''x1map = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x2map = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x3map = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    print(x1map.apix, x2map.apix, x3map.apix)
    print(x1map.origin, x2map.origin, x3map.origin)
    x1map.update_header()
    x2map.update_header()
    x3map.update_header()'''
    
    x1map.fullMap = xdata_r.copy()
    x2map.fullMap = ydata_r.copy()
    x3map.fullMap = zdata_r.copy()
    x1map.update_header()
    x2map.update_header()
    x3map.update_header()
    # end map preparation
    '''
    x1map.write_to_MRC_file('x1map_1.map')
    x2map.write_to_MRC_file('x2map_1.map')
    x3map.write_to_MRC_file('x3map_1.map')
    '''
    # calculate XTY and apply mask
    # ymap=diffmap , mmap = mask
    ymap = np.zeros(dmap.fullMap.shape)
    mmap = np.zeros(mask.fullMap.shape)
    ymap = dmap.fullMap.copy()
    mmap = mask.fullMap.copy()

    y1map = x1map.fullMap*ymap*mmap
    y2map = x2map.fullMap*ymap*mmap
    y3map = x3map.fullMap*ymap*mmap
    # print(np.count_nonzero(y1map==0.0), np.count_nonzero(y2map==0.0), np.count_nonzero(y3map==0.0))
    # calculate XTX  (removed multiply with mask)
    x11map = x1map.fullMap*x1map.fullMap
    # print(np.array_equal(x11map, x1map.fullMap))
    x12map = x1map.fullMap*x2map.fullMap
    x13map = x1map.fullMap*x3map.fullMap
    x22map = x2map.fullMap*x2map.fullMap
    x23map = x2map.fullMap*x3map.fullMap
    x33map = x3map.fullMap*x3map.fullMap
    # filter
    x33map1 = Map(np.zeros(cmap.fullMap.shape),
                  cmap.origin,
                  cmap.apix[0],
                  'mapname',)
    x33map1.fullMap = x33map
    x33map1.update_header()
    x33map1.write_to_MRC_file('x33_map1.map')

    dmap.set_apix_tempy()
    mf = sf_util.radial_fltr(rad, fltr, dmap)
    # dmap_gt = SB.maptree(dmap)
    y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    y2map = mf.mapfilter(y2map, fft_obj, ifft_obj)
    y3map = mf.mapfilter(y3map, fft_obj, ifft_obj)
    x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)
    x12map = mf.mapfilter(x12map, fft_obj, ifft_obj)
    x13map = mf.mapfilter(x13map, fft_obj, ifft_obj)
    x22map = mf.mapfilter(x22map, fft_obj, ifft_obj)
    x23map = mf.mapfilter(x23map, fft_obj, ifft_obj)
    x33map = mf.mapfilter(x33map, fft_obj, ifft_obj)
    x33map1.fullMap = x33map
    x33map1.update_header()
    x33map1.write_to_MRC_file('x33_map_filtered.map')
    # calculate U shifts
    count = 0
    start = timer()
    m = np.zeros([cmap.fullMap.size,3,3])
    v = np.zeros([cmap.fullMap.size, 3])
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
    end = timer()
    print('Set matrix : {0} s'.format(end-start))
    start = timer()
    v[:] = np.linalg.solve(m[:], v[:])
    end = timer()
    print('Solve linalg : {0} s'.format(end-start))
    start = timer()
    x1map_f = v[:, 0]
    x2map_f = v[:, 1]
    x3map_f = v[:, 2]
    # reassign values back to fullMaps
    x1map.fullMap = x1map_f.reshape(cmap.box_size())
    x2map.fullMap = x2map_f.reshape(cmap.box_size())
    x3map.fullMap = x3map_f.reshape(cmap.box_size())
    '''for index in range(0, cmap.fullMap.size):
        pos = mf.gt[1][index]  # gt from maptree_zyx
        p_zyx = (pos[0], pos[1], pos[2])
        x1map.fullMap[p_zyx] = v[index][0]  # .x  # [0]
        x2map.fullMap[p_zyx] = v[index][1]  # .y  # [1]
        x3map.fullMap[p_zyx] = v[index][2]  # .z  # [2]
    '''
    end = timer()
    print('Copy answer : {0} s'.format(end-start))
    '''
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
    '''
    
    
    # 2 - z and x axis swap in np array becomes x,y,z 
    # 3 - back using z,y,x convention by tempy map
    # 4 - fixed hkl cv - chv to cv + chv ... no change since return modulus
    # 5 - tanh_lowpass filter
    # 6 - init x1,x2,x3maps with lowpass map apix
    # make maps
    print('linalg err num: {0}'.format(count))
    return x1map, x2map, x3map


def shift_field_uiso(cmap, dmap, mask, x1map, rad, fltr, fft_obj,
                     ifft_obj, g_cell, verbose=0):
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
    g_reci_ = (cmap.z_size(), cmap.y_size(), cmap.x_size()//2+1)
    g_real_ = (g_reci_[0], g_reci_[1], int(g_reci_[2]-1)*2)
    # fullMap is numpy array
    data_r = cmap.fullMap
    if verbose >= 1:
        start = timer()
    data_c = fft_obj(data_r)
    data_c = data_c.conjugate().copy()
    if verbose >= 1:
        end = timer()
        print('first ft ', end-start)
    ch = (g_real_[0]//2, g_real_[1]//2, g_real_[2]//2+1)
    for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = sf_util.hkl_c((cz, cy, cx), ch, g_real_)
                scl = (2*np.pi*np.pi)*g_cell.metric_reci_lengthsq(hkl.x, hkl.y, hkl.z)
                data_c[cz, cy, cx] = scl * data_c[cz, cy, cx]
    
    # calculate gradient maps
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c)
    x1map.fullMap = data_r.copy()

    # make xmap
    ymap = np.zeros(dmap.fullMap.shape)
    mmap = np.zeros(mask.fullMap.shape)
    ymap = dmap.fullMap.copy()
    mmap = mask.fullMap.copy()
    # calculate XTY apply mask
    y1map = x1map.fullMap*ymap*mmap
    # calculate XTX
    x11map = x1map.fullMap*x1map.fullMap

    # filter maps
    dmap.set_apix_tempy()
    mf = sf_util.radial_fltr(rad, fltr, dmap)
    y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)

    # calculate U shifts
    x1map.fullMap[:, :, :] = y1map[:, :, :] / x11map[:, :, :]

    return x1map
