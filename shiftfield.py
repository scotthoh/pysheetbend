# Python implementation of shift field refinement
# in Flex-EM
# S.W.Hoh, University of York, 2020

# import esf_map_calc
import math
# mapin = mp.readMRC('/home/swh514/Projects/data/EMD-3488/map/emd_3488.map')
'''from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.protein.structure_parser import PDBParser
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.maps.em_map import Map
from TEMPy.math.vector import Vector
from TEMPy.map_process.map_filters import Filter
'''
from TEMPy.StructureBlurrer import StructureBlurrer
from TEMPy.StructureParser import PDBParser
from TEMPy.MapParser import MapParser as mp
from TEMPy.EMMap import Map
from TEMPy.Vector import Vector
from TEMPy.mapprocess.mapfilters import Filter

import numpy as np
from timeit import default_timer as timer
import esf_map_calc as emc
import shiftfield_util as sf_util


try:
    import pyfftw
    pyfftw_flag = True
except ImportError:
    pyfftw_flag = False

# shift field refinement coord
def shift_field_coord(cmap, dmap, mask, x1map, x2map, x3map, rad, fltr, fft_obj, ifft_obj):
    """
    Performs shift field refinement on coordinates
    Arguments:
    *cmap*
      Calculated map from input structure using ESF
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
    g_real_ = (g_reci_[0], g_reci_[1], int(g_reci_[2]-1)*2)  #int((g_reci_[2]-1)*2))
    #g_half_ = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2+1))

    data_r = cmap.fullMap
    boxvol = map_box_vol(cmap)
    start = timer()
    xdata_c = fft_obj(data_r)
    xdata_c[:,:,:] = (xdata_c[:,:,:]).conjugate()
    end = timer()
    ydata_c = np.zeros(xdata_c.shape, dtype='complex128')
    zdata_c = np.zeros(xdata_c.shape, dtype='complex128')
    SB = StructureBlurrer()
    ch = (g_real_[0]//2, g_real_[1]//2, g_real_[2]//2+1)
    i = complex(0.0, 1.0)
    for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = hkl_c((cz, cy, cx), ch, g_real_)
                cdata = i * xdata_c[cz, cy, cx]
                zdata_c[cz, cy, cx] = float((2*np.pi)*hkl.z) * cdata
                ydata_c[cz, cy, cx] = float((2*np.pi)*hkl.y) * cdata
                xdata_c[cz, cy, cx] = float((2*np.pi)*hkl.x) * cdata

    # calculate gradient maps
    # fft complex to real
    start = timer()
    zdata_c[:,:,:] = (zdata_c[:,:,:]).conjugate()
    ydata_c[:,:,:] = (ydata_c[:,:,:]).conjugate()
    xdata_c[:,:,:] = (xdata_c[:,:,:]).conjugate()
    end = timer()
    print('3x conj ', end-start)
    zdata_r = np.zeros(data_r.shape, dtype='float64')
    ydata_r = np.zeros(data_r.shape, dtype='float64')
    xdata_r = np.zeros(data_r.shape, dtype='float64')
    start = timer()
    zdata_r = ifft_obj(zdata_c, zdata_r)
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
    
    x1map.fullMap[:,:,:] = xdata_r[:,:,:]
    x2map.fullMap[:,:,:] = ydata_r[:,:,:]
    x3map.fullMap[:,:,:] = zdata_r[:,:,:]
    x1map.update_header()
    x2map.update_header()
    x3map.update_header()
    # end map preparation
    '''
    x1map.write_to_MRC_file('x1map_1.map')
    x2map.write_to_MRC_file('x2map_1.map')
    x3map.write_to_MRC_file('x3map_1.map')
    '''
    #exit()
    # calculate XTY and apply mask
    # ymap=diffmap , mmap = mask
    ymap = np.zeros(dmap.fullMap.shape)
    mmap = np.zeros(mask.fullMap.shape)
    ymap[:,:,:] = dmap.fullMap[:,:,:]
    mmap[:,:,:] = mask.fullMap[:,:,:]
    #print(x1map.box_size(), ymap.box_size(), mmap.box_size())
    
    y1map = x1map.fullMap*ymap*mmap
    y2map = x2map.fullMap*ymap*mmap
    y3map = x3map.fullMap*ymap*mmap
    #print(np.count_nonzero(y1map==0.0), np.count_nonzero(y2map==0.0), np.count_nonzero(y3map==0.0))
    # calculate XTX and apply mask
    x11map = x1map.fullMap*x1map.fullMap*mmap
    #print(np.array_equal(x11map, x1map.fullMap))
    x12map = x1map.fullMap*x2map.fullMap*mmap
    x13map = x1map.fullMap*x3map.fullMap*mmap
    x22map = x2map.fullMap*x2map.fullMap*mmap
    x23map = x2map.fullMap*x3map.fullMap*mmap
    x33map = x3map.fullMap*x3map.fullMap*mmap
    '''
    print(np.count_nonzero(x11map==0.0), np.count_nonzero(x12map==0.0), np.count_nonzero(x13map==0.0))
    print(np.count_nonzero(x22map==0.0), np.count_nonzero(x23map==0.0), np.count_nonzero(x33map==0.0))
    
    print(np.array_equal(x12map, x11map))
    print(np.array_equal(x22map, x2map.fullMap))
    print(np.array_equal(x33map, x3map.fullMap))
    
    # filter
    #something is wrong with the scipy convolution and filter
    # using clipper filter, still something wrong
    
    print(y1map.shape)
    print(y2map.shape)
    print(x11map.shape)
    print(x33map.shape)
    '''
    '''
    y1map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    y1map1.fullMap = y1map
    y1map1.update_header()
    y1map1.write_to_MRC_fiele('y1_map5.map')

    y2map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    y2map1.fullMap = y2map
    y2map1.update_header()
    y2map1.write_to_MRC_file('y2_map5.map')
    
    y3map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    y3map1.fullMap = y3map
    y3map1.update_header()
    y3map1.write_to_MRC_file('y3_map5.map')
    
    x11map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x11map1.fullMap = x11map
    x11map1.update_header()
    x11map1.write_to_MRC_file('x11_map5.map')
    '''
    x33map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix[0],
                'mapname',)
    x33map1.fullMap = x33map
    x33map1.update_header()
    x33map1.write_to_MRC_file('x33_map1.map')
    '''
    x12map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x12map1.fullMap = x12map
    x12map1.update_header()
    x12map1.write_to_MRC_file('x12_map5.map')
    x13map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x13map1.fullMap = x13map
    x13map1.update_header()
    x13map1.write_to_MRC_file('x13_map5.map')
    x22map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x22map1.fullMap = x22map
    x22map1.update_header()
    x22map1.write_to_MRC_file('x22_map5.map')
    x23map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x23map1.fullMap = x23map
    x23map1.update_header()
    x23map1.write_to_MRC_file('x23_map5.map')
    '''
    dmap.set_apix_tempy()
    mf = sf_util.radial_fltr(rad, fltr, dmap)
    dmap_gt = SB.maptree(dmap)
    
    y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    y2map = mf.mapfilter(y2map, fft_obj, ifft_obj)
    y3map = mf.mapfilter(y3map, fft_obj, ifft_obj)
    x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)
    x12map = mf.mapfilter(x12map, fft_obj, ifft_obj)
    x13map = mf.mapfilter(x13map, fft_obj, ifft_obj)
    x22map = mf.mapfilter(x22map, fft_obj, ifft_obj)
    x23map = mf.mapfilter(x23map, fft_obj, ifft_obj)
    x33map = mf.mapfilter(x33map, fft_obj, ifft_obj)
    '''
    y1map1.fullMap = y1map
    y1map1.update_header()
    y1map1.write_to_MRC_file('y1_map_filtered5.map')
    x11map1.fullMap = x11map
    x11map1.update_header()
    x11map1.write_to_MRC_file('x11_map_filtered5.map')
    x12map1.fullMap = x12map
    x12map1.update_header()
    x12map1.write_to_MRC_file('x12_map_filtered5.map')
    x13map1.fullMap = x13map
    x13map1.update_header()
    x13map1.write_to_MRC_file('x13_map_filtered5.map')
    '''
    x33map1.fullMap = x33map
    x33map1.update_header()
    x33map1.write_to_MRC_file('x33_map_filtered.map')
    
    '''
    y1map = signal.convolve(y1map, fltr, mode='same', method='fft') / np.sum(fltr)
    y2map = signal.convolve(y2map, fltr, mode='same', method='fft') / np.sum(fltr)
    y3map = signal.convolve(y2map, fltr, mode='same', method='fft') / np.sum(fltr)
    x11map = signal.convolve(x11map, fltr, mode='same', method='fft') / np.sum(fltr)
    x12map = signal.convolve(x12map, fltr, mode='same', method='fft') / np.sum(fltr)
    x13map = signal.convolve(x13map, fltr, mode='same', method='fft') / np.sum(fltr)
    x22map = signal.convolve(x22map, fltr, mode='same', method='fft') / np.sum(fltr)
    x23map = signal.convolve(x23map, fltr, mode='same', method='fft') / np.sum(fltr)
    x33map = signal.convolve(x33map, fltr, mode='same', method='fft') / np.sum(fltr)
    '''
    #print(np.count_nonzero(y1map==0.0), np.count_nonzero(y2map==0.0), np.count_nonzero(y3map==0.0))
    #print(np.count_nonzero(x11map==0.0), np.count_nonzero(x12map==0.0), np.count_nonzero(x13map==0.0))
    #print(np.count_nonzero(x22map==0.0), np.count_nonzero(x23map==0.0), np.count_nonzero(x33map==0.0))
    #exit()
    
    #print(ymap_gt[0])
    # calculate U shifts:q
    count = 0
    start = timer()
    for index in range(0, len(dmap_gt[1])):
        # vector v
        pos = dmap_gt[1][index]
        #p_z = pos[2]
        #p_y = pos[1]
        #p_x = pos[0]
        p_zyx = (pos[2], pos[1], pos[0])
        v = np.array([y1map[p_zyx], y2map[p_zyx], y3map[p_zyx]])  # Vector(x, y, z) 
        # v[0] = y1map[index]
        # v[1] = y2map[index]
        # v[2] = y3map[index]
        #print(p_zyx, v)
        # matrix(3,3)
        m = np.zeros([3, 3])
        #np.zeros()
        m[0, 0] = x11map[p_zyx]
        m[0, 1] = x12map[p_zyx]
        m[1, 0] = x12map[p_zyx]
        m[0, 2] = x13map[p_zyx]
        m[2, 0] = x13map[p_zyx]
        m[1, 1] = x22map[p_zyx]
        m[1, 2] = x23map[p_zyx]
        m[2, 1] = x23map[p_zyx]
        m[2, 2] = x33map[p_zyx]

        # solve matrix v c++ m.solve(v)
        # solve linear eq Ax=b for x
        # can use numpy.linalg.solve
        
        try:
            v = np.linalg.solve(m, v)
            #if count < 10:
                #print(v[0], v[1], v[2])
                #count += 1
        except np.linalg.LinAlgError:
            count+=1
            pass #print(v)    

        x1map.fullMap[p_zyx] = v[0] #.x  # [0]
        x2map.fullMap[p_zyx] = v[1] #.y  # [1]
        x3map.fullMap[p_zyx] = v[2] #.z  # [2]
    end = timer()
    print('linalg: ', end-start)
    # 2 - z and x axis swap in np array becomes x,y,z 
    # 3 - back using z,y,x convention by tempy map
    # 4 - fixed hkl cv - chv to cv + chv ... no change since return modulus
    # 5 - tanh_lowpass filter
    # 6 - init x1,x2,x3maps with lowpass map apix
    # make maps
    print('linalg err num: {0}'.format(count))
    return x1map, x2map, x3map

    # do a reciprocal grid and then hkl indices for gradient map coefficient calculation
    # need to do complex part and gradient calculation
    # main(mapin, filename)

def shift_field_coord_c(cmap, dmap, mask, x1map, x2map, x3map, rad, fltr, fft_obj, ifft_obj):
    """
    Performs shift field refinement on coordinates with const
    Arguments:
    *cmap*
      Calculated map from input structure using ESF
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
    #g = cmap.fullMap    # swapshape
    # u=x, v=y, w=z,

    g_reci_ = (cmap.z_size(), cmap.y_size(), cmap.x_size()//2+1)
    g_real_ = (g_reci_[0], g_reci_[1], int(g_reci_[2]-1)*2)  #int((g_reci_[2]-1)*2))
    #g_half_ = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2))

    # g_reci = (cmap.x_size(), cmap.y_size(), int(cmap.z_size()/2+1))
    # g_real = (g_reci[0], g_reci[1], int(g_reci[2]*2))
    # g_half = (int(cmap.x_size()/2), int(cmap.y_size()/2), int(cmap.z_size()/2))

    # fullMap is numpy array
    #fltrmap = Filter(cmap)
    #ftfilter = fltrmap.tanh_lowpass(0.5)
    #lowpass_map = fltrmap.fourier_filter(ftfilter)
    #print(lowpass_map.fullMap.shape, lowpass_map.apix)
    data_r = cmap.fullMap   # .swapaxes(0, 2)
    print(data_r.dtype)
    #data_c = data_r.astype('complex128')
    boxvol = map_box_vol(cmap)

    start = timer()
    #xdata_c = sf_util.fourier_transform(data_r, boxvol, conj=True)
    xdata_c = fft_obj(data_r)
    xdata_c[:,:,:] = (xdata_c[:,:,:]).conjugate()
    end = timer()
    print('first ft ', end-start)
    ydata_c = np.zeros(xdata_c.shape, dtype='complex128')
    zdata_c = np.zeros(xdata_c.shape, dtype='complex128')
    SB = StructureBlurrer()
    #print(data_c.dtype)
    print(data_r.dtype)
    #c = (cmap.x_size(), 0)
    #ch = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2))
    ch = (g_real_[0]//2, g_real_[1]//2, g_real_[2]//2+1)
    i = complex(0.0, 1.0)
    for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = hkl_c((cz, cy, cx), ch, g_real_)
                cdata = i * xdata_c[cz, cy, cx]
                zdata_c[cz, cy, cx] = float((2*np.pi)*hkl.z) * cdata
                ydata_c[cz, cy, cx] = float((2*np.pi)*hkl.y) * cdata
                xdata_c[cz, cy, cx] = float((2*np.pi)*hkl.x) * cdata

    # calculate gradient maps
    # fft complex to real
    #zdata_r = sf_util.inv_fourier_transform(zdata_c, 1.0/boxvol, g_reci_, conj=True)
    #ydata_r = sf_util.inv_fourier_transform(ydata_c, 1.0/boxvol, g_reci_, conj=True)
    #xdata_r = sf_util.inv_fourier_transform(xdata_c, 1.0/boxvol, g_reci_, conj=True)
    
    start = timer()
    zdata_c[:,:,:] = (zdata_c[:,:,:]).conjugate()
    ydata_c[:,:,:] = (ydata_c[:,:,:]).conjugate()
    xdata_c[:,:,:] = (xdata_c[:,:,:]).conjugate()
    end = timer()
    print('3x conj ', end-start)
    zdata_r = np.zeros(data_r.shape, dtype='float64')
    ydata_r = np.zeros(data_r.shape, dtype='float64')
    xdata_r = np.zeros(data_r.shape, dtype='float64')
    start = timer()
    zdata_r = ifft_obj(zdata_c, zdata_r)
    end = timer()
    print('first ifft ', end-start)

    ydata_r = ifft_obj(ydata_c, ydata_r)
    #print(zdata_r is ydata_r)
    xdata_r = ifft_obj(xdata_c, xdata_r)
    #print(xdata_r is ydata_r)
        
    #print(zdata_r.shape)
    #print(ydata_r.shape)
    #print(xdata_r.shape)
    #print(cmap.fullMap.shape)
    # xdata_r = xdata_r.swapaxes(0, 2)
    # ydata_r = ydata_r.swapaxes(0, 2)
    # zdata_r = zdata_r.swapaxes(0, 2)    
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
    x1map.fullMap[:,:,:] = xdata_r[:,:,:]
    x2map.fullMap[:,:,:] = ydata_r[:,:,:]
    x3map.fullMap[:,:,:] = zdata_r[:,:,:]
    x1map.update_header()
    x2map.update_header()
    x3map.update_header()
    # end map preparation
    #x1map.write_to_MRC_file('x1map_1.map')
    #x2map.write_to_MRC_file('x2map_1.map')
    #x3map.write_to_MRC_file('x3map_1.map')
    #exit()
    # calculate XTY and apply mask
    # ymap=diffmap , mmap = mask
    ymap = np.zeros(dmap.fullMap.shape)
    mmap = np.zeros(mask.fullMap.shape)
    ymap[:,:,:] = dmap.fullMap[:,:,:]
    mmap[:,:,:] = mask.fullMap[:,:,:]
    #print(x1map.box_size(), ymap.box_size(), mmap.box_size())
    y0map = ymap*mmap
    y1map = x1map.fullMap*ymap*mmap
    y2map = x2map.fullMap*ymap*mmap
    y3map = x3map.fullMap*ymap*mmap
    # calculate XTX and apply mask
    x00map = mmap
    x01map = x1map.fullMap*mmap
    x02map = x2map.fullMap*mmap
    x03map = x3map.fullMap*mmap
    x11map = x1map.fullMap*x1map.fullMap*mmap
    x12map = x1map.fullMap*x2map.fullMap*mmap
    x13map = x1map.fullMap*x3map.fullMap*mmap
    x22map = x2map.fullMap*x2map.fullMap*mmap
    x23map = x2map.fullMap*x3map.fullMap*mmap
    x33map = x3map.fullMap*x3map.fullMap*mmap
    
    # filter
    #something is wrong with the scipy convolution and filter
    # using clipper filter, still something wrong
    
    '''
    y1map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    y1map1.fullMap = y1map
    y1map1.update_header()
    y1map1.write_to_MRC_fiele('y1_map5.map')
    
    y2map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    y2map1.fullMap = y2map
    y2map1.update_header()
    y2map1.write_to_MRC_file('y2_map5.map')
    
    y3map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    y3map1.fullMap = y3map
    y3map1.update_header()
    y3map1.write_to_MRC_file('y3_map5.map')
    
    x11map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x11map1.fullMap = x11map
    x11map1.update_header()
    x11map1.write_to_MRC_file('x11_map5.map')
    
    x33map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x33map1.fullMap = x33map
    x33map1.update_header()
    x33map1.write_to_MRC_file('x33_map1.map')
    
    x12map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x12map1.fullMap = x12map
    x12map1.update_header()
    x12map1.write_to_MRC_file('x12_map5.map')
    x13map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x13map1.fullMap = x13map
    x13map1.update_header()
    x13map1.write_to_MRC_file('x13_map5.map')
    x22map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x22map1.fullMap = x22map
    x22map1.update_header()
    x22map1.write_to_MRC_file('x22_map5.map')
    x23map1 = Map(np.zeros(cmap.fullMap.shape),
                cmap.origin,
                cmap.apix,
                'mapname',)
    x23map1.fullMap = x23map
    x23map1.update_header()
    x23map1.write_to_MRC_file('x23_map5.map')
    '''
    dmap.set_apix_tempy()
    mf = sf_util.radial_fltr(rad, fltr, dmap)
    dmap_gt = SB.maptree(dmap)
    y0map = mf.mapfilter(y0map, fft_obj, ifft_obj)
    y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    y2map = mf.mapfilter(y2map, fft_obj, ifft_obj)
    y3map = mf.mapfilter(y3map, fft_obj, ifft_obj)
    x00map = mf.mapfilter(x00map, fft_obj, ifft_obj)
    x01map = mf.mapfilter(x01map, fft_obj, ifft_obj)
    x02map = mf.mapfilter(x02map, fft_obj, ifft_obj)
    x03map = mf.mapfilter(x03map, fft_obj, ifft_obj)
    x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)
    x12map = mf.mapfilter(x12map, fft_obj, ifft_obj)
    x13map = mf.mapfilter(x13map, fft_obj, ifft_obj)
    x22map = mf.mapfilter(x22map, fft_obj, ifft_obj)
    x23map = mf.mapfilter(x23map, fft_obj, ifft_obj)
    x33map = mf.mapfilter(x33map, fft_obj, ifft_obj)
    '''
    y1map1.fullMap = y1map
    y1map1.update_header()
    y1map1.write_to_MRC_file('y1_map_filtered5.map')
    x11map1.fullMap = x11map
    x11map1.update_header()
    x11map1.write_to_MRC_file('x11_map_filtered5.map')
    x12map1.fullMap = x12map
    x12map1.update_header()
    x12map1.write_to_MRC_file('x12_map_filtered5.map')
    x13map1.fullMap = x13map
    x13map1.update_header()
    x13map1.write_to_MRC_file('x13_map_filtered5.map')
    '''
    #x33map1.fullMap = x33map
    #x33map1.update_header()
    #x33map1.write_to_MRC_file('x33_map_filtered5.map')
    
    '''
    y1map = signal.convolve(y1map, fltr, mode='same', method='fft') / np.sum(fltr)
    y2map = signal.convolve(y2map, fltr, mode='same', method='fft') / np.sum(fltr)
    y3map = signal.convolve(y2map, fltr, mode='same', method='fft') / np.sum(fltr)
    x11map = signal.convolve(x11map, fltr, mode='same', method='fft') / np.sum(fltr)
    x12map = signal.convolve(x12map, fltr, mode='same', method='fft') / np.sum(fltr)
    x13map = signal.convolve(x13map, fltr, mode='same', method='fft') / np.sum(fltr)
    x22map = signal.convolve(x22map, fltr, mode='same', method='fft') / np.sum(fltr)
    x23map = signal.convolve(x23map, fltr, mode='same', method='fft') / np.sum(fltr)
    x33map = signal.convolve(x33map, fltr, mode='same', method='fft') / np.sum(fltr)
    '''
    #exit()
    
    #print(ymap_gt[0])
    # calculate U shifts:q
    count = 0
    for index in range(0, len(dmap_gt[1])):
        # vector v
        pos = dmap_gt[1][index]
        #p_z = pos[2]
        #p_y = pos[1]
        #p_x = pos[0]
        p_zyx = (pos[2], pos[1], pos[0])
        v = np.array([y0map[p_zyx], y1map[p_zyx], y2map[p_zyx], y3map[p_zyx]])  # Vector(x, y, z) 
        # v[0] = y1map[index]
        # v[1] = y2map[index]
        # v[2] = y3map[index]
        #print(p_zyx, v)
        # matrix(3,3)
        m = np.zeros([4, 4])
        #np.zeros()
        m[0, 0] = x00map[p_zyx]
        m[0, 1] = m[1, 0] = x01map[p_zyx]
        m[0, 2] = m[2, 0] = x02map[p_zyx]
        m[0, 3] = m[3, 0] = x03map[p_zyx]
        m[1, 1] = x11map[p_zyx]
        m[1, 2] = m[2, 1] = x12map[p_zyx]
        m[1, 3] = m[3, 1] = x13map[p_zyx]
        m[2, 2] = x22map[p_zyx]
        m[3, 2] = m[2, 3] = x23map[p_zyx]
        m[3, 3] = x33map[p_zyx]

        # solve matrix v c++ m.solve(v)
        # solve linear eq Ax=b for x
        # can use numpy.linalg.solve
        
        try:
            v = np.linalg.solve(m, v)
            #if count < 10:
                #print(v[0], v[1], v[2])
                #count += 1
        except np.linalg.LinAlgError:
            count+=1
            pass #print(v)    

        x1map.fullMap[p_zyx] = v[1] #.x  # [0]
        x2map.fullMap[p_zyx] = v[2] #.y  # [1]
        x3map.fullMap[p_zyx] = v[3] #.z  # [2]
    # 2 - z and x axis swap in np array becomes x,y,z 
    # 3 - back using z,y,x convention by tempy map
    # 4 - fixed hkl cv - chv to cv + chv ... no change since return modulus
    # 5 - tanh_lowpass filter
    # 6 - init x1,x2,x3maps with lowpass map apix
    # make maps
    print('linalg err num: {0}'.format(count))
    return x1map, x2map, x3map


def shift_field_uiso(cmap, dmap, mask, x1map, rad, fltr, fft_obj, ifft_obj, g_cell):
    """
    Performs shift field refinement on coordinates with const
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
    #g = cmap.fullMap    # swapshape
    # u=x, v=y, w=z,

    g_reci_ = (cmap.z_size(), cmap.y_size(), cmap.x_size()//2+1)
    g_real_ = (g_reci_[0], g_reci_[1], int(g_reci_[2]-1)*2)  #int((g_reci_[2]-1)*2))
    #g_half_ = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2))

    # g_reci = (cmap.x_size(), cmap.y_size(), int(cmap.z_size()/2+1))
    # g_real = (g_reci[0], g_reci[1], int(g_reci[2]*2))
    # g_half = (int(cmap.x_size()/2), int(cmap.y_size()/2), int(cmap.z_size()/2))

    # fullMap is numpy array
    #fltrmap = Filter(cmap)
    #ftfilter = fltrmap.tanh_lowpass(0.5)
    #lowpass_map = fltrmap.fourier_filter(ftfilter)
    #print(lowpass_map.fullMap.shape, lowpass_map.apix)
    data_r = cmap.fullMap   # .swapaxes(0, 2)
    print(data_r.dtype)
    #data_c = data_r.astype('complex128')
    boxvol = map_box_vol(cmap)

    start = timer()
    #xdata_c = sf_util.fourier_transform(data_r, boxvol, conj=True)
    data_c = fft_obj(data_r)
    data_c[:,:,:] = (data_c[:,:,:]).conjugate()
    end = timer()
    print('first ft ', end-start)
    SB = StructureBlurrer()
    #print(data_c.dtype)
    print(data_r.dtype)
    #c = (cmap.x_size(), 0)
    #ch = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2))
    ch = (g_real_[0]//2, g_real_[1]//2, g_real_[2]//2+1)
    for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = hkl_c((cz, cy, cx), ch, g_real_)
                scl = (2*np.pi*np.pi)*g_cell.metric_reci_lengthsq(hkl.x, hkl.y, hkl.z)
                #cell.metric_reci().lengthsq( Coord_reci_frac( *this )
                #{metric_reci().lengthsq return ( v[0]*(v[0]*m00 + v[1]*m01 + v[2]*m02) +
                # v[1]*(v[1]*m11 + v[2]*m12) + v[2]*(v[2]*m22) ); }
                data_c[cz, cy, cx] = scl * data_c[cz, cy, cx]
    
    # calculate gradient maps
    data_c[:,:,:] = (data_c[:,:,:]).conjugate()
    data_r = ifft_obj(data_c)
    x1map.fullMap[:,:,:] = data_r[:,:,:]

    # make xmap
    ymap = np.zeros(dmap.fullMap.shape)
    mmap = np.zeros(mask.fullMap.shape)
    ymap[:,:,:] = dmap.fullMap[:,:,:]
    mmap[:,:,:] = mask.fullMap[:,:,:]
    #print(x1map.box_size(), ymap.box_size(), mmap.box_size())
    y1map = x1map.fullMap*ymap*mmap
    x11map = x1map.fullMap*x1map.fullMap*mmap

    # filter maps
    dmap.set_apix_tempy()
    mf = sf_util.radial_fltr(rad, fltr, dmap)
    #dmap_gt = SB.maptree(dmap)
    y1map = mf.mapfilter(y1map, fft_obj, ifft_obj)
    x11map = mf.mapfilter(x11map, fft_obj, ifft_obj)
    
    # calculate U shifts
    x1map.fullMap[:,:,:] = y1map[:,:,:] / x11map[:,:,:]

    return x1map



def mapGridPositions_radius(densMap, atom, gridtree, radius):
    """
    Returns the indices of the nearest pixels within the radius
    specified to an atom as a list.
    Adapted from TEMPy mapGridPositions
    Arguments:Map
    *densMap*
      Map instance the atom is to be placed on.
    *atom*
      Atom instance.
    *gridtree*
      KDTree of the map coordinates (absolute cartesian)
    *radius*
      radius from the atom coords for nearest pixels
    """
    origin = densMap.origin
    apix = densMap.apix
    x_pos = int(round((atom.x - origin[0]) / apix, 0))
    y_pos = int(round((atom.y - origin[1]) / apix, 0))
    z_pos = int(round((atom.z - origin[2]) / apix, 0))
    if((densMap.x_size() >= x_pos >= 0) and (densMap.y_size() >= y_pos >= 0)
       and (densMap.z_size() >= z_pos >= 0)):
        # search all points within radius of atom
        list_points = gridtree.query_ball_point([
          atom.x, atom.y, atom.z], radius)
        return list_points  # ,(x_pos, y_pos, z_pos)
    else:
        print('Warning, atom out of map box')
        return []


def make_atom_overlay_map1_rad(densMap, prot, gridtree, rad):
    """
    Returns a Map instance with atom locations recorded on
    the voxel within radius with a value of 1
    """
    densMap = densMap.copy()
    densMap.fullMap = densMap.fullMap * 0
    # use mapgridpositions to get the points within radius. faster and efficient
    # this works. resample map before assigning density
    for atm in prot:
        # get list of nearest points of an atom
        points = mapGridPositions_radius(newMap, atm, gridtree[0], rad)
        for ind in points:
            pos = gridtree[1][ind]  # real coordinates of the index
            # initialise AtomShapeFn object with atom
            # get 3D coordinates from map grid position
            # coord_pos = mapgridpos_to_coord(pos, newMap, False)
            # calculate electron density from 3D coordinates and
            # set to the map grid position
            # p_z=int(pos[2]-(newMap.apix/2.0))
            # p_y=int(pos[1]-(newMap.apix/2.0))
            # p_x=int(pos[0]-(newMap.apix/2.0))
            p_z = pos[2]
            p_y = pos[1]
            p_x = pos[0]

            densMap.fullMap[p_z, p_y, p_x] = 1.0
    return densMap


def mapgridpos_to_coord(pos, densMap, ccpem_tempy=False):
    """
    Returns the 3D coordinates of the map grid position given
    Argument:
    *pos*
      map grid position
    *densMap*
      Map instance the atom is placed on
    """

    origin = densMap.origin
    apix = densMap.apix
    midapix = apix/2.0

    if not ccpem_tempy:
        x_coord = pos[0] * apix + origin[0]
        y_coord = pos[1] * apix + origin[1]
        z_coord = pos[2] * apix + origin[2]
    else:
        x_coord = (pos[0] - midapix) * apix + origin[0]
        y_coord = (pos[1] - midapix) * apix + origin[1]
        z_coord = (pos[2] - midapix) * apix + origin[2]

    return (x_coord, y_coord, z_coord)

def fourier_transform(data_r, grid_sam, grid_reci, scale, gt):
    
    grid_size = data_r.size
    #output_shape = data_r.shape[:len(data_r.shape)-1] + \
                                #(data_r.shape[-1]//2 + 1,)
    output_shape = grid_sam
    
    #try:
    if not pyfftw_flag:
        print('no pyfftw')
        raise ImportError
    #start = timer()
    #print('1')
    input_arr = pyfftw.empty_aligned(data_r.shape,
                                        dtype='float64', n=16)
    #end = timer()
    #print('2')
    
    #print(end-start)
    #start = timer()
    #print('3')
    
    output_arr = pyfftw.empty_aligned(output_shape,
                                        dtype='complex128', n=16)
    end = timer()
    #print(end-start)
    #print('4')
    
    # fft planning,
    fft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_FORWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
    input_arr[:, :, :] = data_r[:, :, :]
    fft.update_arrays(input_arr, output_arr)
    #start = timer()
    fft()
    #end = timer()
    #print(end-start)
    #except:
    print("not running")
    #print('test1')
    # scale?
    s = float(scale) / (grid_sam[0]*grid_sam[1]*grid_sam[0])
    
    #n = (grid_reci[0]*grid_reci[1]*grid_reci[2])
    for ind in range(0, len(gt)):
        pos = gt[ind]
        output_arr[pos[2], pos[1], pos[0]] = (s * output_arr[pos[2], pos[1], pos[0]]).conjugate()
    '''
    for oz in range(0, output_shape[0]):
        for oy in range(0, output_shape[1]):
            for ox in range(0, output_shape[2]):
                output_arr[oz, oy, ox] = (s * output_arr[oz, oy, ox]).conjugate()
    '''
    #n = int(np.prod(output_shape))
    #for j in range(0, n):
    #    output_arr[j] = (s * output_arr[j]).conjugate()

    return output_arr


def inv_fourier_transform(data_c, scale, grid_reci, grid_sam):
    # grid_size = data_c.map_size()
    s = float(scale)
    #n = int(np.prod(grid_reci))


    for z in range(0, grid_reci[0]):
        for y in range(0, grid_reci[1]):
            for x in range(0, grid_reci[2]):
                data_c[z, y, x] = (s * data_c[z, y, x]).conjugate()
    
    output_shape = grid_sam #data_c.shape[:len(data_c.shape)-1] + \
                            #    ((data_c.shape[-1] - 1)*2,)

    try:
        if not pyfftw_flag:
            raise ImportError
        start = timer()
        input_arr = pyfftw.empty_aligned(data_c.shape,
                                         dtype='complex128', n=16)
        #end = timer()
        #print(end-start)
        #start = timer()
        output_arr = pyfftw.empty_aligned(output_shape,
                                          dtype='float64', n=16)
        #end = timer()
        #print(end-start)
        # fft planning,
        ifft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_BACKWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
        input_arr[:, :, :] = data_c[:, :, :]
        ifft.update_arrays(input_arr, output_arr)
        #start = timer()
        ifft()
        #end = timer()
        #print(end-start)
    except:
        print('not running')

    return output_arr


def cor_mod(a, b):
    c = math.fmod(a, b)
    if (c < 0):
        c += b
    return int(c)


def hkl_c(c, ch, g):
    # z, y, x; vector(x, y ,z)
    cv = Vector(int(c[2]), int(c[1]), int(c[0]))
    chv = Vector(int(ch[2]), int(ch[1]), int(ch[0]))
    # cv = Vector(int(c[0]), int(c[1]), int(c[2]))
    # chv = Vector(int(ch[0]), int(ch[1]), int(ch[2]))
    v1 = cv + chv
    m1 = Vector(cor_mod(v1.x, g[2]),
          cor_mod(v1.y, g[1]),
          cor_mod(v1.z, g[0]))

    return m1 - chv


def map_box_vol(densMap):
    return densMap.x_size() * densMap.y_size() * densMap.z_size()


if __name__ == '__main__':
    import math
    # mapin = mp.readMRC('/home/swh514/Projects/data/EMD-3488/map/emd_3488.map')
    from TEMPy.protein.structure_blurrer import StructureBlurrer
    from TEMPy.protein.structure_parser import PDBParser
    from TEMPy.maps.map_parser import MapParser as mp
    from TEMPy.maps.em_map import Map
    from TEMPy.math.vector import Vector
    from TEMPy.map_process.map_filters import Filter
    import numpy as np
    from timeit import default_timer as timer
    import esf_map_calc as emc

    try:
        import pyfftw
        pyfftw_flag = True
    except ImportError:
        pyfftw_flag = False

    pdbin = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1.ent'
    struct_inst = PDBParser.read_PDB_file('test', pdbin, hetatm=True, water=False)

    # atomList = struct_inst.atomList
    SB = StructureBlurrer()
    
    cmap = emc.main(mapin, pdbin)
    apix = mapin.apix
    print(apix)
    x_s = int(mapin.x_size() * apix)
    y_s = int(mapin.y_size() * apix)
    z_s = int(mapin.z_size() * apix)
    '''x_s = int(round(mapin.x_size() * mapin.apix))
    y_s = int(round(mapin.y_size() * mapin.apix))
    z_s = int(round(mapin.z_size() * mapin.apix))'''
    newMap = Map(np.zeros((z_s, y_s, x_s)),
                 mapin.origin,
                 apix,
                 'mapname',)
    newMap.apix = (apix * mapin.x_size()) / x_s
    # newMap.update_header()
    print('newmap pre, {0}, {1}, {2}'.format(newMap.box_size(),
                                             newMap.apix,
                                             str(newMap.fullMap.dtype)))
    
    newMap = newMap.downsample_map(mapin.apix, grid_shape=mapin.fullMap.shape)
    newMap.update_header()
    
    gridtree = SB.maptree(newMap)

    # newMap = SB.make_atom_overlay_map1(newMap, struct_inst)
    # mmap = mask
    mmap = make_atom_overlay_map1_rad(newMap, struct_inst, gridtree, 2.5)
    
    print('newmap, {0}, {1}, {2}'.format(mmap.box_size(),
                                         mmap.apix,
                                         str(mmap.fullMap.dtype)))
    print('cmap, {0}, {1}, {2}'.format(cmap.box_size(),
                                         cmap.apix,
                                         str(cmap.fullMap.dtype)))
    
    # exit(0)
    #newMap.write_to_MRC_file('mask3_clipperedmask.map')

    # plan_fft from tempy map_process array_utils.py
    # plan_fft(map_array, )
    # directly use fft = pyfftw.FFTW(arr,fftoutput,
    #                      direction='FFTW_FORWARD',axes=(0,1,2),
    #                      flags=['FFTW_ESTIMATE'])
    # arr =input array (map array)
    # real to complex = forward

    # use pyfftw if available
    pyfftw_flag = 1
    print(newMap.fullMap.dtype)
  
    # for p in range(0,gridtree[1].size):
    # grid sampling
    # swapshape = cmap.fullMap.swapaxes(0, 2).shape
    g = cmap.fullMap    # swapshape
    # u=x, v=y, w=z,
    g_reci_ = (cmap.z_size(), cmap.y_size(), int(cmap.x_size()/2+1))
    g_real_ = (g_reci_[0], g_reci_[1], int(g_reci_[2]*2))
    g_half_ = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2))

    # g_reci = (cmap.x_size(), cmap.y_size(), int(cmap.z_size()/2+1))
    # g_real = (g_reci[0], g_reci[1], int(g_reci[2]*2))
    # g_half = (int(cmap.x_size()/2), int(cmap.y_size()/2), int(cmap.z_size()/2))

    # fullMap is numpy array
    fltrmap = Filter(cmap)
    ftfilter = fltrmap.tanh_lowpass(0.5)
    lowpass_map = fltrmap.fourier_filter(ftfilter)
    print(lowpass_map.fullMap.shape, lowpass_map.apix) 
    data_r = lowpass_map.fullMap  # cmap.fullMap   # .swapaxes(0, 2)
    print(data_r.dtype)
    #data_c = data_r.astype('complex128')
    boxvol = map_box_vol(cmap)
    xdata_c = fourier_transform(data_r, boxvol)
    ydata_c = np.zeros(xdata_c.shape, dtype='complex128')
    zdata_c = np.zeros(xdata_c.shape, dtype='complex128')

    #print(data_c.dtype)
    print(data_r.dtype)
    #c = (cmap.x_size(), 0)
    ch = (int(cmap.z_size()/2), int(cmap.y_size()/2), int(cmap.x_size()/2))
    i = complex(0.0, 1.0)
    for cz in range(0, g_reci_[0]):    # z
        for cy in range(0, g_reci_[1]):     # y
            for cx in range(0, g_reci_[2]):    # x
                hkl = hkl_c((cz, cy, cx), ch, g_real_)
                cdata = i * xdata_c[cz, cy, cx]
                zdata_c[cz, cy, cx] = float((2*np.pi)*hkl.z) * cdata
                ydata_c[cz, cy, cx] = float((2*np.pi)*hkl.y) * cdata
                xdata_c[cz, cy, cx] = float((2*np.pi)*hkl.x) * cdata
                # hkl = hkl_c((cx, cy, cz), ch, g_real)
                # cdata = i * xdata_c[cx, cy, cz]
                # xdata_c[cx, cy, cz] = float((2*np.pi)*hkl.x) * cdata
                # ydata_c[cx, cy, cz] = float((2*np.pi)*hkl.y) * cdata
                # zdata_c[cx, cy, cz] = float((2*np.pi)*hkl.z) * cdata


    # calculate gradient maps
    # fft complex to real
    zdata_r = inv_fourier_transform(zdata_c, 1.0/boxvol, g_reci_)
    ydata_r = inv_fourier_transform(ydata_c, 1.0/boxvol, g_reci_)
    xdata_r = inv_fourier_transform(xdata_c, 1.0/boxvol, g_reci_)
    print(zdata_r.shape)
    print(ydata_r.shape)
    print(xdata_r.shape)
    print(cmap.fullMap.shape)
    # xdata_r = xdata_r.swapaxes(0, 2)
    # ydata_r = ydata_r.swapaxes(0, 2)
    # zdata_r = zdata_r.swapaxes(0, 2)    
    x1map = Map(np.zeros(lowpass_map.fullMap.shape),
                lowpass_map.origin,
                lowpass_map.apix[0],
                'mapname',)
    x2map = Map(np.zeros(cmap.fullMap.shape),
                lowpass_map.origin,
                lowpass_map.apix[0],
                'mapname',)
    x3map = Map(np.zeros(lowpass_map.fullMap.shape),
                lowpass_map.origin,
                lowpass_map.apix[0],
                'mapname',)
    
    print(x1map.apix, x2map.apix, x3map.apix)
    print(x1map.origin, x2map.origin, x3map.origin)
    
    #blurrer = StructureBlurrer()
    #gridtree = blurrer.maptree(x1map)

    #for ind in gridtree[1]:
    for x in range(0, lowpass_map.x_size()):
        for y in range(0, lowpass_map.y_size()):
            for z in range(0, lowpass_map.z_size()):
                x1map.fullMap[z, y, x] = xdata_r[z, y, x]
                x2map.fullMap[z, y, x] = ydata_r[z, y, x]
                x3map.fullMap[z, y, x] = zdata_r[z, y, x]
    # end map preparation
    x1map.write_to_MRC_file('x1map_7.map')
    x2map.write_to_MRC_file('x2map_7.map')
    x3map.write_to_MRC_file('x3map_7.map')
    
    # calculate XTY and apply mask
    # ymap=diffmap , mmap = mask
    y1map = x1map*ymap*mmap
    y2map = x2map*ymap*mmap
    y3map = x3map*ymap*mmap

    # calculate XTX and apply mask
    x11map = x1map*x1map*mmap
    x12map = x1map*x2map*mmap
    x13map = x1map*x2map*mmap
    x22map = x2map*x2map*mmap
    x23map = x2map*x3map*mmap
    x33map = x3map*x3map*mmap

    # filter

    # calculate U shifts
    for index in ymap:
        # vector v
        v = Vector(y1map[index], y2map[index], y3map[index])  # Vector(x, y, z) 
        # v[0] = y1map[index]
        # v[1] = y2map[index]
        # v[2] = y3map[index]

        # matrix(3,3)
        m = np.zeros([3, 3])
        m[0, 0] = x11map[index]
        m[0, 1] = m[1, 0] = x12map[index]
        m[0, 2] = m[2, 0] = x13map[index]
        m[1, 1] = x22map[index]
        m[1, 2] = m[2, 1] = x23map[index]
        m[2, 2] = x33map[index]

        # solve matrix v c++ m.solve(v)
    
        x1map[index] = v.x # [0]
        x2map[index] = v.y # [1]
        x3map[index] = v.z # [2]
    # 2 - z and x axis swap in np array becomes x,y,z 
    # 3 - back using z,y,x convention by tempy map
    # 4 - fixed hkl cv - chv to cv + chv ... no change since return modulus
    # 5 - tanh_lowpass filter
    # 6 - init x1,x2,x3maps with lowpass map apix
    # 7 - lowpass 
    # make maps
    

    # do a reciprocal grid and then hkl indices for gradient map coefficient calculation
    # need to do complex part and gradient calculation
    # main(mapin, filename)