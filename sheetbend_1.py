# Python implementation of csheebend to perform shift field refinement
# Copyright 2018 Kevin Cowtan & University of York all rights reserved
# Author: Soon Wen Hoh, University of York 2020
'''from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.protein.scoring_functions import ScoringFunctions
from TEMPy.protein.structure_parser import PDBParser
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.maps.em_map import Map
from TEMPy.math.vector import Vector
from TEMPy.map_process.map_filters import Filter
'''
from __future__ import print_function # python 3 proof 
from TEMPy.StructureBlurrer import StructureBlurrer
from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.StructureParser import PDBParser
from TEMPy.MapParser import MapParser as mp
from TEMPy.EMMap import Map
from TEMPy.mapprocess.mapfilters import Filter
from TEMPy.mapprocess import array_utils

import numpy as np
from timeit import default_timer as timer
import esf_map_calc as emc
import shiftfield as shiftfield
import shiftfield_util as sf_util
from scipy.interpolate import Rbf, RegularGridInterpolator
import pseudoregularizer
import os
import scale_map.map_scaling as DFM

from sheetbend_cmdln_parser import sheetbendParser
# Parse command line input
SP = sheetbendParser()
SP.get_args()
# Set variables from parsed arguments
ippdb = SP.args.pdbin
ipmap = SP.args.mapin
if ippdb is None and ipmap is None:
    print('No input map or input structure. What do you want to do?\n\
          Program terminated...')
    exit()

ipmap2 = SP.args.mapin2
opmap = SP.args.mapout  # sheetbend_mapout_result.map
oppdb = SP.args.pdbout  # sheetbend_pdbout_result.pdb
res = SP.args.res  # -1.0
resbycyc = SP.args.res_by_cyc  # None
ncyc = SP.args.cycle  # 1
refxyz = SP.args.refxyz  # False
refuiso = SP.args.refuiso  # False
postrefxyz = SP.args.postrefxyz  # False
postrefuiso = SP.args.postrefuiso  # False
pseudoreg = SP.args.pseudoreg  # False
rad = SP.args.rad  # -1.0
radscl = SP.args.radscl  # 4.0
xmlout = SP.args.xmlout  # program.xml
output_intermediate = SP.args.intermediate  # False
verbose = SP.args.verbose  # 0
ncycrr = 1  # refine-regularise-cycle
fltr = 2  # quadratic filter
hetatom = True
biso_range = SP.args.biso_range
#ulo = sf_util.b2u(biso_range[0])
#uhi = sf_util.b2u(biso_range[1])

timelog = sf_util.Profile()
run_c = '9'
# defaults
#if res <= 0.0:
#    print('Please specify the resolution of the map!')
#    exit()
SP.print_args()

if not refxyz and not refuiso:
    refxyz = True

if resbycyc is None:
    if res > 0.0:
        resbycyc = [res]
    else:
        # need to change this python3 can use , end=''
        print('Please specify the resolution of the map or ',end='')
        print('resolution by cycle.')
        exit('Program terminated ...')

if refuiso or postrefuiso:
    print('B-factor bounds : {0} < B < {1}'.format(biso_range[0],
                                                   biso_range[1]))

'''
#ippdb = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1.ent'
ippdb = '/home/swh514/Projects/testing_ground/shiftfield_python/testrun/translate_1_5ni1_a.pdb'    #translate_1_5ni1_a.pdb'
#ippdb = '/home/swh514/Projects/testing_ground/shiftfield_python/testrun/check_FT/trans1angxyz_uiso/initial_testout_sheetbend1_withorthmat_3.pdb'
ipmap = '/home/swh514/Projects/data/EMD-3488/map/emd_3488.map'
rad = -1.0
radscl = 4.0 # def 4.0
res = 6
ncyc = 2
ncycrr = 1 # refine-regularise-cycle
fltr = 2 # quadratic
inclconst = False
#print(inclconst)
resbycyc = [6.0, 3.0]
refuiso = False
postrefuiso = True
pseudoreg = True
hetatom = False
# defaults
refxyz = True # in future for xyx, uiso, aniso
'''

print('Radscl: {0}'.format(radscl))
if len(resbycyc) == 0:
    resbycyc.append(res)

# initialise results list
results = []

# read model
if ippdb is not None:
    struc_id = os.path.basename(ippdb).split('.')[0]
    # for psedoregularizer use BioPy structure
    # original_structure_BioPy = PDBParser.read_PDB_file_BioPy(struc_id, 
    #                                                         ippdb,
    #                                                         hetatm=hetatom,
    #                                                         water=False)
    # TEMPy structure class
    model = PDBParser.read_PDB_file(struc_id, ippdb, hetatm=hetatom,
                                    water=False)
    original_structure = model.copy()
# read map
if ipmap is not None:
    mapin = mp.readMRC(ipmap)
    scorer = ScoringFunctions()
    SB = StructureBlurrer()
    # gridtree = SB.maptree(mapin)
    # get cell details
    cell = sf_util.Cell(mapin.header[10], mapin.header[11], mapin.header[12],
                        mapin.header[13], mapin.header[14], mapin.header[15])
    # set gridshape
    # maybe need to do a prime number check on grid dimensions
    gridshape = sf_util.GridDimension(mapin)
    timelog.start('fftplan')
    fft_obj = sf_util.plan_fft(gridshape)
    timelog.end('fftplan')

    timelog.start('ifftplan')
    ifft_obj = sf_util.plan_ifft(gridshape)
    timelog.end('ifftplan')

if pseudoreg:
    print('PSEUDOREGULARIZE')
    timelog.start('Pseudoreg')
    preg = pseudoregularizer.Pseudoregularize(original_structure)
    timelog.end('Pseudoreg')

# CASE 1:
# Refine model against EM data
# Applying the accumulating shift method used to refine map against observations
if ippdb is not None:
    print('Refine Model against EM data')
    # 1. Read model and observed EM map
    # 2. Before first cycle, make shift maps
    # 3. In first cycle, make shifted maps using shifts
    # 4. Run shiftfield, accumulate shifts on shifted maps
    # 5. At end of cycle, update shift maps
    # 6. Calculate final map after cycle loop
    # 7. Apply shifts
    # Calculate input map threshold/contour
    #apix = mapin.apix
    #x_s = int(mapin.x_size() * apix)
    results = []
    map_shifted = mapin.copy()
    map_shifted.fullMap = map_shifted.fullMap * 0        
    dumap = map_shifted.copy()
    dvmap = map_shifted.copy()
    dwmap = map_shifted.copy()
    # calculate map from structure
    cmap = mapin.copy()
    cmap.fullMap = cmap.fullMap * 0
    timelog.start('MapDensityCalc')
    cmap = model.calculate_rho(2.5, mapin, cmap)
    timelog.end('MapDensityCalc')
    timelog.start('Scoring')
    mapin_t = scorer.calculate_map_threshold(mapin)
    cmap_t = 1.5*cmap.std()
    ovl_map1, ovl_mdl1 = scorer.calculate_overlap_scores(mapin, cmap,
                                                         mapin_t, cmap_t)
    timelog.end('Scoring')
    print('Calculated map, model thresholds : {0:.4f}, {1:.4f}'
          .format(mapin_t, cmap_t))
    print('Initial Map-Model Overlap Score : {0:.4f}'.format(ovl_map1))
    print('Initial Model-Map Overlap Score : {0:.4f}'.format(ovl_mdl1))
    # setup regular grid interpolator for cmap
    timelog.start('InterpolSetup')
    nz, ny, nx = mapin.fullMap.shape
    zls = np.linspace(0, nz, num=nz, endpoint=False)
    yls = np.linspace(0, ny, num=ny, endpoint=False)
    xls = np.linspace(0, nx, num=nx, endpoint=False)
    cmap_interp = RegularGridInterpolator((zls, yls, xls), cmap.fullMap,
                                          bounds_error=False, fill_value=0.0)
    timelog.end('InterpolSetup')
    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    indi = np.vstack([zg.ravel(), yg.ravel(), xg.ravel()]).T
    im = np.zeros([cmap.fullMap.size, 3])
    dumap_f = np.ravel(dumap.fullMap)
    dvmap_f = np.ravel(dvmap.fullMap)
    dwmap_f = np.ravel(dwmap.fullMap)
    #indi_frac = indi / np.array([nz, ny, nx])

    # mask map
    
    '''
    apix = mapin.apix
    x_s = int(map_shifted.x_size() * apix)
    y_s = int(map_shifted.y_size() * apix)
    z_s = int(map_shifted.z_size() * apix)
    newMap = Map(np.zeros((z_s, y_s, x_s)),
                         map_shifted.origin,
                         apix,
                         'mapname',)
    newMap.apix = (apix * map_shifted.x_size()) / x_s
    newMap = newMap.downsample_map(mapin.apix,
                                   grid_shape=mapin.fullMap.shape)
    newMap.update_header()
    newMap = model.make_mask(2.5, newMap)
    mmap = newMap.copy()
    '''
    # mmap = mask
    #newMap.fullMap[:] = 1.0
    
    # maskmap at current reso
    #testmaskmap = Map(np.zeros((z_s, y_s, x_s)),
    #             map_shifted.origin,
    #             apix,
    #             'mapname',)
    #testmaskmap.apix = (apix * map_shifted.x_size()) / x_s
    #testmaskmap = newMap.downsample_map(mapin.apix,
    #                               grid_shape=mapin.fullMap.shape)
    #testmaskmap.update_header()
    #testmaskmap = SB.make_atom_overlay_map1(newMap, model)
    #timelog.end('MaskMap')

    # make mask from input map, lowpass 15A,
    timelog.start('MaskMap')
    fltrmap = Filter(mapin)
    ftfilter = array_utils.tanh_lowpass(fltrmap.fullMap.shape,
                                        mapin.apix/15.0,
                                        fall=1)
    lp_mapin = fltrmap.fourier_filter(ftfilter=ftfilter,
                                      inplace=False)

    mapt = scorer.calculate_map_threshold(lp_mapin)
    print(mapt)
    mmap = mapin.copy()
    mmap.fullMap = (lp_mapin.fullMap > mapt) * 1.0
    soft_mask_arr = array_utils.softmask_mean(mmap.fullMap, window=5)
    mmap.fullMap = soft_mask_arr
    mmap.update_header()
    scorer.calculate_map_threshold(mmap)
    mmap.fullMap = (soft_mask_arr > mapt) * 1.0
    mmap.update_header()
    timelog.end('MaskMap')
    #testmaskmap.write_to_MRC_file('testmaskmap_{0}.map'.format(run_c))

    for cycrr in range(0, ncycrr):
        print('\nRefine-regularise cycle: {0}\n'.format(cycrr+1))
        # loop over cycles
        for cyc in range(0, ncyc):
            shift_vars = []
            shift_U = []
            # check for final cycle
            lastcyc = True if cyc == ncyc-1 else False

            # set resolution
            fcyc = float(cyc) / max(float(ncyc-1), 1.0)
            fres = fcyc*float(len(resbycyc)-1)
            ires0 = int(fres)
            ires1 = min(ires0+1, int(len(resbycyc)-1))
            dres = fres - float(ires0)
            rcyc = resbycyc[ires0] + dres*(resbycyc[ires1] - resbycyc[ires0])

            # set radius if not user specified
            radcyc = rad
            if radcyc <= 0.0:
                radcyc = radscl*rcyc
            print('\nCycle: {0}   Resolution: {1}   Radius: {2}\n'
                  .format(cyc+1, rcyc, radcyc))

            # calculate map value (interpolate cmap)
            timelog.start('CalcMapVal')
            im[:, 0] = dwmap_f * map_shifted.z_size()
            im[:, 1] = dvmap_f * map_shifted.y_size()
            im[:, 2] = dumap_f * map_shifted.x_size()
            c_grid = indi - im
            '''outcsv = 'shiftmaps_withorthmat_{0}.csv'.format(cyc+1)
            fopen = open(outcsv, 'w')
            for j in range(0, len(dumap_f)):
                fopen.write('{0}, {1}, {2}, {3}, {4}\n'.format(j, indi[j], dumap_f[j], dvmap_f[j], dwmap_f[j]))
            fopen.close()'''
            interp_val = cmap_interp(c_grid)
            map_shifted.fullMap = interp_val.reshape(cmap.box_size())
            timelog.end('CalcMapVal')
            # problem with this shiftmap values
            #map_shifted.write_to_MRC_file('map_shifted_0.map')
            # continue from here, 24May21
            print("Done map density calc.")
            # make xmaps
            # truncate resolution - low pass filter
            # spherical tophat function fall=0.01 tophat
            # frequency from 0:0.5, 0.1 =10Angs, 0.5 = 2Angs ? or apix dependent?
            # 1/Angs = freq or apix/reso = freq?
            timelog.start('LowPassMaps')
            fltrmap = Filter(mapin)
            ftfilter = array_utils.tanh_lowpass(fltrmap.fullMap.shape,
                                                map_shifted.apix/rcyc,
                                                fall=0.5)
            lp_map = fltrmap.fourier_filter(ftfilter=ftfilter,
                                            inplace=False)
            lp_map.set_apix_tempy()
            fltr_cmap = Filter(map_shifted)
            ftfilter = array_utils.tanh_lowpass(fltr_cmap.fullMap.shape,
                                                map_shifted.apix/rcyc,
                                                fall=0.5)
            lp_cmap = fltr_cmap.fourier_filter(ftfilter=ftfilter,
                                               inplace=False)
            timelog.end('LowPassMaps')
            timelog.start('Scoring')
            lpmap_t = scorer.calculate_map_threshold(lp_map)
            # calculate model contour
            t = 2.5 if rcyc > 10.0 else 2.0 if rcyc > 6.0 else 1.5
            mapshift_t = 1.5*map_shifted.std()
            fltrcmap_t = t*np.std(lp_cmap.fullMap)
            print('Calculated map threshold is ', end='')
            print('{0:.4f} and {1:.4f} at current resolution'
                  .format(mapin_t, lpmap_t))

            print('Calculated model threshold is ', end='')
            print('{0:.4f} and {1:.4f} at current resolution'
                  .format(mapshift_t, fltrcmap_t))

            ovl_map1, ovl_mdl1 = scorer.calculate_overlap_scores(mapin,
                                                                 map_shifted,
                                                                 mapin_t,
                                                                 mapshift_t)
            ovl_map2, ovl_mdl2 = scorer.calculate_overlap_scores(lp_map,
                                                                 lp_cmap,
                                                                 lpmap_t,
                                                                 fltrcmap_t)
            timelog.end('Scoring')
            print('TEMPys scores :')
            print('Fraction of map overlapping with model: ', end='')
            print('{0:.4f} and {1:.4f} (current resolution)'
                  .format(ovl_map1, ovl_map2))
            print('Fraction of model overlapping with map: ', end='')
            print('{0:.4f} and {1:.4f} (current resolution)'
                  .format(ovl_mdl1, ovl_mdl2))

            #if ovl_mdl1 > 0.5:
            #    pseudoreg = True
            #mmap = sf_util.make_atom_overlay_map1_rad(newMap, model,
            #                                          gridtree, 2.5)
            #print(np.count_nonzero(mmap.fullMap==0.0))
            # difference map at current reso, mmap
            #dmap = DFM.get_diffmap12(map_curreso, cmap, rcyc, rcyc)2
            if verbose >= 1:
                start = timer()
            timelog.start('DiffMap')
            dmap = DFM.get_diffmap12(lp_map, lp_cmap, rcyc, rcyc)
            timelog.end('DiffMap')
            #outmmap = Map(np.zeros(mapin.fullMap.shape),
            #           list(mapin.origin),
            #            mapin.apix, 'mapname')
            #testmaskmap.set_newmap_data_header(outmmap)
            #outmmap.update_header()
            
            if verbose >= 1:
                end = timer()
                print('Diff map calc: {0} s '.format(end-start))
            # xyz shiftfield refine pass in cmap, dmap, mmap, x1map, x2map,
            # x3map, radcyc, fltr, fftobj, iffobj
            print("REFINE XYZ")
            '''
            cmap.write_to_MRC_file('cmap.map')
            newdmap = Map(np.zeros(dmap.fullMap.shape),
                        list(dmap.origin),
                        dmap.apix, 'mapname')
            dmap.set_newmap_data_header(newdmap)
            newdmap.update_header()
            newdmap.write_to_MRC_file('newdmap.map')
            mmap.write_to_MRC_file('mmap.map')
            '''
            x1map = Map(np.zeros(map_shifted.fullMap.shape),
                        map_shifted.origin,
                        map_shifted.apix,
                        'mapname',)
            x2map = Map(np.zeros(map_shifted.fullMap.shape),
                        map_shifted.origin,
                        map_shifted.apix,
                        'mapname',)
            x3map = Map(np.zeros(map_shifted.fullMap.shape),
                        map_shifted.origin,
                        map_shifted.apix,
                        'mapname',)
            
            if refxyz:
                timelog.start('Shiftfield')
                x1map, x2map, x3map = shiftfield.shift_field_coord(lp_cmap,
                                                                   dmap, mmap,
                                                                   x1map,
                                                                   x2map,
                                                                   x3map,
                                                                   radcyc,
                                                                   fltr,
                                                                   fft_obj,
                                                                   ifft_obj)
                timelog.end('Shiftfield')
                timelog.start('Interpolate')
                interp_x1 = RegularGridInterpolator((zls, yls, xls),
                                                    x1map.fullMap)
                interp_x2 = RegularGridInterpolator((zls, yls, xls),
                                                    x2map.fullMap)
                interp_x3 = RegularGridInterpolator((zls, yls, xls),
                                                    x3map.fullMap)
                #x1map.write_to_MRC_file('x1out.map')
                #x2map.write_to_MRC_file('x2out.map')
                #x3map.write_to_MRC_file('x3out.map')
                # continue from here 26 May. use grid coord for interpolation
                # interpolate values and convert from fraction to orthogonal values
                
                dumap_f += 2.0*interp_x1(indi)  # np.modf(2.0*interp_x1(indi))[0]  # x direction
                dvmap_f += 2.0*interp_x2(indi)  # np.modf(2.0*interp_x2(indi))[0]  # y direction
                dwmap_f += 2.0*interp_x3(indi)  # np.modf(2.0*interp_x3(indi))[0]  # z direction
                timelog.end('Interpolate')
            '''
            if refuiso or (lastcyc and postrefuiso):
                print('REFINE U ISO')
                timelog.start('UISO')
                x1map = shiftfield.shift_field_uiso(lp_cmap, dmap, mmap, x1map,
                                                    radcyc, fltr, fft_obj,
                                                    ifft_obj, cell)
                timelog.end('UISO')
                for atm in model.atomList:
                    timelog.start('CropMap')
                    try:
                        ax, ay, az, amass = SB.mapGridPosition(map_curreso,
                                                               atm)
                    except TypeError:
                        print(atm.write_to_PDB())
                        exit()
                    xyz0, nb_xyz = sf_util.get_lowerbound_posinnewbox(ax, ay,
                                                                      az)
                    local_x1map = sf_util.crop_mapgrid_points(xyz0[0], xyz0[1],
                                                              xyz0[2], x1map)
                    timelog.end('CropMap')
                    zg, yg, xg = np.mgrid[0:local_x1map.z_size(),
                                          0:local_x1map.y_size(),
                                          0:local_x1map.x_size()]
                    rbf_x = xg.ravel()
                    rbf_y = yg.ravel()
                    rbf_z = zg.ravel()
                    # Interpolation
                    timelog.start('Interpolate')
                    rbf_x1 = Rbf(rbf_z, rbf_y, rbf_x, local_x1map.fullMap,
                                 function='cubic', mode='3-D')
                    timelog.end('Interpolate')
                    du = 1.0*rbf_x1(nb_xyz[2], nb_xyz[1], nb_xyz[0])
                    if verbose >= 1:
                        shift_U.append([atm.temp_fac, du, sf_util.u2b(du)])
                    
                    du = uhi if du > uhi else ulo if du < ulo else du
                    atm.temp_fac = atm.temp_fac - sf_util.u2b(du)
            '''
            temp_result = sf_util.ResultsByCycle(cycrr, cyc, rcyc, radcyc,
                                                 ovl_map1, ovl_map2,
                                                 ovl_mdl1, ovl_mdl2, 0.0)
            results.append(temp_result)

    # calculate final map & applying final shifts
    #outcsv = 'shiftvars_total_{0}.csv'.format(run_c)
    #fopen = open(outcsv, 'w')
    #for j in range(0, len(dumap_f)):
    #    fopen.write('{0}, {1}, {2}, {3}\n'.format(j, dumap_f[j], dvmap_f[j],
    #                dwmap_f[j]))
    #fopen.close()

    timelog.start('Interpolate')
    interp_dwmap = RegularGridInterpolator((zls, yls, xls),
                                           dwmap.fullMap)
    interp_dvmap = RegularGridInterpolator((zls, yls, xls),
                                           dvmap.fullMap)
    interp_dumap = RegularGridInterpolator((zls, yls, xls),
                                           dumap.fullMap)

    p = model.map_grid_position_array(mapin, False)
    p = np.flip(p, 1)
    du = interp_dumap(p)
    dv = interp_dvmap(p)
    dw = interp_dwmap(p)
    timelog.end('Interpolate')
    timelog.start('UpdateModel')
    for i in range(len(model)):
        dx, dy, dz = np.matmul(np.array([du[i], dv[i], dw[i]]),
                               cell.orthmat)
        #dx, dy, dz = dxyz[i]
        if verbose >= 1:
            shift_vars.append([du[i], dv[i], dw[i],
                              dx, dy, dz])
        model.atomList[i].translate(dx, dy, dz)
    timelog.end('UpdateModel')
    model.write_to_PDB('{0}_final_{1}.pdb'.format(oppdb.strip('.pdb'), run_c),
                       hetatom=hetatom)

    if pseudoreg:
        print('PSEUDOREGULARIZE')
        timelog.start('PSEUDOREG')
        start = timer()
        model = preg.regularize_frag(model)
        timelog.end('PSEUDOREG')
        end = timer()
        timelog.start('MapDensityCalc')
        cmap = mapin.copy()
        cmap.fullMap = cmap.fullMap * 0
        cmap = model.calculate_rho(2.5, mapin, cmap)
        #cmap = emc.calc_map_density(mapin, model)
        timelog.end('MapDensityCalc')
        timelog.start('Scoring')
        cmap_t = 1.5*cmap.std()
        ovl_mapf, ovl_mdlf = scorer.calculate_overlap_scores(mapin, cmap,
                                                             mapin_t,
                                                             cmap_t)
        timelog.end('Scoring')
        print('TEMPys scores :')
        print('Fraction of map overlapping with model: {0:.3f}'
              .format(ovl_mapf))
        print('Fraction of model overlapping with map: {0:.3f}'
              .format(ovl_mdlf))
        print('time : ', end-start)
    
    if refuiso:
        print('REFINE U ISO')
        cmap = mapin.copy()
        cmap.fullMap = cmap.fullMap * 0
        timelog.start('MapDensityCalc')
        cmap = model.calculate_rho(2.5, mapin, cmap)
        timelog.end('MapDensityCalc')
        timelog.start('DiffMap')
        dmap = DFM.get_diffmap12(mapin, cmap, resbycyc[-1], resbycyc[-1])
        timelog.end('DiffMap')
        x1map = Map(np.zeros(map_shifted.fullMap.shape),
                    map_shifted.origin,
                    map_shifted.apix,
                    'mapname',)
        timelog.start('UISO')
        x1map = shiftfield.shift_field_uiso(cmap, dmap, mmap, x1map,
                                            radcyc, fltr, fft_obj,
                                            ifft_obj, cell)
        timelog.end('UISO')
        timelog.start('Interpolate')
        interp_x1 = RegularGridInterpolator((zls, yls, xls),
                                            x1map.fullMap)
        v = model.map_grid_position_array(mapin, False)
        v = np.flip(v, 1)
        du = 1.0*interp_x1(v)
        timelog.end('Interpolate')
        timelog.start('UpdateModel')
        #du = sf_util.limit_uiso(du, ulo, uhi)
        db = sf_util.u2b(du)
        for i in range(len(model)):
            atm_bval = model.atomList[i].temp_fac - db[i]
            model.atomList[i].temp_face = sf_util.limit_biso(atm_bval,
                                                             biso_range[0],
                                                             biso_range[1])
            if verbose >= 1:
                shift_U.append([model.atomList[i].temp_fac, du[i], db[i]])
        timelog.end('UpdateModel')

    model.write_to_PDB('{0}_final_pseudoreg_{1}.pdb'
                       .format(oppdb.strip('.pdb'), run_c), hetatom=hetatom)
    outcsv = 'shiftvars_atoms_total_{0}.csv'.format(run_c)
    uisocsv = 'shiftuiso_atoms_total_{0}.csv'.format(run_c)
    fopen = open(outcsv, 'w')
    uisoopen = open(uisocsv, 'w')
    if verbose >= 1:
        for j in range(0, len(shift_vars)):
            fopen.write('{0}, {1}\n'.format(j, shift_vars[j]))
            uisoopen.write('{0}, {1}\n'.format(j, shift_U[j]))
        fopen.close()
        uisoopen.close()

    # calculate final map applying final shifts
    timelog.start('CalcMapVal')
    im[:, 0] = dwmap_f * map_shifted.z_size()
    im[:, 1] = dvmap_f * map_shifted.y_size()
    im[:, 2] = dumap_f * map_shifted.x_size()
    c_grid = indi - im
    interp_val = cmap_interp(c_grid)
    map_shifted.fullMap = interp_val.reshape(cmap.box_size()).copy()
    timelog.end('CalcMapVal')
    map_shifted.write_to_MRC_file('mapshift_{0}.map'.format(run_c))

    timelog.profile_log()
    if xmlout is not None:
        f = open(xmlout, 'w')
        for i in range(0, len(results)):
            if i == 0:
                results[i].write_xml_results_start(f)
            results[i].write_xml_results_cyc(f)
            if i == len(results)-1:
                results[i].write_xml_results_end(f)
        f.close()
    #cmap.fullMap = cmap.fullMap * 0
    #cmap = model.calculate_rho(2.5, mapin, cmap)
    # make shifted map
    #map_shifted = Map(mapin)
    #map_shifted.fullMap = map_shifted.fullMap * 0
    #dumap, dvmap, dwmap = Map(map_shifted), Map(map_shifted), Map(map_shifted)
    #print(dumap.apix, dvmap.apix, dwmap.apix)
    
    #dvmap.write_to_MRC_file('dvmap,map')
    #dwmap.write_to_MRC_file('dwmap,map')
    #exit()

