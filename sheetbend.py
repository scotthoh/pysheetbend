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
hetatom = False
biso_range = SP.args.biso_range
ulo = sf_util.b2u(biso_range[0])
uhi = sf_util.b2u(biso_range[1])

timelog = sf_util.Profile()

# defaults
#if res <= 0.0:
#    print('Please specify the resolution of the map!')
#    exit()

if not refxyz and not refuiso:
    refxyz = True

if resbycyc is None:
    if res > 0.0:
        resbycyc = [res]
    else:
        # need to change this python3 can use , end=''
        print('Please specify the resolution of the map or')
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
    structure = PDBParser.read_PDB_file(struc_id,
                                        ippdb,
                                        hetatm=hetatom,
                                        water=False)
    original_structure = structure.copy()
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
    preg = pseudoregularizer.Pseudoregularize(original_structure)


# CASE 1:
# Refine model against EM data
if ippdb is not None:
    print('Refine Model against EM data')
    # calculate input map threshold/contour
    mapin_t = scorer.calculate_map_threshold(mapin)
    zg = np.linspace(0, mapin.z_size(), num=mapin.z_size(),
                     endpoint=False)
    yg = np.linspace(0, mapin.y_size(), num=mapin.y_size(),
                     endpoint=False)
    xg = np.linspace(0, mapin.x_size(), num=mapin.x_size(),
                     endpoint=False)
                
    # Loop over refine regularise cycles
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

            # truncate resolution - low pass filter
            # spherical tophat function fall=0.01 tophat
            fltrmap = Filter(mapin)
            # frequency from 0:0.5, 0.1 =10Angs, 0.5 = 2Angs ? or apix dependent?
            # 1/Angs = freq or apix/reso = freq?
            ftfilter = array_utils.tanh_lowpass(fltrmap.fullMap.shape,
                                                mapin.apix/rcyc, fall=0.5)
            lp_map = fltrmap.fourier_filter(ftfilter=ftfilter,
                                            inplace=False)
            lp_map.set_apix_tempy()
            map_curreso = Map(lp_map.fullMap, lp_map.origin,
                              lp_map.apix, mapin.filename)
            if verbose > 5:
                print(lp_map.__class__.__name__)
                print(lp_map.apix)
                print(map_curreso.__class__.__name__)
                print(map_curreso.apix)
            
            # Calculate electron density with b-factors from input model
            if verbose >= 1:
                start = timer()
            timelog.start('MapDensity')

            cmap = map_curreso.copy()
            cmap.fullMap = cmap.fullMap * 0
            cmap = structure.calculate_rho(2.5, mapin, cmap)
            #cmap = emc.calc_map_density(map_curreso, structure)
            fltr_cmap = Filter(cmap)
            ftfilter = array_utils.tanh_lowpass(fltr_cmap.fullMap.shape,
                                                mapin.apix/rcyc, fall=0.5)
            lp_cmap = fltr_cmap.fourier_filter(ftfilter=ftfilter,
                                               inplace=False)
            timelog.end('MapDensity')
            if verbose >= 1:
                end = timer()
                print('density calc ', end-start)
            if verbose > 5 and cyc == 0:
                DFM.write_mapfile(lp_cmap, 'cmap_cyc1.map')
                map_curreso.write_to_MRC_file('mapcurreso_cyc1.map')
            if verbose > 5 and cyc == ncyc-1:
                DFM.write_mapfile(lp_cmap, 'cmap_finalcyc.map')
                map_curreso.write_to_MRC_file('mapcurreso_final.map')
            # calculate fsc and envelope score(TEMPy) instead of R and R-free
            # use envelop score (TEMPy)... calculating average fsc
            # like refmac might take too much?
            if verbose >= 1:
                start = timer()
            timelog.start('Scoring')
            # calculate map contour
            mapcurreso_t = scorer.calculate_map_threshold(map_curreso)
            print('Calculated input map volume threshold is ',)  # end='')
            print('{0:.2f} and is {1:.2f} at current resolution.'
                  .format(mapin_t, mapcurreso_t))

            # calculate model contour
            t = 2.5 if rcyc > 10.0 else 2.0 if rcyc > 6.0 else 1.5
            '''if rcyc > 10.0:
                t = 2.5
            elif rcyc > 6.0:
                t = 2.0
            else:
                t = 1.5'''
            cmap_t = 1.5*cmap.std()
            fltrcmap_t = t*np.std(lp_cmap.fullMap)
            print('Calculated model threshold is ',)  # end='')
            print('{0:.2f} and is {1:.2f} at current resolution'
                  .format(cmap_t, fltrcmap_t))

            ovl_map1, ovl_mdl1 = scorer.calculate_overlap_scores(mapin, cmap,
                                                                 mapin_t,
                                                                 cmap_t)
            ovl_map2, ovl_mdl2 = scorer.calculate_overlap_scores(map_curreso,
                                                                 lp_cmap,
                                                                 mapcurreso_t,
                                                                 fltrcmap_t)
            timelog.end('Scoring')
            if verbose >= 1:
                end = timer()
                print('Score mod: {0} s'.format(end-start))

            print('TEMPys scores :')
            print('Fraction of map overlapping with model: {0:.3f} and {1:.3f} at \
                   current resolution'.format(ovl_map1, ovl_map2))
            print('Fraction of model overlapping with map: {0:.3f} and {1:.3f} at \
                   current resolution'.format(ovl_mdl1, ovl_mdl2))

            # make xmap
            apix = map_curreso.apix
            x_s = int(map_curreso.x_size() * apix)
            y_s = int(map_curreso.y_size() * apix)
            z_s = int(map_curreso.z_size() * apix)
            newMap = Map(np.zeros((z_s, y_s, x_s)),
                         map_curreso.origin,
                         apix,
                         'mapname',)
            newMap.apix = (apix * map_curreso.x_size()) / x_s
            newMap = newMap.downsample_map(map_curreso.apix,
                                           grid_shape=map_curreso.fullMap.shape)
            newMap.update_header()
            # maskmap at current reso
            #newMap = SB.make_atom_overlay_map1(newMap, structure)
            # mmap = mask
            newMap.fullMap[:] = 1.0
            mmap = newMap.copy()
            #mmap = sf_util.make_atom_overlay_map1_rad(newMap, structure,
            #                                          gridtree, 2.5)
            #print(np.count_nonzero(mmap.fullMap==0.0))
            # difference map at current reso, mmap
            #dmap = DFM.get_diffmap12(map_curreso, cmap, rcyc, rcyc)2
            if verbose >= 1:
                start = timer()
            timelog.start('DiffMap')
            dmap = DFM.get_diffmap12(lp_map, lp_cmap, rcyc, rcyc)
            timelog.end('DiffMap')
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
            x1map = Map(np.zeros(lp_cmap.fullMap.shape),
                        lp_cmap.origin,
                        lp_cmap.apix,
                        'mapname',)
            x2map = Map(np.zeros(lp_cmap.fullMap.shape),
                        lp_cmap.origin,
                        lp_cmap.apix,
                        'mapname',)
            x3map = Map(np.zeros(lp_cmap.fullMap.shape),
                        lp_cmap.origin,
                        lp_cmap.apix,
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
                # Read pdb and update
                # need to use fractional coordinates for updating
                # the derivatives.
                # use linear interpolation instead of cubic
                # size of x,y,z for x1map=x2map=x3map
                
                timelog.start('Interpolate')
                interp_x1 = RegularGridInterpolator((zg, yg, xg),
                                                    x1map.fullMap)
                interp_x2 = RegularGridInterpolator((zg, yg, xg),
                                                    x2map.fullMap)
                interp_x3 = RegularGridInterpolator((zg, yg, xg),
                                                    x3map.fullMap)

                count = 0
                v = structure.map_grid_position_array(map_curreso, False)
                v = np.flip(v, 1)
                du = 2.0*interp_x1(v)
                dv = 2.0*interp_x2(v)
                dw = 2.0*interp_x3(v)
                for i in range(len(structure)):
                    dx, dy, dz = np.matmul(np.array([du[i], dv[i], dw[i]]),
                                           cell.orthmat)
                    if verbose >= 1:
                        shift_vars.append([du[i], dv[i], dw[i],
                                          dx, dy, dz])
                    structure.atomList[i].translate(dx, dy, dz)
                timelog.end('Interpolate')

            # U-isotropic refinement
            if refuiso or (lastcyc and postrefuiso):
                print('REFINE U ISO')
                timelog.start('UISO')
                x1map = shiftfield.shift_field_uiso(lp_cmap, dmap, mmap, x1map,
                                                    radcyc, fltr, fft_obj,
                                                    ifft_obj, cell)
                timelog.end('UISO')
                timelog.start('Interpolate')
                interp_x1 = RegularGridInterpolator((zg, yg, xg),
                                                    x1map.fullMap)
                v = structure.map_grid_position_array(map_curreso, False)
                v = np.flip(v, 1)
                du = 1.0*interp_x1(v)
                du = sf_util.limit_uiso(du, ulo, uhi)
                db = sf_util.u2b(du)
                for i in range(len(structure)):
                    structure.atomList[i].temp_fac -= db[i]
                    if verbose >= 1:
                        shift_U.append([structure.atomList[i].temp_fac,
                                        du[i], db[i]])
                timelog.end('Interpolate')
                    
            temp_result = sf_util.ResultsByCycle(cycrr, cyc, rcyc, radcyc,
                                                 ovl_map1, ovl_map2,
                                                 ovl_mdl1, ovl_mdl2, 0.0)
            '''
            temp_result = results_by_cycle()
            temp_result.cycle = cyc
            temp_result.cyclerr = cycrr
            temp_result.resolution = rcyc
            temp_result.radius = radcyc
            temp_result.envscore = score_mod
            temp_result.envscore_reso = score_mod_curreso
            temp_result.fscavg = 0.0
            '''
            results.append(temp_result)
            if output_intermediate:
                outname = '{0}_{1}.pdb'.format(oppdb.strip('.pdb'), cyc+1)
                structure.write_to_PDB(outname, hetatom=hetatom)
            if len(shift_vars) != 0:
                outcsv = 'shiftvars1_withorthmat_{0}.csv'.format(cyc+1)
                fopen = open(outcsv, 'w')
                for j in range(0, len(shift_vars)):
                    fopen.write('{0}, {1}\n'.format(j, shift_vars[j]))
                fopen.close()
            if len(shift_U) != 0:
                outusiocsv = 'shiftuiso_u2b_{0}.csv'.format(cyc+1)
                fuiso = open(outusiocsv, 'w')
                for j in range(0, len(shift_U)):
                    fuiso.write('{0}, {1}\n'.format(j, shift_U[j]))
                fuiso.close()
            # end of cycle loop
        if pseudoreg:
            print('PSEUDOREGULARIZE')
            timelog.start('PSEUDOREG')
            start = timer()
            structure = preg.regularize_frag(structure)
            timelog.end('PSEUDOREG')
            end = timer()
            timelog.start('MapDensity')
            cmap = mapin.copy()
            cmap.fullMap = cmap.fullMap * 0
            cmap = structure.calculate_rho(2.5, mapin, cmap)
            #cmap = emc.calc_map_density(mapin, structure)
            timelog.start('MapDensity')
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

        # end of psedo reg loop

# CASE 2: refine against observations
# Initial application should be to improve map quality and local resolution,
# and reduce the impact of the choice of numner of classes.
# Thus, this is more suitable to be used in refinement of 3D classes followed
# by averaging of those classes
'''
if ippdb is None:
    print('Refine Map against another Map.')
    # 1. Don't read model, read map
    # 2. Before first cycle, make shift maps
    # 3. In first cycle, make shifted maps using shifts
    # 4. Run shiftfield, accumulate shifts on shifted maps
    # 5. At end of cycle, update shift maps
    # 6. Calculate final map after cycle loop
    mapin_t = scorer.calculate_map_threshold(mapin)
    if ipmap2 is None:
        print('No work map provided to refine. Please use option -m2|--mapin2')
        exit('Progrma terminated...')
    
    mapin2 = mp.readMRC(ipmap2)
    map_shifted = Map(mapin2)  # make shifted map
    map_shifted.fullMap = map_shifted.fullMap * 0
    for cycrr in range(0, ncycrr):
'''


# write final pdb
if ippdb is not None:
    structure.write_to_PDB('{0}_final.pdb'.format(oppdb.strip('.pdb')),
                           hetatom=hetatom)

# write xml results
if xmlout is not None:
    f = open(xmlout, 'w')
    for i in range(0, len(results)):
        if i == 0:
            results[i].write_xml_results_start(f)
        results[i].write_xml_results_cyc(f)
        if i == len(results)-1:
            results[i].write_xml_results_end(f)
    f.close()

timelog.profile_log()

# need to use ccpem python tempy for the diff map
