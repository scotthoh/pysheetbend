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
from scipy.interpolate import Rbf
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

ipmap1 = SP.args.mapin1
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
radscl = SP.args.radscl  # 5.0
xmlout = SP.args.xmlout  # program.xml
output_intermediate = SP.args.intermediate  # False
verbose = SP.args.verbose  # 0
ncycrr = 1  # refine-regularise-cycle
fltr = 2  # quadratic filter
hetatom = False
biso_range = SP.args.biso_range
ulo = sf_util.b2u(biso_range[0])
uhi = sf_util.b2u(biso_range[1])

# temp
inclconst = False
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
        print('Please specify the resolution of the map or \
               resolution by cycle.')
        exit()

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
    structure_instance = PDBParser.read_PDB_file(struc_id,
                                                 ippdb,
                                                 hetatm=hetatom,
                                                 water=False)
    original_structure = structure_instance.copy()
# read map
if ipmap is not None:
    mapin = mp.readMRC(ipmap)
    scorer = ScoringFunctions()
    SB = StructureBlurrer()
    gridtree = SB.maptree(mapin)
    # get cell details
    cell = sf_util.Cell(mapin.header[10], mapin.header[11], mapin.header[12],
                        mapin.header[13], mapin.header[14], mapin.header[15])
    # set gridshape
    # maybe need to do a prime number check on grid dimensions
    gridshape = sf_util.grid_dim(mapin)
    if verbose >= 1:
        start = timer()
    fft_obj = sf_util.plan_fft(gridshape)
    if verbose >= 1:
        end = timer()
        print('plan fft : {0} s'.format(end-start))

    if verbose >= 1:
        start = timer()
    ifft_obj = sf_util.plan_ifft(gridshape)
    if verbose >= 1:
        end = timer()
        print('plan ifft : {0} s'.format(end-start))
if pseudoreg:
    print('PSEUDOREGULARIZE')
    preg = pseudoregularizer.Pseudoregularize(original_structure)


# CASE 1:
# Refine model against EM data
if ippdb is not None:
    print('Refine Model against EM data')
    # Loop over refine regularise cycles
    for cycrr in range(0, ncycrr):
        print('\nRefine-regularise cycle: {0}\n'.format(cycrr+1))
        # calculate input map threshold/contour
        mapin_t = scorer.calculate_map_threshold(mapin)
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
            cmap = emc.calc_map_density(map_curreso, structure_instance)
            fltr_cmap = Filter(cmap)
            ftfilter = array_utils.tanh_lowpass(fltr_cmap.fullMap.shape,
                                                mapin.apix/rcyc, fall=0.5)
            lp_cmap = fltr_cmap.fourier_filter(ftfilter=ftfilter,
                                               inplace=False)
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
            # calculate map contour
            mapcurreso_t = scorer.calculate_map_threshold(map_curreso)
            print('Calculated input map volume threshold is ', end='')
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
            print('Calculated model threshold is ', end='')
            print('{0:.2f} and is {1:.2f} at current resolution'
                  .format(cmap_t, fltrcmap_t))

            ovl_map1, ovl_mdl1 = scorer.calculate_overlap_scores(mapin, cmap,
                                                                 mapin_t,
                                                                 cmap_t)
            ovl_map2, ovl_mdl2 = scorer.calculate_overlap_scores(map_curreso,
                                                                 lp_cmap,
                                                                 mapcurreso_t,
                                                                 fltrcmap_t)
            
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
            #newMap = SB.make_atom_overlay_map1(newMap, structure_instance)
            # mmap = mask
            newMap.fullMap[:] = 1.0
            mmap = newMap.copy()
            #mmap = sf_util.make_atom_overlay_map1_rad(newMap, structure_instance,
            #                                          gridtree, 2.5)
            #print(np.count_nonzero(mmap.fullMap==0.0))
            # difference map at current reso, mmap
            #dmap = DFM.get_diffmap12(map_curreso, cmap, rcyc, rcyc)2

            if verbose >= 1:
                start = timer()
            dmap = DFM.get_diffmap12(lp_map, lp_cmap, rcyc, rcyc)
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
                x1map, x2map, x3map = shiftfield.shift_field_coord(lp_cmap,
                                                                   dmap, mmap,
                                                                   x1map,
                                                                   x2map,
                                                                   x3map,
                                                                   radcyc,
                                                                   fltr,
                                                                   fft_obj,
                                                                   ifft_obj)

                # Read pdb and update
                # need to use fractional coordinates for updating
                # the derivatives.
                # rbf can be used just need to crop down to atm point
                # in map 64 points around (4,4,4) similar to clipper
                count = 0
                for atm in structure_instance.atomList:
                    # transform back to orth_coord after getting du, dv, dw
                    # use scipy griddate gd((x,y,z), v, (X,Y,Z), method='cubic')
                    # map array in fullMap is in Z,Y,X so the point to
                    # interpolate should by z,y,x instead of x,y,z
                    # get mapgridpos of atom
                    try:
                        ax, ay, az, am = SB.mapGridPosition(map_curreso, atm)
                    except TypeError:
                        print(atm.write_to_PDB())
                        exit()
                    xyz0, nb_xyz = sf_util.get_lowerbound_posinnewbox(ax, ay,
                                                                      az)
                    if verbose >= 1:
                        start = timer()
                    local_x1map = sf_util.crop_mapgrid_points(xyz0[0], xyz0[1],
                                                              xyz0[2], x1map)
                    local_x2map = sf_util.crop_mapgrid_points(xyz0[0], xyz0[1],
                                                              xyz0[2], x2map)
                    local_x3map = sf_util.crop_mapgrid_points(xyz0[0], xyz0[1],
                                                              xyz0[2], x3map)
                    if verbose >= 1:
                        end = timer()
                        if count == 0:
                            print('Crop map : {0}'.format(end-start))
                        
                    zg, yg, xg = np.mgrid[0:local_x1map.z_size(),
                                          0:local_x1map.y_size(),
                                          0:local_x1map.x_size()]
                    rbf_x = xg.ravel()
                    rbf_y = yg.ravel()
                    rbf_z = zg.ravel()
                    if verbose >= 1:
                        start = timer()
                    rbf_x1 = Rbf(rbf_z, rbf_y, rbf_x, local_x1map.fullMap,
                                 function='cubic', mode='3-D')
                    rbf_x2 = Rbf(rbf_z, rbf_y, rbf_x, local_x2map.fullMap,
                                 function='cubic', mode='3-D')
                    rbf_x3 = Rbf(rbf_z, rbf_y, rbf_x, local_x3map.fullMap,
                                 function='cubic', mode='3-D')
                    if verbose >= 1:
                        end = timer()
                        if count == 0:
                            print('Interpolate time : {0}'.format(end-start))
                    du = 2.0*rbf_x1(nb_xyz[2], nb_xyz[1], nb_xyz[0])
                    dv = 2.0*rbf_x2(nb_xyz[2], nb_xyz[1], nb_xyz[0])
                    dw = 2.0*rbf_x3(nb_xyz[2], nb_xyz[1], nb_xyz[0])
                    # transform to orthogonal coordinates
                    dxyz = np.matmul(np.array([du, dv, dw]), cell.orthmat)
                    if verbose >= 1:
                        shift_vars.append([du, dv, dw, dxyz])
                    atm.set_x(atm.x + dxyz[0])
                    atm.set_y(atm.y + dxyz[1])
                    atm.set_z(atm.z + dxyz[2])
                    count += 1

            # U-isotropic refinement
            if refuiso or (lastcyc and postrefuiso):
                print('REFINE U ISO')
                x1map = shiftfield.shift_field_uiso(lp_cmap, dmap, mmap, x1map,
                                                    radcyc, fltr, fft_obj,
                                                    ifft_obj, cell)

                for atm in structure_instance.atomList:
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
                    zg, yg, xg = np.mgrid[0:local_x1map.z_size(),
                                          0:local_x1map.y_size(),
                                          0:local_x1map.x_size()]
                    rbf_x = xg.ravel()
                    rbf_y = yg.ravel()
                    rbf_z = zg.ravel()
                    # Interpolation
                    rbf_x1 = Rbf(rbf_z, rbf_y, rbf_x, local_x1map.fullMap,
                                 function='cubic', mode='3-D')
                    du = 1.0*rbf_x1(nb_xyz[2], nb_xyz[1], nb_xyz[0])
                    if verbose >= 1:
                        shift_U.append([atm.temp_fac, du, sf_util.u2b(du)])
                    
                    du = uhi if du > uhi else ulo if du < ulo else du
                    atm.temp_fac = atm.temp_fac - sf_util.u2b(du)
                    
            temp_result = sf_util.results_by_cycle(cycrr, cyc, rcyc, radcyc,
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
                structure_instance.write_to_PDB(outname, hetatom=hetatom)
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
            start = timer()
            structure_instance = preg.regularize_frag(structure_instance)
            end = timer()
            cmap = emc.calc_map_density(mapin, structure_instance)
            cmap_t = 1.5*cmap.std()
            ovl_mapf, ovl_mdlf = scorer.calculate_overlap_scores(mapin, cmap,
                                                                 mapin_t,
                                                                 cmap_t)
            print('TEMPys scores :')
            print('Fraction of map overlapping with model: {0:.3f}'
                  .format(ovl_mapf))
            print('Fraction of model overlapping with map: {0:.3f}'
                  .format(ovl_mdlf))
            print('time : ', end-start)

        # end of psedo reg loop

# write final pdb
structure_instance.write_to_PDB('{0}_final.pdb'.format(oppdb.strip('.pdb')),
                                hetatom=hetatom)

# write xml results
f = open(xmlout, 'w')
for i in range(0, len(results)):
    if i == 0:
        results[i].write_xml_results_start(f)
    results[i].write_xml_results_cyc(f)
    if i == len(results)-1:
        results[i].write_xml_results_end(f)

f.close()

# need to use ccpem python tempy for the diff map
