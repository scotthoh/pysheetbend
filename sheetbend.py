"""
Python implementation of csheetbend to perform shift field refinement
Copyright 2018 Kevin Cowtan & University of York all rights reserved
Author: Soon Wen Hoh, University of York 2020

License: ?
"""

from __future__ import print_function  # python 3 proof
import sys
import os
import datetime
from timeit import default_timer as timer
from scipy.ndimage.measurements import maximum
from scipy.interpolate import RegularGridInterpolator
from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.StructureParser import PDBParser
from TEMPy.MapParser import MapParser as mp
from TEMPy.mapprocess.mapfilters import Filter
from TEMPy.mapprocess import array_utils
from TEMPy.ProtRep_Biopy import BioPy_Structure
import numpy as np
import numpy.ma as ma
import shiftfield
import shiftfield_util as sf_util
import pseudoregularizer
import map_scaling as DFM
from sheetbend_cmdln_parser import SheetbendParser

# sys.path.append('/home/swh514/Projects/testing_ground')
# sys.path.append('/y/people/swh514/Documents/Projects/sheetbend_python')
# import scale_map.map_scaling as DFM

"""
# original downloaded TEMPy source not CCP-EM TEMPy source
from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.protein.scoring_functions import ScoringFunctions
from TEMPy.protein.structure_parser import PDBParser
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.maps.em_map import Map
from TEMPy.map_process.map_filters import Filter
from TEMPy.map_process import array_utils
"""

# tracemalloc


def has_converged(model0, model1, coor_tol, bfac_tol):
    coor_sum_sq = 0
    bfac_sum_sq = 0
    num_atms = len(model0)
    for n in range(0, num_atms):
        d = model0.atomList[n].distance_from_atom(model1.atomList[n])
        coor_sum_sq += np.square(d)
        d_bfac = model0.atomList[n].temp_fac - model1.atomList[n].temp_fac
        bfac_sum_sq += np.square(d_bfac)

    coor_rmsd = np.sqrt(coor_sum_sq / num_atms)
    bfac_rmsd = np.sqrt(bfac_sum_sq / num_atms)
    print("Testing for convergence")
    print(f"Coordinate RMSD : {coor_rmsd}")
    print(f"B-factor RMSD : {bfac_rmsd}")
    return coor_rmsd < coor_tol and bfac_rmsd < bfac_tol


# tracemalloc.start()
# Parse command line input
print(f"Started at {datetime.datetime.now()}")
SP = SheetbendParser()
SP.get_args()
# Set variables from parsed arguments
ippdb = SP.args.pdbin
ipmap = SP.args.mapin
if ippdb is None and ipmap is None:
    print(
        "No input map or input structure. What do you want to do?\n\
          Program terminated..."
    )
    exit()

nomask = SP.args.nomask
ipmap2 = SP.args.mapin2
ipmask = SP.args.maskin
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
hetatm_present = False
biso_range = SP.args.biso_range
# ulo = sf_util.b2u(biso_range[0])
# uhi = sf_util.b2u(biso_range[1])
mid_pseudoreg = False  # True
timelog = sf_util.Profile()
SP.print_args()
# defaults
# if res <= 0.0:
#    print('Please specify the resolution of the map!')
#    exit()

if not refxyz and not refuiso:
    refxyz = True
    print("Setting refxyz to True")

if resbycyc is None:
    if res > 0.0:
        resbycyc = [res]
    else:
        # need to change this python3 can use , end=''
        print("Please specify the resolution of the map or")
        print("resolution by cycle.")
        exit("Program terminated ...")
elif res > 0.0:
    if res > resbycyc[-1]:
        resbycyc0 = resbycyc
        resbycyc = [resbycyc0[0], res]

if refuiso or postrefuiso:
    print("B-factor bounds : {0} < B < {1}".format(biso_range[0], biso_range[1]))

"""
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
"""

print("Radscl: {0}".format(radscl))
if len(resbycyc) == 0:
    resbycyc.append(res)

# initialise results list
results = []

# read model
if ippdb is not None:
    struc_id = os.path.basename(ippdb).split(".")[0]
    structure = PDBParser.read_PDB_file(struc_id, ippdb, hetatm=hetatom, water=False)
    new_reordered_struct = []
    chainList = structure.split_into_chains()
    for c in chainList:
        c.reorder_residues()
        new_reordered_struct = np.append(new_reordered_struct, c)

    # original_structure = structure.copy()
    structure = BioPy_Structure(new_reordered_struct)
    for atm in structure.atomList:
        if atm.record_name == "HETATM":
            hetatm_present = True
            break
    original_structure = structure.copy()
# read map
if ipmap is not None:
    mapin = mp.readMRC(ipmap)
    apix0 = mapin.apix
    ori0 = mapin.origin
    # print(f"mapin dtype : {mapin.fullMap.dtype}")
    # mapin.write_to_MRC_file('mapin_after_read.map')
    scorer = ScoringFunctions()
    # gridtree = SB.maptree(mapin)
    # get cell details
    cell = sf_util.Cell(
        mapin.header[10],
        mapin.header[11],
        mapin.header[12],
        mapin.header[13],
        mapin.header[14],
        mapin.header[15],
    )
    # set gridshape
    # maybe need to do a prime number check on grid dimensions
    # gridshape = sf_util.GridDimension(mapin)
    # timelog.start("fftplan")
    # fft_obj = sf_util.plan_fft(gridshape)
    # timelog.end("fftplan")

    # timelog.start("ifftplan")
    # ifft_obj = sf_util.plan_ifft(gridshape)
    # timelog.end("ifftplan")
    if isinstance(apix0, tuple):
        nyquist_res = maximum([2.0 * apix0[i] for i in (0, 1, 2)])
        min_d = maximum(apix0)
    else:
        nyquist_res = 2.0 * apix0
        min_d = apix0

    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)
    else:
        samp_rate = 1.5  # default same value from CLIPPER

if pseudoreg or mid_pseudoreg:
    print("PSEUDOREGULARIZE")
    preg = pseudoregularizer.Pseudoregularize(original_structure)

# create mask map if non is given at input
# a mask that envelopes the whole particle/volume of interest
if nomask:
    # mmap = mapin.copy()
    mmap = ma.make_mask_none(mapin.fullMap.shape)
    # mmap = ma.masked_array(mapin.fullMap, mask=mask_arr)
    # print(mmap)
    # mmap.fullMap[:] = 1.0
    # print(f"mmap dtype : {mmap.fullMap.dtype}")
    ipmask = mmap.copy()

if ipmask is None:
    timelog.start("MaskMap")
    fltrmap = Filter(mapin)
    ftfilter = array_utils.tanh_lowpass(fltrmap.fullMap.shape, apix0 / 15.0, fall=1)
    print("fullmap {0}, fltr {1} ".format(fltrmap.fullMap.shape, ftfilter.shape))
    lp_maskin = fltrmap.fourier_filter(ftfilter=ftfilter, inplace=False)
    mapt = scorer.calculate_map_threshold(lp_maskin)
    mmap = mapin.copy()
    mmap.fullMap = (lp_maskin.fullMap > mapt) * 1.0
    soft_mask_arr = array_utils.softmask_mean(mmap.fullMap, window=5)
    mmap.fullMap = soft_mask_arr
    mmap.update_header()
    scorer.calculate_map_threshold(mmap)
    mmap.fullMap = (soft_mask_arr > mapt) * 1.0
    mmap.update_header()
    timelog.end("MaskMap")
elif not nomask:
    maskin = mp.readMRC(ipmask)
    mmap = ma.make_mask(maskin)
    # invert the values for numpy masked_array, true is masked/invalid, false is valid
    mmap = ~mmap

# CASE 1:
# Refine model against EM data
if ippdb is not None:
    print("Refine Model against EM data")
    # calculate input map threshold/contour
    mapin_t = scorer.calculate_map_threshold(mapin)
    """zg = np.linspace(0, mapin.z_size(), num=mapin.z_size(),
                     endpoint=False)
    yg = np.linspace(0, mapin.y_size(), num=mapin.y_size(),
                     endpoint=False)
    xg = np.linspace(0, mapin.x_size(), num=mapin.x_size(),
                     endpoint=False)
    """
    for cycrr in range(0, ncycrr):
        print("\nRefine-regularise cycle: {0}\n".format(cycrr + 1))
        # previously: loop over cycles
        # new: loop over resolutions
        # for ires in range(0, len(resbycyc)):
        #    lastres = (ires == len(resbycyc)-1)
        #    rcyc = resbycyc[ires]
        #    #coor_tol = 0.01 * rcyc
        #    #bfac_tol = 0.5 * rcyc

        for cyc in range(0, ncyc):
            shift_vars = []
            shift_U = []
            # check for final cycle
            lastcyc = True if cyc == ncyc - 1 else False

            # set resolution
            fcyc = (cyc) / max(float(ncyc - 1), 1.0)
            fres = fcyc * float(len(resbycyc) - 1)
            ires0 = int(fres)
            ires1 = min(ires0 + 1, int(len(resbycyc) - 1))
            dres = fres - float(ires0)
            rcyc = resbycyc[ires0] + dres * (resbycyc[ires1] - resbycyc[ires0])

            # set radius if not user specified
            radcyc = rad
            if radcyc <= 0.0:
                radcyc = radscl * rcyc
            print(
                "\nCycle: {0}   Resolution: {1}   Radius: {2}\n".format(
                    cyc + 1, rcyc, radcyc
                )
            )

            # downsample maps
            # changes 22-25 Oct 21 downsampling maps (coarser grids),
            # larger pixel size/spacing used. res/(2*samprate) = min spacing,
            # samp rate default = 1.5 ; took gemmi code
            # faster overall; but should be able to optimize more by calculating
            # better gridshapes.
            # don't have to use lowpass as it doesn't affect results much
            spacing = rcyc / (2 * samp_rate)
            print(f"calculated spacing : {spacing}")
            downsamp_shape, spacing = sf_util.calc_best_grid_apix(
                spacing, (cell.a, cell.b, cell.c)
            )

            downsamp_map = mapin.downsample_map(spacing, downsamp_shape)
            downsamp_map.update_header()
            downsamp_shape = downsamp_map.fullMap.shape
            downsamp_apix = downsamp_map.apix

            mmap = ma.make_mask_none(downsamp_shape)
            zg = np.linspace(
                0, downsamp_map.z_size(), num=downsamp_map.z_size(), endpoint=False
            )
            yg = np.linspace(
                0, downsamp_map.y_size(), num=downsamp_map.y_size(), endpoint=False
            )
            xg = np.linspace(
                0, downsamp_map.x_size(), num=downsamp_map.x_size(), endpoint=False
            )

            gridshape = sf_util.GridDimension(downsamp_shape)
            timelog.start("fftplan")
            fft_obj = sf_util.plan_fft(gridshape)
            timelog.end("fftplan")

            timelog.start("ifftplan")
            ifft_obj = sf_util.plan_ifft(gridshape)
            timelog.end("ifftplan")
            print(f"downsample shape : {downsamp_shape}")
            print(f"downsample apix : {downsamp_apix}")
            if verbose >= 1:
                start = timer()
            timelog.start("MapDensity")

            cmap = downsamp_map.copy()
            cmap.fullMap = cmap.fullMap * 0
            cmap = structure.calculate_rho(2.5, downsamp_map, cmap)
            timelog.end("MapDensity")
            if verbose >= 1:
                end = timer()
                print("density calc ", end - start)

            # calculate difference map
            # truncate resolution - low pass filter; lpfiltb = True
            # lowpass filtering map doesn't result in much difference
            # spherical tophat function fall=0.01 tophat
            # in terms of b-factor shifts
            # refsc > dust,refsc=False > refsc, dust > dust
            # in terms of model better to worst
            # dust,refsc=False > refsc > dust > refsc, dust
            if verbose >= 1:
                start = timer()
            timelog.start("DiffMap")
            scl_map, scl_cmap, dmap = DFM.get_diffmap12(
                downsamp_map,
                cmap,
                rcyc,
                rcyc,
                cyc=cyc + 1,
                lpfiltb=False,
                flag_dust=False,
                refsc=False,
                verbose=verbose,
            )
            timelog.end("DiffMap")
            if verbose >= 1:
                end = timer()
                print("Diff map calc: {0} s ".format(end - start))

            # fltrmap = Filter(downsamp_map) #mapin)
            # print(f'fltrmap apix : {fltrmap.apix}')
            # fltrmap.apix = apix0
            # frequency from 0:0.5, 0.1 =10Angs, 0.5 = 2Angs ? or apix dependent?
            # 1/Angs = freq or apix/reso = freq?
            # print(f'filtermap dtype : {fltrmap.fullMap.dtype}')
            # ftfilter = array_utils.tanh_lowpass(downsamp_shape,   #fltrmap.fullMap.shape,
            #                                    downsamp_apix/rcyc, fall=0.2)
            # lp_map = fltrmap.fourier_filter(ftfilter=ftfilter,
            #                                inplace=False)
            # print(f'check apix lpmap {lp_map.apix}')
            # lp_map.set_apix_tempy()
            # lp_map.apix = apix0
            # map_curreso = Map(downsamp_map.fullMap, ori0,
            #                  downsamp_apix, mapin.filename)
            if verbose >= 0:
                print("check apix")
                print(mapin.__class__.__name__)
                print(mapin.apix)
                print(scl_map.__class__.__name__)
                print(scl_map.apix)
                print(scl_cmap.__class__.__name__)
                print(scl_cmap.apix)

            # Calculate electron density with b-factors from input model
            # if verbose >= 1:
            #    start = timer()
            # timelog.start('MapDensity')

            # cmap = map_curreso.copy()
            # cmap.fullMap = cmap.fullMap * 0
            # cmap = structure.calculate_rho(2.5, downsamp_map, cmap)
            # cmap.write_to_MRC_file('cmap.map')
            # with mrcfile.open('cmap.map', mode='r+', permissive=True) as mrc:
            #    mrc.update_header_from_data()

            # cmap = emc.calc_map_density(map_curreso, structure)
            # fltr_cmap = Filter(cmap)
            # fltr_cmap.apix = apix0
            # ftfilter = array_utils.tanh_lowpass(downsamp_shape,
            #                                    downsamp_apix/rcyc, fall=0.2)
            # lp_cmap = fltr_cmap.fourier_filter(ftfilter=ftfilter,
            #                                   inplace=False)
            # timelog.end('MapDensity')
            # if verbose >= 1:
            #    end = timer()
            #    print('density calc ', end-start)
            # if verbose > 5 and cyc == 0:
            # DFM.write_mapfile(scl_cmap, f'cmap_cyc{cyc+1}.map')
            # DFM.write_mapfile(scl_map, f'mapcurreso_cyc{cyc+1}.map')
            # mapin.write_to_MRC_file('mapin.map')
            # try servalcat fofc calc
            # structure.write_to_PDB('temp_structfile.pdb', hetatom=False)
            # mapmask_arr = emda_methods.mask_from_map((cell.c, cell.b, cell.a,
            #                                          cell.gamma, cell.beta,
            #                                          cell.alpha),
            #                                         mapin.fullMap)
            # mapmask_arr = np.flip(mapmask_arr, 1)
            # emda_methods.write_mrc(mapmask_arr, 'mapmask_emda.map',
            #                       (cell.a, cell.b, cell.c,
            #                        cell.alpha, cell.beta, cell.gamma))
            # mapmask = mapin.copy()
            # mapmask.fullMap = cmap.fullMap * 0
            # mapmask.fullMap = mapmask_arr.copy()
            # mapmask.update_header()
            # mapmask.write_to_MRC_file('mapmask_emda.map')
            # fofc_args = f'--model temp_structfile.pdb --resolution {rcyc} '
            # fofc_args += f'--map {ipmap} --mask mapmask.mrc '
            # fofc_args += '--normalized_map'
            # proc1 = subprocess.Popen("servalcat fofc {0}".format(fofc_args),
            #                         stdout=subprocess.DEVNULL,
            #                         shell=True)
            # print('Running servalcat fofc calc')
            # sys.stdout.flush()
            # proc1.wait()
            # diffmapin = mp.readMRC('diffmap_normalized_fofc.mrc')
            # if verbose > 5 and cyc == ncyc-1:
            #    DFM.write_mapfile(lp_cmap, 'cmap_finalcyc.map')
            #    map_curreso.write_to_MRC_file('mapcurreso_final.map')
            # calculate fsc and envelope score(TEMPy) instead of R and R-free
            # use envelop score (TEMPy)... calculating average fsc
            # like refmac might take too much?
            if verbose >= 1:
                start = timer()

            timelog.start("Scoring")
            # calculate map contour
            dmapin_t = scorer.calculate_map_threshold(downsamp_map)
            mapcurreso_t = scorer.calculate_map_threshold(scl_map)
            print("Calculated input map volume threshold is ", end="")
            print(
                "{0:.2f} and {1:.2f} (current resolution).".format(
                    dmapin_t, mapcurreso_t
                )
            )

            # calculate model contour
            t = 2.5 if rcyc > 10.0 else 2.0 if rcyc > 6.0 else 1.5
            """if rcyc > 10.0:
                t = 2.5
            elif rcyc > 6.0:
                t = 2.0
            else:
                t = 1.5"""
            cmap_t = 1.5 * cmap.std()
            # fltrcmap_t = t*scl_cmap.std() #np.std(scl_cmap.fullMap)
            fltrcmap_t = scorer.calculate_map_threshold(scl_cmap)
            print("Calculated model threshold is ", end="")
            print("{0:.2f} and {1:.2f} (current resolution)".format(cmap_t, fltrcmap_t))

            ovl_map1, ovl_mdl1 = scorer.calculate_overlap_scores(
                downsamp_map, cmap, dmapin_t, cmap_t
            )
            ovl_map2, ovl_mdl2 = scorer.calculate_overlap_scores(
                scl_map, scl_cmap, mapcurreso_t, fltrcmap_t
            )
            timelog.end("Scoring")
            if verbose >= 1:
                end = timer()
                print("Score mod: {0} s".format(end - start))

            print("TEMPys scores :")
            print("Fraction of map overlapping with model: ", end="")
            print("{0:.3f} and {1:.3f} (current resolution)".format(ovl_map1, ovl_map2))
            print("Fraction of model overlapping with map: ", end="")
            print("{0:.3f} and {1:.3f} (current resolution)".format(ovl_mdl1, ovl_mdl2))
            # if ovl_mdl1 > 5.0:
            #    mid_pseudoreg = True
            # make xmap
            # apix = map_curreso.apix
            # x_s = int(map_curreso.x_size() * apix)
            # y_s = int(map_curreso.y_size() * apix)
            # z_s = int(map_curreso.z_size() * apix)
            # newMap = Map(np.zeros((z_s, y_s, x_s)),
            #             map_curreso.origin,
            #             apix,
            #             'mapname',)
            # newMap.apix = (apix * map_curreso.x_size()) / x_s
            # newMap = newMap.downsample_map(map_curreso.apix,
            #                               grid_shape=map_curreso.fullMap.shape)
            # newMap.update_header()
            # maskmap at current reso
            # newMap = SB.make_atom_overlay_map1(newMap, structure)
            # mmap.fullMap += newMap.fullMap
            # mmap.fullMap = (mmap.fullMap >= 1.0) * 1.0
            # mmap.write_to_MRC_file('create_maskout.map')

            # mmap = mask
            # newMap.fullMap[:] = 1.0
            # mmap = newMap.copy()
            # mmap = sf_util.make_atom_overlay_map1_rad(newMap, structure,
            #                                          gridtree, 2.5)
            # print(np.count_nonzero(mmap.fullMap==0.0))
            # difference map at current reso, mmap
            # dmap = DFM.get_diffmap12(lp_map, lp_cmap, rcyc, rcyc)
            # if verbose >= 1:
            #    start = timer()
            # timelog.start('DiffMap')
            # scl_map, scl_cmap, dmap = DFM.get_diffmap12(lp_map, lp_cmap,
            #                                            rcyc, rcyc,
            #                                            cyc=cyc+1,
            #                                            verbose=verbose)
            # timelog.end('DiffMap')
            # if verbose >= 1:
            #    end = timer()
            #    print('Diff map calc: {0} s '.format(end-start))
            # xyz shiftfield refine pass in cmap, dmap, mmap, x1map, x2map,
            # x3map, radcyc, fltr, fftobj, iffobj
            """
            cmap.write_to_MRC_file('cmap.map')
            
            newdmap = Map(np.zeros(dmap.fullMap.shape),
                          list(dmap.origin),
                          dmap.apix, 'mapname')
            dmap.set_newmap_data_header(newdmap)
            newdmap.update_header()
            dmapfname = "newdmap_{0}.map".format(cyc+1)
            newdmap.write_to_MRC_file(dmapfname)
            """
            # mmap.write_to_MRC_file('mmap.map')

            # x1map = Map(np.zeros(scl_cmap.fullMap.shape),
            #            scl_cmap.origin,
            #            downsamp_apix,
            #            'mapname',)
            # x2map = Map(np.zeros(scl_cmap.fullMap.shape),
            #            scl_cmap.origin,
            #            downsamp_apix,
            #            'mapname',)
            # x3map = Map(np.zeros(scl_cmap.fullMap.shape),
            #            scl_cmap.origin,
            #            downsamp_apix,
            #            'mapname',)

            if refxyz:
                print("REFINE XYZ")
                timelog.start("Shiftfield")
                # print(f'maps dtype, lpcmap : {scl_cmap.fullMap.dtype}')
                # print(f'mmap dtype : {mmap.fullMap.dtype}')
                print(f"dmap dtype : {dmap.fullMap.dtype}")
                # lp_cmap.fullMap = lp_cmap.fullMap.astype('float64')
                # dmap.fullMap = dmap.fullMap.astype('float64')
                x1m, x2m, x3m = shiftfield.shift_field_coord(
                    scl_cmap.fullMap,
                    dmap.fullMap,
                    mmap,
                    radcyc,
                    fltr,
                    ori0,
                    downsamp_apix,
                    fft_obj,
                    ifft_obj,
                    cyc + 1,
                    verbose=verbose,
                )
                # x1m, x2m, x3m = shiftfield.shift_field_coord(lp_cmap.fullMap,
                #                                             dmap.fullMap,
                #                                             mmap,
                #                                             radcyc, fltr,
                #                                             lp_cmap.origin,
                #                                             lp_cmap.apix,
                #                                             fft_obj,
                #                                             ifft_obj, cyc+1,
                #                                             verbose=verbose)

                """
                x1map, x2map, x3map = shiftfield.shift_field_coord(lp_cmap,
                                                                   dmap, mmap,
                                                                   x1map,
                                                                   x2map,
                                                                   x3map,
                                                                   radcyc,
                                                                   fltr,
                                                                   fft_obj,
                                                                   ifft_obj)"""
                # x1map.fullMap = x1m.copy()
                # x2map.fullMap = x2m.copy()
                # x3map.fullMap = x3m.copy()
                timelog.end("Shiftfield")
                # Read pdb and update
                # use linear interpolation instead of cubic
                # size of x,y,z for x1map=x2map=x3map
                # print(x1map.fullMap.box_size())
                timelog.start("Interpolate")
                interp_x1 = RegularGridInterpolator((zg, yg, xg), x1m)
                interp_x2 = RegularGridInterpolator((zg, yg, xg), x2m)
                interp_x3 = RegularGridInterpolator((zg, yg, xg), x3m)

                count = 0
                v = structure.map_grid_position_array(scl_map, False)
                v = np.flip(v, 1)
                dx = (2.0 * interp_x1(v)) * cell.a
                dy = (2.0 * interp_x2(v)) * cell.b
                dz = (2.0 * interp_x3(v)) * cell.c
                timelog.end("Interpolate")
                timelog.start("UpdateModel")
                for i in range(len(structure)):
                    # dx, dy, dz = np.matmul(np.array([du[i], dv[i], dw[i]]),
                    #                       cell.orthmat)
                    if verbose >= 1:
                        shift_vars.append(
                            [
                                structure.atomList[i].get_name(),
                                structure.atomList[i].get_x(),
                                structure.atomList[i].get_y(),
                                structure.atomList[i].get_z(),
                                dx[i],
                                dy[i],
                                dz[i],
                            ]
                        )  # du[i], dv[i], dw[i],
                    structure.atomList[i].translate(dx[i], dy[i], dz[i])
                timelog.end("UpdateModel")

            if mid_pseudoreg:
                print("PSEUDOREGULARIZE")
                timelog.start("MIDPSEUDOREG")
                structure = preg.regularize_frag(structure)
                timelog.end("MIDPSEUDOREG")
                # timelog.start('MapDensity')
                # cmap = mapin.copy()
                # cmap.fullMap = cmap.fullMap * 0
                # cmap = structure.calculate_rho(2.5, mapin, cmap)
                ##cmap = emc.calc_map_density(mapin, structure)
                # timelog.start('MapDensity')
                # timelog.start('Scoring')    print('PSEUDOREGULARIZE')
                # cmap_t = 1.5*cmap.std()
                # ovl_mapf, ovl_mdlf = scorer.calculate_overlap_scores(mapin, cmap,
                #                                                    mapin_t,
                #                                                    cmap_t)
                # timelog.end('Scoring')
                # print('TEMPys scores :')
                # print('Fraction of map overlapping with model: {0:.3f}'
                #    .format(ovl_mapf))
                # print('Fraction of model overlapping with map: {0:.3f}'
                #    .format(ovl_mdlf))
                # print('time : ', end-start)

            # U-isotropic refinement

            if refuiso or (postrefuiso and lastcyc):
                print("REFINE U ISO")
                timelog.start("UISO")
                x1m = shiftfield.shift_field_uiso(
                    scl_cmap.fullMap,
                    dmap.fullMap,
                    mmap,
                    radcyc,
                    fltr,
                    ori0,
                    downsamp_apix,
                    fft_obj,
                    ifft_obj,
                    (cell.a, cell.b, cell.c),
                )
                # x1map.fullMap = x1m.copy()
                timelog.end("UISO")
                timelog.start("Interpolate")
                interp_x1 = RegularGridInterpolator((zg, yg, xg), x1m)
                v = structure.map_grid_position_array(scl_map, False)
                v = np.flip(v, 1)
                du = 1.0 * interp_x1(v)
                db = sf_util.u2b(du)
                timelog.end("Interpolate")
                timelog.start("UpdateModel")
                for i in range(len(structure)):
                    temp_fac = structure.atomList[i].temp_fac - db[i]
                    temp_fac = sf_util.limit_biso(
                        temp_fac, biso_range[0], biso_range[1]
                    )
                    structure.atomList[i].temp_fac = temp_fac
                    if verbose >= 1:
                        shift_U.append([structure.atomList[i].temp_fac, du[i], db[i]])
                timelog.end("UpdateModel")
            temp_result = sf_util.ResultsByCycle(
                cycrr, cyc, rcyc, radcyc, ovl_map1, ovl_map2, ovl_mdl1, ovl_mdl2, 0.0
            )

            """
            temp_result = results_by_cycle()
            temp_result.cycle = cyc
            temp_result.cyclerr = cycrr
            temp_result.resolution = rcyc
            temp_result.radius = radcyc
            temp_result.envscore = score_mod
            temp_result.envscore_reso = score_mod_curreso
            temp_result.fscavg = 0.0
            """
            results.append(temp_result)
            if output_intermediate:
                outname = "{0}_{1}.pdb".format(oppdb.strip(".pdb"), cyc + 1)
                structure.write_to_PDB(outname, hetatom=hetatm_present)
            if len(shift_vars) != 0 and verbose >= 2:
                outcsv = "shiftvars1_linalg_{0}.csv".format(cyc + 1)
                fopen = open(outcsv, "w")
                for j in range(0, len(shift_vars)):
                    fopen.write("{0}, {1}\n".format(j, shift_vars[j]))
                fopen.close()
            sys.stdout.flush()

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #    print(stat)

            if len(shift_U) != 0 and verbose >= 2:
                outusiocsv = "shiftuiso_u2b_{0}.csv".format(cyc + 1)
                fuiso = open(outusiocsv, "w")
                for j in range(0, len(shift_U)):
                    fuiso.write("{0}, {1}\n".format(j, shift_U[j]))
                fuiso.close()
            # end of cycle loop
        if pseudoreg:
            print("PSEUDOREGULARIZE")
            timelog.start("PSEUDOREG")
            structure = preg.regularize_frag(structure)
            timelog.end("PSEUDOREG")
            timelog.start("MapDensity")
            cmap = mapin.copy()
            cmap.fullMap = cmap.fullMap * 0
            cmap = structure.calculate_rho(2.5, mapin, cmap)
            # cmap = emc.calc_map_density(mapin, structure)
            timelog.start("MapDensity")
            timelog.start("Scoring")
            cmap_t = 1.5 * cmap.std()
            ovl_mapf, ovl_mdlf = scorer.calculate_overlap_scores(
                mapin, cmap, mapin_t, cmap_t
            )
            timelog.end("Scoring")
            print(f"End of refine-regularise cycle {cycrr+1}")
            print("TEMPys scores :")
            print("Fraction of map overlapping with model: {0:.3f}".format(ovl_mapf))
            print("Fraction of model overlapping with map: {0:.3f}".format(ovl_mdlf))
        # end of psedo reg loop

# CASE 2: refine against observations
# Initial application should be to improve map quality and local resolution,
# and reduce the impact of the choice of numner of classes.
# Thus, this is more suitable to be used in refinement of 3D classes followed
# by averaging of those classes
"""
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
"""


# write final pdb
if oppdb is not None:
    outfname = "{0}_sheetbendfinal.pdb".format(oppdb.strip(".pdb"))
    structure.write_to_PDB(f"{outfname}", hetatom=hetatm_present)  # preg.got_hetatm
else:
    outfname = "{0}_sheetbendfinal.pdb".format(ippdb.strip(".pdb"))
    structure.write_to_PDB(f"{outfname}", hetatom=hetatm_present)

# write xml results
if xmlout is not None:
    f = open(xmlout, "w")
    for i in range(0, len(results)):
        if i == 0:
            results[i].write_xml_results_start(f, outfname, ippdb)
        results[i].write_xml_results_cyc(f)
        if i == len(results) - 1:
            results[i].write_xml_results_end(f)
    f.close()

print(f"Ended at {datetime.datetime.now()}")
timelog.profile_log()


# need to use ccpem python tempy for the diff map
