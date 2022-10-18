"""
Python implementation of csheetbend to perform shift field refinement
Copyright 2018 Kevin Cowtan & University of York all rights reserved
Author: Soon Wen Hoh, University of York 2020

License: GNU LESSER GENERAL PUBLIC LICENSE v2.1
"""

from __future__ import print_function, absolute_import
from argparse import ArgumentError
from struct import Struct  # python 3 proof
import sys
import datetime
from timeit import default_timer as timer
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

# from pysheetbend.utils.logger import log2file
from pysheetbend.utils.dencalc import calculate_density, calculate_density_with_boxsize
import gemmi

#
# from TEMPy.ScoringFunctions import ScoringFunctions
# from TEMPy.MapParser import MapParser as mp
import numpy as np
import numpy.ma as ma
from pysheetbend.shiftfield import shift_field_coord, shift_field_uiso
import pysheetbend.shiftfield_util as sf_util

# import pysheetbend.pseudoregularizer as pseudoregularizer
import pysheetbend.pseudoregularizer_gemmi as pseudoregularizer

# import pysheetbend.map_scaling as DFM
from pysheetbend.utils import fileio, map_funcs, cell, dencalc

# from memory_profiler import profile

# from utils.map_funcs import prepare_mask_filter

# sys.path.append('/home/swh514/Projects/testing_ground')
# sys.path.append('/y/people/swh514/Documents/Projects/sheetbend_python')
# import scale_map.map_scaling as DFM

# original downloaded TEMPy source not CCP-EM TEMPy source
# from TEMPy.protein.structure_blurrer import StructureBlurrer
# from TEMPy.protein.scoring_functions import ScoringFunctions

# from TEMPy.protein.structure_parser import PDBParser
# from TEMPy.maps.map_parser import MapParser as mp

"""
from TEMPy.maps.em_map import Map
from TEMPy.map_process.map_filters import Filter
from TEMPy.map_process import array_utils
"""
# tracemalloc
# tracemalloc.start()


# @profile
def main(args):
    """
    Run main
    """
    functionType = map_funcs.functionType(2)

    # Set variables from parsed arguments
    ippdb = args.pdbin
    ipmap = args.mapin
    if ippdb is None or ipmap is None:
        raise ArgumentError(
            "Please provide both --pdbin and --mapin to refine model against "
            "a map.\n Program terminated...\n"
        )
        # print(
        #    "Please provide --pdbin and --mapin to refine model against a map.\n"
        #    "Program terminated...\n"
        # )

    nomask = args.nomask
    ipmask = args.maskin
    oppdb = args.pdbout  # sheetbend_pdbout_result.pdb
    res = args.res  # -1.0
    resbycyc = args.res_by_cyc  # None
    ncyc = args.cycle  # 1
    refxyz = args.refxyz  # False
    refuiso = args.refuiso  # False
    postrefxyz = args.postrefxyz  # False
    postrefuiso = args.postrefuiso  # False
    pseudoreg = args.pseudoreg  # no, yes, or postref
    rad = args.rad  # -1.0
    radscl = args.radscl  # 4.0
    xmlout = args.xmlout  # program.xml
    output_intermediate = args.intermediate  # False
    verbose = args.verbose  # 0
    ncycrr = 1  # refine-regularise-cycle
    fltr = 2  # quadratic filter
    hetatom = args.hetatom  # True by default
    hetatm_present = False  # for file writeout in case no hetatm present
    biso_range = args.biso_range
    # ulo = sf_util.b2u(biso_range[0])
    # uhi = sf_util.b2u(biso_range[1])
    # need to make Profile singleton
    timelog = sf_util.Profile()
    # defaults
    # if res <= 0.0:
    #    print('Please specify the resolution of the map!')
    #    exit()

    if not refxyz and not refuiso:
        refxyz = True
        print("Setting --coord to True for coordinates refinement.")

    if resbycyc is None:
        if res > 0.0:
            resbycyc = [res]
        else:
            # need to change this python3 can use , end=''
            raise ArgumentError(
                "Please specify the resolution of the map "
                "or resolution by cycle.\n"
                "Program terminated ...\n"
            )
    elif res > 0.0:
        if res > resbycyc[-1]:
            resbycyc0 = resbycyc
            resbycyc = [resbycyc0[0], res]
    if refuiso or postrefuiso:
        print(f"B-factor bounds : {biso_range[0]} < B < {biso_range[1]}")

    # initialise results list
    results = []
    # read map
    mapin, grid_info = fileio.read_map(ipmap)
    # read model
    structure, hetatm_present = fileio.get_structure(ippdb, verbose)
    # check if map is cubic, make cubic map
    if (
        grid_info.grid_shape[0] != grid_info.grid_shape[1]
        or grid_info.grid_shape[1] != grid_info.grid_shape[2]
    ):
        mapin, grid_info = map_funcs.make_map_cubic(mapin, grid_info)

    # mapin = mp.readMRC(ipmap)
    fullMap = mapin.grid
    # check map box size and reduce if too big?
    apix0 = grid_info.voxel_size
    ori0 = grid_info.grid_start

    # set unit cell from map to structure if (1.0 1.0 1.0)
    if not structure.cell.is_crystal():
        structure.cell = fullMap.unit_cell
    else:
        sf_util.match_model_map_unitcell(structure, fullMap)
    # structure, hetatm_present = sf_util.get_structure(ippdb)
    # structure0 = structure.copy()
    # copy first model
    structure0 = structure.clone()

    # if origin not 0,0,0, offset structure in advance
    # and keep the offset
    if np.any(grid_info.origin):  # not zero
        tr = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*-grid_info.origin))
        structure[0].transform_pos_and_adp(tr)
        grid_info.origin = np.array([0.0, 0.0, 0.0])

    # print(f"mapin dtype :  {mapin.fullMap.dtype}")
    # mapin.write_to_MRC_file('mapin_after_read.map')
    ##scorer = ScoringFunctions()
    # get cell details
    # try use GEMMI UnitCell

    # set gridshape
    # maybe need to do a prime number check on grid dimensions
    # gridshape = sf_util.GridDimension(mapin.fullMap.shape)
    # res = 2/rec_cell[i]/grid_shape[i] for i in (0,1,2)
    # nyq_res = max(res)
    # nyquist_res = maximum([2.0 * apix0[i] for i in (0, 1, 2)])

    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)

    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)  # so that the grid size matches original
    else:
        samp_rate = 1.5  # default; CLIPPER's oversampling parameter

    if pseudoreg != "no":
        print("Setting model for Pseudoregularization")
        ## change here for psedu with gemmi
        preg = pseudoregularizer.Pseudoregularize(structure0)

    # create mask map if non is given at input
    # a mask that envelopes the whole particle/volume of interest
    # if nomask create numpy mask array no mask
    if nomask:
        mmap = ma.make_mask_none(grid_info.grid_shape)
        # ipmask = mmap.copy()  # so it will skip the subsequent logic test for ipmask
    elif ipmask is None:
        timelog.start("MaskMap")
        # struc_map = mapin.copy()
        # struc_map.fullMap = struc_map.fullMap * 0.0
        # if not np.any(grid_info.origin):
        struc_map = calculate_density_with_boxsize(
            structure,
            reso=res,
            rate=samp_rate,
            grid_shape=grid_info.grid_shape,
        )
        # else:
        #    struc_map = calculate_density_with_boxsize(
        #        structure,
        #        reso=res,
        #        rate=samp_rate,
        #        grid_shape=grid_info.grid_shape,
        #        origin=grid_info.origin,
        #    )
        #    grid_info.origin = np.array([0.0, 0.0, 0.0])

        # struc_map = structure.calculate_rho(2.5, mapin)
        mmap = map_funcs.make_mask_from_maps(
            [fullMap, struc_map],
            grid_info,
            res,
            lpfilt_pre=True,
        )
        # tempmap = fullMap + struc_map
        # this is slow
        # fdatar = map_funcs.prepare_mask_filter(apix0)
        # mmap = fftconvolve(tempmap, fdatar, mode="same")
        # filt_data_r, f000 = map_funcs.make_filter_edge_centered(grid_info, functionType.QUADRATIC)
        # fft_obj = sf_util.plan_fft(grid_info, input_dtype=np.float32)
        # ifft_obj = sf_util.plan_ifft(grid_info, input_dtype=np.float32)

        # mmap = map_funcs.fft_convolution_filter(
        #    tempmap,
        #    filt_data_r,
        #    1.0 / f000,
        #    fft_obj,
        #    ifft_obj,
        #    grid_info,
        # )

        # mmapt = map_funcs.calculate_map_threshold(mmap)
        # mmap = ma.masked_less(mmap, mmapt)
        timelog.end("MaskMap")
    else:
        maskin, maskin_grid_info = fileio.read_map(ipmask)
        # maskin = mp.readMRC(ipmask)
        # invert the values for numpy masked_array, true is masked/invalid, false is valid
        mmap = ma.make_mask(np.logical_not(maskin.grid))

    # CASE 1:
    # Refine model against EM data
    print("Refine model against EM map")
    # calculate input map threshold/contour
    mapin_t = map_funcs.calculate_map_threshold(fullMap)
    """zg = np.linspace(0, mapin.z_size(), num=mapin.z_size(),
                    endpoint=False)
    yg = np.linspace(0, mapin.y_size(), num=mapin.y_size(),
                    endpoint=False)
    xg = np.linspace(0, mapin.x_size(), num=mapin.x_size(),
                    endpoint=False)
    """
    for cycrr in range(0, ncycrr):
        if pseudoreg != 'no':
            print("\nRefine-regularise cycle: {0}\n".format(cycrr + 1))
        else:
            print("\nRefine cycle: {0}\n".format(cycrr + 1))
        for cyc in range(0, ncyc):
            shift_vars = []
            shift_u = []
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
                if rcyc <= 8.0 and radscl > 5.0:
                    radscl = 5.0
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
            # print(f"mapin spacing, grid : {mapin.apix}; {mapin.fullMap.shape}")
            # print(f"cell : {cell.a}, {cell.b}, {cell.c}")
            print(f"Calculated spacing : {spacing}")
            downsamp_shape, downsamp_apix = sf_util.calc_best_grid_apix(
                spacing,
                (fullMap.unit_cell.a, fullMap.unit_cell.b, fullMap.unit_cell.c),
            )
            # start = timer()
            # resample runs faster
            print("Resample by box size")
            timelog.start("Resample")

            ### change here
            downsamp_map = map_funcs.resample_data_by_boxsize(
                fullMap,
                downsamp_shape,
            )
            timelog.end("Resample")  #
            # downsamp_map.update_header()
            # need to write?
            downsamp_shape = downsamp_map.shape
            downsamp_apix = map_funcs.calculate_pixel_size(
                fullMap.unit_cell, downsamp_shape
            )
            # downsamp_apix = downsamp_map.apix
            # need to update this for mask if true

            zg = np.linspace(
                # 0, downsamp_map.z_size(), num=downsamp_map.z_size(), endpoint=False
                0,
                downsamp_shape[2],
                num=downsamp_shape[2],
                endpoint=False,
            )
            yg = np.linspace(
                # 0, downsamp_map.y_size(), num=downsamp_map.y_size(), endpoint=False
                0,
                downsamp_shape[1],
                num=downsamp_shape[1],
                endpoint=False,
            )
            xg = np.linspace(
                # 0, downsamp_map.x_size(), num=downsamp_map.x_size(), endpoint=False
                0,
                downsamp_shape[0],
                num=downsamp_shape[0],
                endpoint=False,
            )

            gridshape = cell.GridInfo(
                downsamp_shape,
                grid_info.grid_start,
                downsamp_shape,
                downsamp_apix,
                grid_info.origin,
            )
            # gridshape = sf_util.GridDimension(downsamp_shape)
            timelog.start("fftplan")
            fft_obj = sf_util.plan_fft(gridshape, input_dtype=np.float32)

            timelog.end("fftplan")
            timelog.start("ifftplan")
            ifft_obj = sf_util.plan_ifft(gridshape, input_dtype=np.complex64)
            timelog.end("ifftplan")
            print(f"Downsample shape : {downsamp_shape}")
            print(f"Downsample apix : {downsamp_apix}")
            if verbose >= 1:
                start = timer()
            timelog.start("MapDensity")
            # cmap = downsamp_map.copy()
            # cmap.fullMap = cmap.fullMap * 0
            ####
            # 6 Jun 22
            # something wrong with the grid size with dencalc
            # cmap_grid = dencalc.calculate_density(structure, rcyc, rate=samp_rate)
            # cmap_grid = dencalc.calculate_density(
            #    structure=structure,
            #    reso=rcyc,
            #    rate=(rcyc / (2 * downsamp_apix[0])),
            # )
            # if not np.any(grid_info.origin):
            cmap_grid = calculate_density_with_boxsize(
                structure=structure,
                reso=rcyc,
                rate=samp_rate,
                grid_shape=downsamp_shape,
            )
            # else:
            #    cmap_grid = calculate_density_with_boxsize(
            #        structure=structure,
            #        reso=rcyc,
            #        rate=samp_rate,
            #        grid_shape=downsamp_shape,
            #        origin=grid_info.origin,
            #    )
            if verbose >= 3:
                fileio.write_map_as_MRC(
                    cmap_grid, fullMap.unit_cell, outpath=f'cmapgrid_{cyc+1}.mrc'
                )
                fileio.write_map_as_MRC(
                    downsamp_map, fullMap.unit_cell, outpath=f'downsamp_map_{cyc+1}.mrc'
                )
            # cmap = structure.calculate_rho(2.5, downsamp_map)
            timelog.end("MapDensity")
            # if verbose >= 1:
            #    end = timer()
            #    logger.info("Density calc {0}".format(end - start))
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
            # need to calculate difference map
            print("downsamp_map")
            print(downsamp_map.shape)
            print("cmap")
            print(cmap_grid.shape)

            # print('array dtypes')
            # print(downsamp_map.dtype)
            # print(cmap_grid.dtype)
            scl_map, scl_cmap, dmap = map_funcs.calc_diffmap(
                downsamp_map,
                cmap_grid,
                rcyc,
                rcyc,
                gridshape,
                gridshape,
                lpfilt_pre=True,
                lpfilt_post=False,
                refscl=False,
                randsize=0.1,
                flag_dust=False,
                verbose=verbose,
                fft_obj=fft_obj,
                ifft_obj=ifft_obj,
            )
            if nomask:
                mmap = ma.make_mask_none(downsamp_shape)
            else:
                mmap = map_funcs.make_mask_from_maps(
                    [scl_map, scl_cmap],
                    gridshape,
                    res,
                    lpfilt_pre=True,
                )
            # print(scl_map.dtype)
            # print(scl_cmap.dtype)
            # print(dmap.dtype)

            # scl_map, scl_cmap, dmap = DFM.get_diffmap12(
            #    downsamp_map,
            #    cmap,
            #    rcyc,
            #    rcyc,
            #    cyc=cyc + 1,
            #    lpfiltb=False,
            #    flag_dust=False,
            #    refsc=False,
            #    verbose=verbose,
            # )

            timelog.end("DiffMap")
            if verbose >= 1:
                end = timer()
                print("Diff map calc: {0} s ".format(end - start))

            if verbose >= 3:
                fileio.write_map_as_MRC(
                    dmap, fullMap.unit_cell, outpath=f'dmap_{cyc+1}.mrc'
                )
                fileio.write_map_as_MRC(
                    scl_map, fullMap.unit_cell, outpath=f'scl_map_{cyc+1}.mrc'
                )
                fileio.write_map_as_MRC(
                    scl_cmap, fullMap.unit_cell, outpath=f'scl_cmap_{cyc+1}.mrc'
                )

            # if verbose >= 0:
            #    logger.info("check apix")
            #    logger.info(mapin.__class__.__name__)
            #    logger.info(mapin.apix)
            #    logger.info(scl_map.__class__.__name__)
            #    logger.info(scl_map.apix)
            #    logger.info(scl_cmap.__class__.__name__)
            #    logger.info(scl_cmap.apix)
            # if verbose >= 3:
            #    testoutmap = dmap.copy()
            #    DFM.write_mapfile(testoutmap, f"test_diffmap{cyc+1}.map")
            # calculate fsc and envelope score(TEMPy) instead of R and R-free
            # use envelop score (TEMPy)... calculating average fsc
            # like refmac might take too much?
            if verbose >= 1:
                start = timer()
            timelog.start("Scoring")
            # calculate map contour
            # dmapin_t = map_funcs.calculate_map_threshold(scl_map)
            mapcurreso_t = map_funcs.calculate_map_threshold(scl_map)
            # print("Calculated input map volume threshold is ", end="")
            # m = "Calculated input map volume threshold is "
            # m += "{0:.2f} and {1:.2f} (current resolution).\n".format(
            #    dmapin_t, mapcurreso_t
            # )
            # calculate model contour
            t = 2.5 if rcyc > 10.0 else 2.0 if rcyc > 6.0 else 1.5
            # cmap_t = t * cmap_grid.std()
            # fltrcmap_t = t*scl_cmap.std() #np.std(scl_cmap.fullMap)
            # fltrcmap_t = t * np.nanstd(scl_cmap)
            fltrcmap_t = map_funcs.calculate_map_threshold(scl_cmap)
            # print("Calculated model threshold is ", end="")
            m = "Calculated model threshold is "
            m += "{0:.2f} and {1:.2f} (current resolution)\n".format(
                mapcurreso_t,
                fltrcmap_t,
            )
            print(m)

            # ovl_map1, ovl_mdl1 = map_funcs.calculate_overlap_scores(
            #    scl_map, scl_cmap, dmapin_t, cmap_t
            # )
            ovl_map, ovl_mdl = map_funcs.calculate_overlap_scores(
                scl_map, scl_cmap, mapcurreso_t, fltrcmap_t
            )
            timelog.end("Scoring")
            if verbose >= 1:
                end = timer()
                # logger.info("Score mod: {0} s".format(end - start))
                print("Score mod: {0} s".format(end - start))
            # print("Fraction of map overlapping with model: ", end="")
            m = "TEMPy scores :\n"
            m += " Fraction of map overlapping with model: "
            m += "{0:.3f} \n".format(ovl_map)
            # print("Fraction of model overlapping with map: ", end="")
            m += " Fraction of model overlapping with map: "
            m += "{0:.3f}\n".format(ovl_mdl)
            # logger.info(m)
            print(m)
            if refxyz:
                # logger.info("REFINE XYZ")
                print("REFINE XYZ")
                timelog.start("Shiftfield")
                # print(f'maps dtype, lpcmap : {scl_cmap.fullMap.dtype}')
                # print(f'mmap dtype : {mmap.fullMap.dtype}')
                # logger.info(f"dmap dtype : {dmap.fullMap.dtype}")
                # print(f"dmap dtype : {dmap.fullMap.dtype}")
                x1m, x2m, x3m = shift_field_coord(
                    scl_cmap,  # .fullMap,
                    dmap,  # .fullMap,
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
                timelog.end("Shiftfield")
                # Read pdb and update
                # use linear interpolation instead of cubic
                # size of x,y,z for x1map=x2map=x3map
                # print(x1map.fullMap.box_size())
                timelog.start("Interpolate")
                interp_x1 = RegularGridInterpolator((xg, yg, zg), x1m)
                interp_x2 = RegularGridInterpolator((xg, yg, zg), x2m)
                interp_x3 = RegularGridInterpolator((xg, yg, zg), x3m)
                count = 0
                v = map_funcs.map_grid_position_array(gridshape, structure)
                # print(xg, yg, zg)
                # v = np.flip(v, 1)
                # scaling for shifts use 1 (em) instead of 2 (xtal)
                dx = (1.0 * interp_x1(v)) * fullMap.unit_cell.a
                dy = (1.0 * interp_x2(v)) * fullMap.unit_cell.b
                dz = (1.0 * interp_x3(v)) * fullMap.unit_cell.c
                timelog.end("Interpolate")
                timelog.start("UpdateModel")

                index = 0
                for cra in structure[0].all():
                    cra.atom.pos += gemmi.Position(dx[index], dy[index], dz[index])
                # for i in range(len(structure)):
                #    if verbose >= 2:
                #        shift_vars.append(
                #            [
                #                structure.atomList[i].get_name(),
                #                structure.atomList[i].get_x(),
                #                structure.atomList[i].get_y(),
                #                structure.atomList[i].get_z(),
                #                dx[i],
                #                dy[i],
                #                dz[i],
                #            ]
                #        )  # du[i], dv[i], dw[i],
                #    structure.atomList[i].translate(dx[i], dy[i], dz[i])
                timelog.end("UpdateModel")
            if pseudoreg == "yes":
                # logger.info("PSEUDOREGULARIZE")
                timelog.start("INTPSEUDOREG")
                structure = preg.regularize_frag(structure)
                timelog.end("INTPSEUDOREG")
            # U-isotropic refinement
            if refuiso or (postrefuiso and lastcyc):
                # logger.info("REFINE U ISO")
                timelog.start("UISO")
                x1m = shift_field_uiso(
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
                timelog.end("UISO")
                timelog.start("Interpolate")
                interp_x1 = RegularGridInterpolator((xg, yg, zg), x1m)
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
                    if verbose >= 2:
                        shift_u.append([structure.atomList[i].temp_fac, du[i], db[i]])
                timelog.end("UpdateModel")
            temp_result = sf_util.ResultsByCycle(
                cycrr,
                cyc,
                rcyc,
                radcyc,
                ovl_map,
                ovl_mdl,
                0.0,
            )
            results.append(temp_result)
            if output_intermediate:
                outname = "{0}_{1}.pdb".format(oppdb.strip(".pdb"), cyc + 1)
                # structure.write_to_PDB(outname, hetatom=hetatm_present)
                structure.write_minimal_pdb(f'{outname}')
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
            if len(shift_u) != 0 and verbose >= 2:
                outusiocsv = "shiftuiso_u2b_{0}.csv".format(cyc + 1)
                fuiso = open(outusiocsv, "w")
                for j in range(0, len(shift_u)):
                    fuiso.write("{0}, {1}\n".format(j, shift_u[j]))
                fuiso.close()
            # end of cycle loop
        if pseudoreg == "postref":
            # logger.info("PSEUDOREGULARIZE")
            timelog.start("PSEUDOREG")
            structure = preg.regularize_frag(structure)
            timelog.end("PSEUDOREG")
            timelog.start("MapDensity")
            # cmap = mapin.copy()
            # cmap.fullMap = cmap.fullMap * 0
            cmap = structure.calculate_rho(2.5, mapin)
            # cmap = emc.calc_map_density(mapin, structure)
            timelog.start("MapDensity")
            timelog.start("Scoring")
            cmap_t = 1.5 * cmap.std()
            ##ovl_mapf, ovl_mdlf = scorer.calculate_overlap_scores(
            ##    mapin, cmap, mapin_t, cmap_t
            ##)
            timelog.end("Scoring")
            # logger.info(f"End of refine-regularise cycle {cycrr+1}")
            # logger.info("TEMPys scores :")
            # logger.info(
            #    "Fraction of map overlapping with model: {0:.3f}".format(ovl_mapf)
            # )
            # logger.info(
            #    "Fraction of model overlapping with map: {0:.3f}".format(ovl_mdlf)
            # )
        # end of psedo reg loop
    # write final pdb
    if oppdb is not None:
        outfname = "{0}_sheetbendfinal.pdb".format(oppdb.strip(".pdb"))
        # structure.write_to_PDB(f"{outfname}", hetatom=hetatm_present)  # preg.got_hetatm
        structure.write_minimal_pdb(f'{outfname}')
    else:
        outfname = "{0}_sheetbendfinal.pdb".format(ippdb.strip(".pdb"))
        # structure.write_to_PDB(f"{outfname}", hetatom=hetatm_present)
        structure.write_minimal_pdb(f'{outfname}')

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

    # logger.info(f"Ended at {datetime.datetime.now()}")
    if verbose >= 2:
        timelog.profile_log()


if __name__ == "__main__":
    # parser = SheetbendParser()
    # parser.get_args()
    # main(parser.args)
    import sys
    from sheetbend_cmdln_parser import SheetbendParser

    sb_parser = SheetbendParser()  # args=args)
    sb_parser.get_args(sys.argv[1:])
    main(sb_parser.args)
