"""
Python implementation of csheetbend to perform shift field refinement
Copyright 2018 Kevin Cowtan & University of York all rights reserved
Author: Soon Wen Hoh, University of York 2020

License: ?
"""

from __future__ import print_function  # python 3 proof
from argparse import ArgumentError
import sys
import datetime
from timeit import default_timer as timer
from scipy.interpolate import RegularGridInterpolator

from scipy.signal import fftconvolve, resample

import gemmi
import numpy as np
import numpy.ma as ma
from pysheetbend.shiftfield import shift_field_coord
import pysheetbend.shiftfield_util as sf_util
import pysheetbend.pseudoregularizer_gemmi as pseudoregularizer
from pysheetbend.utils import fileio, map_funcs, cell
import pandas as pd

# from pysheetbend.utils.dencalc import calculate_density, calculate_density_with_boxsize

# from memory_profiler import profile

# refine map to map

"""
from TEMPy.maps.em_map import Map
from TEMPy.map_process.map_filters import Filter
from TEMPy.map_process import array_utils
"""
# tracemalloc
# tracemalloc.start()


# @profile
def main(args):
    '''
    Run main refine map to map
    '''
    functionType = map_funcs.functionType(2)

    ipmap = args.mapin
    tgt_map = args.mapin2
    if ipmap is None or tgt_map is None:
        raise ArgumentError(
            'Please provide --mapin and --mapin2 to refine map (--mapin2) '
            'against a reference map (--mapin).\n'
            'Program terminated...\n'
        )

    nomask = args.nomask
    ipmask = args.maskin
    mapout = args.mapout
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
        print('Setting --coord to True for coordinates refinement.')

    if resbycyc is None:
        if res > 0.0:
            resbycyc = [res]
        else:
            # need to change this python3 can use , end=''
            raise ArgumentError(
                'Please specify the resolution of the map or resolution by cycle.\n'
                'Program terminated ...\n'
            )
            sys.exit()
    elif res > 0.0:
        if res > resbycyc[-1]:
            resbycyc0 = resbycyc
            resbycyc = [resbycyc0[0], res]
    if refuiso or postrefuiso:
        print(f"B-factor bounds : {biso_range[0]} < B < {biso_range[1]}")
    if len(resbycyc) == 0:
        resbycyc.append(res)

    # initialise results list
    results = []
    # read maps
    mapin, grid_info = fileio.read_map(ipmap)
    mapin_tgt, grid_info_tgt = fileio.read_map(tgt_map)

    # check if map is cubic, make cubic map, add padding with
    if (
        grid_info.grid_shape[0] != grid_info.grid_shape[1]
        or grid_info.grid_shape[1] != grid_info.grid_shape[2]
    ):
        mapin, grid_info = map_funcs.make_map_cubic(mapin, grid_info)

    if (
        grid_info_tgt.grid_shape[0] != grid_info_tgt.grid_shape[1]
        or grid_info_tgt.grid_shape[1] != grid_info_tgt.grid_shape[2]
    ):
        mapin_tgt, grid_info_tgt = map_funcs.make_map_cubic(mapin_tgt, grid_info_tgt)

    apix0 = grid_info.voxel_size
    ori0 = grid_info.grid_start
    apix0_tgt = grid_info_tgt.voxel_size
    ori0_tgt = grid_info_tgt.grid_start

    # get cell details
    # try use GEMMI UnitCell
    fullMap = mapin.grid

    # set gridshape
    # maybe need to do a prime number check on grid dimensions
    # gridshape = sf_util.GridDimension(mapin.fullMap.shape)
    # res = 2/rec_cell[i]/grid_shape[i] for i in (0,1,2)
    # nyq_res = max(res)
    # nyquist_res = maximum([2.0 * apix0[i] for i in (0, 1, 2)])

    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)

    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)  # so that grid size matches original
    else:
        samp_rate = 1.5  # default same value from CLIPPER

    # create mask map if non is given at input
    # a mask that envelopes the whole particle/volume of interest
    # if nomask:
    #    mmap = ma.make_mask_none(mapin.fullMap.shape)
    #    # ipmask = mmap.copy()  # so it will skip the subsequent logic test for ipmask
    if ipmask is not None:
        maskin, maskin_grid_info = fileio.read_map(ipmask)

    # CASE 2:
    # Refine map against another map; EM data
    # 1. Read target map and reference EM map
    # 2. Before first cycle, make shift maps
    # 3. In first cycle, make shifted maps using shifts
    # 4. Run shiftfield, accumulate shifts on shifted maps
    #    a. make shifted,du,dv,dwmaps, based on original grid
    #    b. at each resolution run, loop through shifted mapgridpoints
    #       convert mapgridpoints to coord_frac then -= coord_frac of du,dv,dw[mapgridpoints]
    #       map_shifted[mapgridpoint] = mapin.interpolate(coord_frac)
    # truncate mapin to current cycle resolution, calculate difference map
    #    c. make x1,x2,x3maps with grid from current resolution
    #    d. cmap is calculated from the accumulated map_shifted
    #    e. loop through dumap mapgridpoints, convert to coord_frac
    #    f. x1,x2,x3map interpolate(coord_frac) then add to du,dv,dwmaps
    #
    # 5. At end of cycle, update shift maps
    #    a. loop through map_shifted mapgridpoints, convert to coord_frac
    #    b. coord_frac subtract coord_frac of du,dv,dwmap(mapgridpoints)
    #    c. assign mapin interpolate(coord_frac) to mapshifted[mapgridpoints]
    # 6. Calculate final map after cycle loop
    # 7. Apply shifts
    # Calculate input map threshold/contour

    print("Refine EM map against EM map")
    # calculate input map threshold/contour
    mapin_t = map_funcs.calculate_map_threshold(fullMap)
    # make shift maps
    map_shifted = np.zeros(grid_info_tgt.grid_shape, dtype=np.float32)
    dumap = np.zeros(grid_info_tgt.grid_shape, dtype=np.float32)
    dvmap = np.zeros(grid_info_tgt.grid_shape, dtype=np.float32)
    dwmap = np.zeros(grid_info_tgt.grid_shape, dtype=np.float32)
    nz0 = np.linspace(0, 1, num=grid_info_tgt.grid_shape[2], endpoint=False)
    ny0 = np.linspace(0, 1, num=grid_info_tgt.grid_shape[1], endpoint=False)
    nx0 = np.linspace(0, 1, num=grid_info_tgt.grid_shape[0], endpoint=False)
    mapintgt_interp = RegularGridInterpolator(
        (nx0, ny0, nz0),
        np.array(mapin_tgt.grid, copy=False),
        bounds_error=False,
        fill_value=0.0,
    )
    print(nx0[0], ny0[0], nz0[0], nx0[-1], ny0[-1], nz0[-1])
    coord_frac, im = map_funcs.grid_coord_to_frac(map_shifted, grid_info_tgt)
    xi, yi, zi = zip(*im)

    # target map is mapin_tgt, reference map is mapin
    tgtmapin_t = map_funcs.calculate_map_threshold(mapin_tgt.grid)
    ovl_map1, ovl_map2 = map_funcs.calculate_overlap_scores(
        mapin.grid, mapin_tgt.grid, mapin_t, tgtmapin_t
    )
    print(f'Calculated map thresholds are {mapin_t} (ref) and {tgtmapin_t} (target).')
    print(f'Initial target to reference map overlap scores {ovl_map2}.')

    # setup RegularGridInterpolator

    """zg = np.linspace(0, mapin.z_size(), num=mapin.z_size(),
                    endpoint=False)
    yg = np.linspace(0, mapin.y_size(), num=mapin.y_size(),
                    endpoint=False)
    xg = np.linspace(0, mapin.x_size(), num=mapin.x_size(),
                    endpoint=False)
    """
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
        print(f"Calculated spacing : {spacing}")
        downsamp_shape, spacing = sf_util.calc_best_grid_apix(
            spacing,
            (fullMap.unit_cell.a, fullMap.unit_cell.b, fullMap.unit_cell.c),
        )
        # resampled_map = #
        timelog.start('Resample')
        downsamp_map = map_funcs.resample_data_by_boxsize(fullMap, downsamp_shape)
        # downsamp_map.update_header()
        timelog.end('Resample')
        downsamp_shape = downsamp_map.shape
        downsamp_apix = map_funcs.calculate_pixel_size(
            fullMap.unit_cell,
            downsamp_shape,
        )
        gridshape = cell.GridInfo(
            downsamp_shape,
            grid_info.grid_start,
            downsamp_shape,
            downsamp_apix,
            grid_info.origin,
        )

        # zg = np.linspace(0, 1, num=downsamp_shape[2], endpoint=False)
        # yg = np.linspace(0, 1, num=downsamp_shape[1], endpoint=False)
        # xg = np.linspace(0, 1, num=downsamp_shape[0], endpoint=False)
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
        # cmap is calculated from the shiftsmaps
        # get frac coords from the grid index positions
        # cf -= dwmap[zyx_pos], dvmap dumap
        # mapshifted[zyx_pos] = map_in.interp(cf)
        # problems with interpolating coordinates out of bounds 0 - 0.99 not 1 here therefore out of bounds
        # find a way to make the grids extends to 1?
        coord_frac1 = (
            coord_frac
            - np.array([dumap[xi, yi, zi], dvmap[xi, yi, zi], dwmap[xi, yi, zi]]).T
        )
        coord_frac1 = map_funcs.cor_mod_array(coord_frac1, [1.0, 1.0, 1.0])
        # if cyc > 0:
        #    pd.DataFrame(coord_frac1).to_csv(
        #        f'test_coord_frac1_{cyc}.csv', header=None, index=None
        #    )
        # for i in range(len(coord_frac1)):
        #    if (coord_frac1[i] > nx0[-1]).any() or (coord_frac1[i] < nx0[0]).any():
        #        print(i, coord_frac1[i])
        shifts_arr = mapintgt_interp(coord_frac1)
        map_shifted[xi, yi, zi] = shifts_arr
        cmap = map_funcs.resample_data_by_boxsize(map_shifted, downsamp_shape)
        # cmap = downsamp_map.copy()
        # cmap.fullMap = cmap.fullMap * 0
        # cmap = structure.calculate_rho(2.5, downsamp_map)
        timelog.end("MapDensity")
        # if verbose >= 1:
        #    end = timer()
        #    print("Density calc {0}".format(end - start))
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
        scl_map, scl_cmap, dmap = map_funcs.calc_diffmap(
            downsamp_map,
            cmap,
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
        timelog.end("DiffMap")
        # mmap = ma.make_mask_none(downsamp_shape)
        if nomask:
            mmap = ma.make_mask_none(downsamp_shape)
        elif ipmask is None:
            downsamp_mask = map_funcs.make_mask_from_maps(
                [scl_map, scl_cmap],
                gridshape,
                res,
                lpfilt_pre=True,
                radcyc=radcyc,
            )
            mmap = ma.make_mask(np.logical_not(downsamp_mask))
        else:
            downsamp_mask = map_funcs.resample_data_by_boxsize(
                maskin.grid,
                downsamp_shape,
            )
            filt_data_r, f000 = map_funcs.make_filter_edge_centered(
                gridshape, filter_radius=radcyc, function=functionType
            )
            downsamp_mask = map_funcs.fft_convolution_filter(
                downsamp_mask,
                filt_data_r,
                1.0 / f000,
                fft_obj,
                ifft_obj,
                grid_info,
            )
            mmap = ma.make_mask(np.logical_not(downsamp_mask))

        if verbose >= 1:
            end = timer()
            print("Diff map calc: {0} s ".format(end - start))
        # if verbose >= 0:
        #    print("check apix")
        #    print(mapin.__class__.__name__)
        #    print(mapin.apix)
        #    print(scl_map.__class__.__name__)
        #    print(scl_map.apix)
        #    print(scl_cmap.__class__.__name__)
        #    print(scl_cmap.apix)
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
        # dmapin_t = map_funcs.calculate_map_threshold(downsamp_map)
        mapcurreso_t = map_funcs.calculate_map_threshold(scl_map)
        # print("Calculated input map volume threshold is ", end="")
        # m = "Calculated input map volume threshold is "
        # m += "{0:.2f} and {1:.2f} (current resolution).\n".format(
        # ]    dmapin_t, mapcurreso_t
        # )
        # calculate model contour
        # t = 2.5 if rcyc > 10.0 else 2.0 if rcyc > 6.0 else 1.5
        # cmap_t = 1.5 * cmap.std()
        # fltrcmap_t = t*scl_cmap.std() #np.std(scl_cmap.fullMap)
        fltrcmap_t = map_funcs.calculate_map_threshold(scl_cmap)
        # print("Calculated model threshold is ", end="")
        m = "Calculated model threshold is "
        m += "{0:.2f} and {1:.2f} (current resolution)\n".format(
            mapcurreso_t,
            fltrcmap_t,
        )
        print(m)
        # ovl_map1, ovl_mdl1 = map_funcs.calculate_overlap_scores(
        #    downsamp_map, cmap, dmapin_t, cmap_t
        # )
        ovl_map, ovl_mdl = map_funcs.calculate_overlap_scores(
            scl_map, scl_cmap, mapcurreso_t, fltrcmap_t
        )
        timelog.end("Scoring")
        if verbose >= 1:
            end = timer()
            print("Score mod: {0} s".format(end - start))
        # print("Fraction of map overlapping with model: ", end="")
        m = "TEMPy scores :\n"
        m += " Fraction of map overlapping with model: "
        m += "{0:.3f} \n".format(ovl_map)
        # print("Fraction of model overlapping with map: ", end="")
        m += " Fraction of model overlapping with map: "
        m += "{0:.3f} \n".format(ovl_mdl)
        print(m)
        if refxyz:
            print("REFINE XYZ")
            timelog.start("Shiftfield")
            # print(f'maps dtype, lpcmap : {scl_cmap.fullMap.dtype}')
            # print(f'mmap dtype : {mmap.fullMap.dtype}')
            print(f"dmap dtype : {dmap.dtype}")
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
            x1m_resample = x1m.copy()
            x2m_resample = x2m.copy()
            x3m_resample = x3m.copy()
            x1m_resample = resample(x1m_resample, grid_info.grid_shape[0], axis=0)
            x1m_resample = resample(x1m_resample, grid_info.grid_shape[1], axis=1)
            x1m_resample = resample(x1m_resample, grid_info.grid_shape[2], axis=2)
            x2m_resample = resample(x2m_resample, grid_info.grid_shape[0], axis=0)
            x2m_resample = resample(x2m_resample, grid_info.grid_shape[1], axis=1)
            x2m_resample = resample(x2m_resample, grid_info.grid_shape[2], axis=2)
            x3m_resample = resample(x3m_resample, grid_info.grid_shape[0], axis=0)
            x3m_resample = resample(x3m_resample, grid_info.grid_shape[1], axis=1)
            x3m_resample = resample(x3m_resample, grid_info.grid_shape[2], axis=2)
            # interp_x1 = RegularGridInterpolator((zg, yg, xg), x1m)
            # interp_x2 = RegularGridInterpolator((zg, yg, xg), x2m)
            # interp_x3 = RegularGridInterpolator((zg, yg, xg), x3m)
            # count = 0
            # v = structure.map_grid_position_array(scl_map, False)
            # v = np.flip(v, 1)
            # scaling for shifts use 1 (em) instead of 2 (xtal)
            ## something wrong here in interpolation out of bounds??
            dumap += x1m_resample  # 1.0 * interp_x1(coord_frac)
            dvmap += x2m_resample  # 1.0 * interp_x2(coord_frac)
            dwmap += x3m_resample  # 1.0 * interp_x3(coord_frac)
            # dx = (1.0 * interp_x1(v)) * cell.a
            # dy = (1.0 * interp_x2(v)) * cell.b
            # dz = (1.0 * interp_x3(v)) * cell.c
            timelog.end("Interpolate")
        temp_result = sf_util.ResultsByCycle(
            0,
            cyc,
            rcyc,
            radcyc,
            ovl_map1,
            ovl_map2,
            0.0,
        )
        results.append(temp_result)
        # if output_intermediate:
        sys.stdout.flush()
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #    print(stat)
        # end of cycle loop

    # calculate final map
    coord_frac -= np.array([dumap[xi, yi, zi], dvmap[xi, yi, zi], dwmap[xi, yi, zi]]).T
    shifts_arr = mapintgt_interp(coord_frac)
    map_shifted[xi, yi, zi] = shifts_arr[:]
    # du, dv, dw maps
    dumap = dumap * fullMap.unit_cell.parameters[0]
    dvmap = dvmap * fullMap.unit_cell.parameters[1]
    dwmap = dwmap * fullMap.unit_cell.parameters[2]
    print(dumap.dtype)
    print(type(dumap))
    fileio.write_map_as_MRC(
        dumap,
        fullMap.unit_cell,
        outpath='dumap_shiftfield.mrc',
    )
    fileio.write_map_as_MRC(
        dvmap,
        fullMap.unit_cell,
        outpath='dvmap_shiftfield.mrc',
    )
    fileio.write_map_as_MRC(
        dwmap,
        fullMap.unit_cell,
        outpath='dwmap_shiftfield.mrc',
    )

    # write final map
    fileio.write_map_as_MRC(
        map_shifted,
        fullMap.unit_cell,
        outpath=mapout,
    )

    # map_shifted.write_to_MRC_file(mapout)
    # print(map_shifted)
    # write xml results
    if xmlout is not None:
        f = open(xmlout, "w")
        for i in range(0, len(results)):
            if i == 0:
                results[i].write_xml_results_start(f, mapout, tgt_map)
            results[i].write_xml_results_cyc(f)
            if i == len(results) - 1:
                results[i].write_xml_results_end(f)
        f.close()

    print(f"Ended at {datetime.datetime.now()}")
    if verbose >= 2:
        timelog.profile_log()


if __name__ == "__main__":
    # parser = SheetbendParser()
    # parser.get_args()
    # main(parser.args)
    import sys
    from sheetbend_cmdln_parser import SheetbendParser

    sb_parser = SheetbendParser()
    sb_parser.get_args(sys.argv[1:])
    main(sb_parser.args)
