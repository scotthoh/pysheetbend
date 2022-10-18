"""
Python implementation of csheetbend to perform shift field refinement
Copyright 2018 Kevin Cowtan & University of York all rights reserved
Author: Soon Wen Hoh, University of York 2020

License: ?
"""

from __future__ import print_function  # python 3 proof
import sys
import datetime
from timeit import default_timer as timer
import logging
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve, resample

#

from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.MapParser import MapParser as mp
import numpy as np
import numpy.ma as ma
import pysheetbend.shiftfield
import pysheetbend.shiftfield_util as sf_util
import pysheetbend.pseudoregularizer
import pysheetbend.map_scaling as DFM
from pysheetbend.sheetbend_cmdln_parser import SheetbendParser
from memory_profiler import profile

# refine map to map

"""
from TEMPy.maps.em_map import Map
from TEMPy.map_process.map_filters import Filter
from TEMPy.map_process import array_utils
"""
# tracemalloc
# tracemalloc.start()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("pysheetbend.log")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


@profile
def main():
    """
    Run main
    """
    # Setup logger file and console
    # print(f"Started at {datetime.datetime.now()}")
    logger.info(f"Started at {datetime.datetime.now()}")
    # Parse command line input
    sb_parser = SheetbendParser()  # args=args)
    sb_parser.get_args()
    # sb_parser.print_args()
    logger.info(sb_parser.show_args())
    # Set variables from parsed arguments
    # ippdb = sb_parser.args.pdbin
    ipmap = sb_parser.args.mapin
    if ipmap is None:
        logger.error(
            "Please provide --mapin and --mapin2 to refine map(--mapin2) "
            "against a reference map(--mapin).\n"
            "Program terminated...\n"
        )
        sys.exit()

    nomask = sb_parser.args.nomask
    ipmask = sb_parser.args.maskin
    tgt_map = sb_parser.args.mapin2
    mapout = sb_parser.args.mapout
    res = sb_parser.args.res  # -1.0
    resbycyc = sb_parser.args.res_by_cyc  # None
    ncyc = sb_parser.args.cycle  # 1
    refxyz = sb_parser.args.refxyz  # False
    refuiso = sb_parser.args.refuiso  # False
    postrefxyz = sb_parser.args.postrefxyz  # False
    postrefuiso = sb_parser.args.postrefuiso  # False
    pseudoreg = sb_parser.args.pseudoreg  # no, yes, or postref
    rad = sb_parser.args.rad  # -1.0
    radscl = sb_parser.args.radscl  # 4.0
    xmlout = sb_parser.args.xmlout  # program.xml
    output_intermediate = sb_parser.args.intermediate  # False
    verbose = sb_parser.args.verbose  # 0
    ncycrr = 1  # refine-regularise-cycle
    fltr = 2  # quadratic filter
    hetatom = sb_parser.args.hetatom  # True by default
    hetatm_present = False  # for file writeout in case no hetatm present
    biso_range = sb_parser.args.biso_range
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
        # print("Setting --coord to True for coordinates refinement.")
        logger.info("Setting --coord to True for coordinates refinement.")
    if resbycyc is None:
        if res > 0.0:
            resbycyc = [res]
        else:
            # need to change this python3 can use , end=''
            logger.error(
                "Please specify the resolution of the map or resolution by cycle.\n"
                "Program terminated ...\n"
            )
            sys.exit()
    elif res > 0.0:
        if res > resbycyc[-1]:
            resbycyc0 = resbycyc
            resbycyc = [resbycyc0[0], res]
    if refuiso or postrefuiso:
        logger.info(f"B-factor bounds : {biso_range[0]} < B < {biso_range[1]}")
    if len(resbycyc) == 0:
        resbycyc.append(res)

    # initialise results list
    results = []
    # read model

    # read map
    mapin = mp.readMRC(ipmap)
    mapin_tgt = mp.readMRC(tgt_map)
    apix0 = mapin.apix  # tempy apix should be numpy array
    ori0 = mapin.origin  # same as above
    # print(f"mapin dtype : {mapin.fullMap.dtype}")
    # mapin.write_to_MRC_file('mapin_after_read.map')
    scorer = ScoringFunctions()
    # get cell details
    # try use GEMMI UnitCell
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
    gridshape0 = sf_util.GridDimension(mapin.fullMap.shape)
    print(gridshape0.grid_sam)
    print(gridshape0.g_reci)
    print(gridshape0.g_real)
    print(gridshape0.g_half)
    # res = 2/rec_cell[i]/grid_shape[i] for i in (0,1,2)
    # nyq_res = max(res)
    # nyquist_res = maximum([2.0 * apix0[i] for i in (0, 1, 2)])
    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)

    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)
    else:
        samp_rate = 1.5  # default same value from CLIPPER

    # create mask map if non is given at input
    # a mask that envelopes the whole particle/volume of interest
    if nomask:
        mmap = ma.make_mask_none(mapin.fullMap.shape)
        # ipmask = mmap.copy()  # so it will skip the subsequent logic test for ipmask
    if ipmask:
        maskin = mp.readMRC(ipmask)
        # invert the values for numpy masked_array, true is masked/invalid, false is valid
        mmap = ma.make_mask(np.logical_not(maskin))

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

    logger.info("Refine EM map against EM map")
    # calculate input map threshold/contour
    mapin_t = scorer.calculate_map_threshold(mapin)
    # make shift maps
    map_shifted = mapin_tgt.copy()
    map_shifted.fullMap = map_shifted.fullMap * 0
    dumap = map_shifted.copy()
    dumap.fullMap = dumap.fullMap * 0
    dvmap = map_shifted.copy()
    dvmap.fullMap = dvmap.fullMap * 0
    dwmap = map_shifted.copy()
    dwmap.fullMap = dvmap.fullMap * 0
    nz0 = np.linspace(0, 1, num=mapin_tgt.z_size(), endpoint=False)
    ny0 = np.linspace(0, 1, num=mapin_tgt.y_size(), endpoint=False)
    nx0 = np.linspace(0, 1, num=mapin_tgt.x_size(), endpoint=False)
    mapintgt_interp = RegularGridInterpolator((nz0, ny0, nx0), mapin_tgt.fullMap)
    coord_frac, im = sf_util.grid_coord_to_frac(map_shifted)
    zi, yi, xi = zip(*im)

    # target map is mapin_tgt, reference map is mapin
    tgtmapin_t = scorer.calculate_map_threshold(mapin_tgt)
    ovl_map1, ovl_map2 = scorer.calculate_overlap_scores(
        mapin, mapin_tgt, mapin_t, tgtmapin_t
    )
    logger.info(
        "Calculated map thresholds are {mapin_t} (ref) and {tgtmapin_t} (target)."
    )
    logger.info("Initial target to reference map overlap scores {ovl_map2}.")

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
            radcyc = radscl * rcyc
        logger.info(
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
        logger.info(f"Calculated spacing : {spacing}")
        downsamp_shape, spacing = sf_util.calc_best_grid_apix(
            spacing, (cell.a, cell.b, cell.c)
        )
        # resampled_map = #
        downsamp_map = mapin.resample_by_box_size(downsamp_shape)
        # downsamp_map.update_header()
        downsamp_shape = downsamp_map.fullMap.shape
        downsamp_apix = downsamp_map.apix
        mmap = ma.make_mask_none(downsamp_shape)
        zg = np.linspace(0, 1, num=downsamp_map.z_size(), endpoint=False)
        yg = np.linspace(0, 1, num=downsamp_map.y_size(), endpoint=False)
        xg = np.linspace(0, 1, num=downsamp_map.x_size(), endpoint=False)
        gridshape = sf_util.GridDimension(downsamp_shape)
        timelog.start("fftplan")
        fft_obj = sf_util.plan_fft(gridshape)
        timelog.end("fftplan")
        timelog.start("ifftplan")
        ifft_obj = sf_util.plan_ifft(gridshape)
        timelog.end("ifftplan")
        logger.info(f"Downsample shape : {downsamp_shape}")
        logger.info(f"Downsample apix : {downsamp_apix}")
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
            - np.array([dwmap[zi, yi, xi], dvmap[zi, yi, xi], dumap[zi, yi, xi]]).T
        )
        coord_frac1 = shiftfield.cor_mod1(coord_frac1, [1.0, 1.0, 1.0])
        if cyc > 0:
            for i in coord_frac1:
                print(i)

        shifts_arr = mapintgt_interp(coord_frac1)
        map_shifted.fullMap[zi, yi, xi] = shifts_arr
        cmap = map_shifted.resample_by_box_size(downsamp_shape)
        # cmap = downsamp_map.copy()
        # cmap.fullMap = cmap.fullMap * 0
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
            logger.info("Diff map calc: {0} s ".format(end - start))
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
        dmapin_t = scorer.calculate_map_threshold(downsamp_map)
        mapcurreso_t = scorer.calculate_map_threshold(scl_map)
        # print("Calculated input map volume threshold is ", end="")
        m = "Calculated input map volume threshold is "
        m += "{0:.2f} and {1:.2f} (current resolution).\n".format(
            dmapin_t, mapcurreso_t
        )
        # calculate model contour
        # t = 2.5 if rcyc > 10.0 else 2.0 if rcyc > 6.0 else 1.5
        cmap_t = 1.5 * cmap.std()
        # fltrcmap_t = t*scl_cmap.std() #np.std(scl_cmap.fullMap)
        fltrcmap_t = scorer.calculate_map_threshold(scl_cmap)
        # print("Calculated model threshold is ", end="")
        m += "Calculated model threshold is "
        m += "{0:.2f} and {1:.2f} (current resolution)\n".format(cmap_t, fltrcmap_t)
        logger.info(m)
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
        # print("Fraction of map overlapping with model: ", end="")
        m = "TEMPy scores :\n"
        m += " Fraction of map overlapping with model: "
        m += "{0:.3f} and {1:.3f} (current resolution)\n".format(ovl_map1, ovl_map2)
        # print("Fraction of model overlapping with map: ", end="")
        m += " Fraction of model overlapping with map: "
        m += "{0:.3f} and {1:.3f} (current resolution)\n".format(ovl_mdl1, ovl_mdl2)
        logger.info(m)
        if refxyz:
            print("REFINE XYZ")
            timelog.start("Shiftfield")
            # print(f'maps dtype, lpcmap : {scl_cmap.fullMap.dtype}')
            # print(f'mmap dtype : {mmap.fullMap.dtype}')
            print(f"dmap dtype : {dmap.fullMap.dtype}")
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
            timelog.end("Shiftfield")
            # Read pdb and update
            # use linear interpolation instead of cubic
            # size of x,y,z for x1map=x2map=x3map
            # print(x1map.fullMap.box_size())
            timelog.start("Interpolate")
            print(gridshape0.grid_sam)
            print(gridshape0.g_reci)
            print(gridshape0.g_real)
            print(gridshape0.g_half)
            x1m_resample = x1m.copy()
            x2m_resample = x2m.copy()
            x3m_resample = x3m.copy()
            x1m_resample = resample(x1m_resample, gridshape0.grid_sam[0], axis=0)
            x1m_resample = resample(x1m_resample, gridshape0.grid_sam[1], axis=1)
            x1m_resample = resample(x1m_resample, gridshape0.grid_sam[2], axis=2)
            x2m_resample = resample(x2m_resample, gridshape0.grid_sam[0], axis=0)
            x2m_resample = resample(x2m_resample, gridshape0.grid_sam[1], axis=1)
            x2m_resample = resample(x2m_resample, gridshape0.grid_sam[2], axis=2)
            x3m_resample = resample(x3m_resample, gridshape0.grid_sam[0], axis=0)
            x3m_resample = resample(x3m_resample, gridshape0.grid_sam[1], axis=1)
            x3m_resample = resample(x3m_resample, gridshape0.grid_sam[2], axis=2)
            print(x1m_resample.shape)
            print(x1m_resample.shape)
            print(x1m_resample.shape)
            # interp_x1 = RegularGridInterpolator((zg, yg, xg), x1m)
            # interp_x2 = RegularGridInterpolator((zg, yg, xg), x2m)
            # interp_x3 = RegularGridInterpolator((zg, yg, xg), x3m)
            # count = 0
            # v = structure.map_grid_position_array(scl_map, False)
            # v = np.flip(v, 1)
            # scaling for shifts use 1 (em) instead of 2 (xtal)
            ## something wrong here in interpolation out of bounds??
            dumap.fullMap += x1m_resample  # 1.0 * interp_x1(coord_frac)
            dvmap.fullMap += x2m_resample  # 1.0 * interp_x2(coord_frac)
            dwmap.fullMap += x3m_resample  # 1.0 * interp_x3(coord_frac)
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
            ovl_mdl1,
            ovl_mdl2,
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
    coord_frac -= np.array([dwmap[zi, yi, xi], dvmap[zi, yi, xi], dumap[zi, yi, xi]]).T
    shifts_arr = mapintgt_interp(coord_frac)
    map_shifted.fullMap[zi, yi, xi] = shifts_arr
    # write final map
    map_shifted.write_to_MRC_file(mapout)
    print(map_shifted)
    # write xml results
    if xmlout is not None:
        f = open(xmlout, "w")
        for i in range(0, len(results)):
            if i == 0:
                results[i].write_xml_results_start(f, mapout, mapin)
            results[i].write_xml_results_cyc(f)
            if i == len(results) - 1:
                results[i].write_xml_results_end(f)
        f.close()

    logger.info(f"Ended at {datetime.datetime.now()}")
    timelog.profile_log()


if __name__ == "__main__":
    # parser = SheetbendParser()
    # parser.get_args()
    # main(parser.args)
    main()
