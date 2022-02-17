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
from scipy.signal import fftconvolve

#

from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.MapParser import MapParser as mp
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


# @profile
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
    ippdb = sb_parser.args.pdbin
    ipmap = sb_parser.args.mapin
    if ippdb is None or ipmap is None:
        logger.error(
            "Please provide --pdbin and --mapin to refine model against a map.\n"
            "Program terminated...\n"
        )
        # print(
        #    "Please provide --pdbin and --mapin to refine model against a map.\n"
        #    "Program terminated...\n"
        # )
        sys.exit()

    nomask = sb_parser.args.nomask
    ipmask = sb_parser.args.maskin
    oppdb = sb_parser.args.pdbout  # sheetbend_pdbout_result.pdb
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
            logger.error("Please specify the resolution of the map")
            logger.error("or resolution by cycle.\n")
            logger.error("Program terminated ...\n")
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
    structure, hetatm_present = sf_util.get_structure(ippdb, hetatom)
    structure0 = structure.copy()

    # read map
    mapin = mp.readMRC(ipmap)
    # check map box size and reduce if too big

    apix0 = mapin.apix
    ori0 = mapin.origin
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
    gridshape = sf_util.GridDimension(mapin.fullMap.shape)
    # res = 2/rec_cell[i]/grid_shape[i] for i in (0,1,2)
    # nyq_res = max(res)
    # nyquist_res = maximum([2.0 * apix0[i] for i in (0, 1, 2)])
    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)

    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)
    else:
        samp_rate = 1.5  # default same value from CLIPPER

    if pseudoreg != "no":
        logger.info("PSEUDOREGULARIZE")
        preg = pseudoregularizer.Pseudoregularize(structure0, hetatm_present)

    # create mask map if non is given at input
    # a mask that envelopes the whole particle/volume of interest
    if nomask:
        mmap = ma.make_mask_none(mapin.fullMap.shape)
        # ipmask = mmap.copy()  # so it will skip the subsequent logic test for ipmask
    elif ipmask is None:
        timelog.start("MaskMap")
        # struc_map = mapin.copy()
        # struc_map.fullMap = struc_map.fullMap * 0.0
        struc_map = structure.calculate_rho(2.5, mapin)
        tempmap = mapin.fullMap + struc_map.fullMap
        f_radius = 15.0
        rad = shiftfield.effective_radius(2, f_radius)
        pad = 5
        win_points = int(f_radius * 2) + 1 + (pad * 2)
        start = (f_radius + pad) * -1
        end = f_radius + pad
        rad_x = np.linspace(start, end, num=win_points)
        rad_y = np.linspace(start, end, num=win_points)
        rad_z = np.linspace(start, end, num=win_points)
        rad_x = rad_x * apix0
        rad_y = rad_y * apix0
        rad_z = rad_z * apix0
        rad_x = rad_x ** 2
        rad_y = rad_y ** 2
        rad_z = rad_z ** 2
        dist = np.sqrt(rad_x[:, None, None] + rad_y[:, None] + rad_x)
        dist_ind = zip(*np.nonzero(dist < rad))
        fdatar = np.zeros(dist.shape)
        count = 0
        # f000 = 0.0
        # fill the radial function map
        for i in dist_ind:
            rf = shiftfield.fltr(dist[i], f_radius, 2)
            # print(gt[1][i][0], gt[1][i][1], gt[1][i][2], rf)
            count += 1
            fdatar[i] = rf
        mmap = fftconvolve(tempmap, fdatar, mode="same")
        mmapt = scorer.calculate_map_threshold(mmap)
        mmap = ma.masked_less(mmap, mmapt)
        timelog.end("MaskMap")
    else:
        maskin = mp.readMRC(ipmask)
        # invert the values for numpy masked_array, true is masked/invalid, false is valid
        mmap = ma.make_mask(np.logical_not(maskin))

    # CASE 1:
    # Refine model against EM data
    logger.info("Refine Model against EM data")
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
        logger.info("\nRefine-regularise cycle: {0}\n".format(cycrr + 1))
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
            logger.info(f"Downsample shape : {downsamp_shape}")
            logger.info(f"Downsample apix : {downsamp_apix}")
            if verbose >= 1:
                start = timer()
            timelog.start("MapDensity")
            # cmap = downsamp_map.copy()
            # cmap.fullMap = cmap.fullMap * 0
            cmap = structure.calculate_rho(2.5, downsamp_map)
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
                interp_x1 = RegularGridInterpolator((zg, yg, xg), x1m)
                interp_x2 = RegularGridInterpolator((zg, yg, xg), x2m)
                interp_x3 = RegularGridInterpolator((zg, yg, xg), x3m)
                count = 0
                v = structure.map_grid_position_array(scl_map, False)
                v = np.flip(v, 1)
                # scaling for shifts use 1 (em) instead of 2 (xtal)
                dx = (1.0 * interp_x1(v)) * cell.a
                dy = (1.0 * interp_x2(v)) * cell.b
                dz = (1.0 * interp_x3(v)) * cell.c
                timelog.end("Interpolate")
                timelog.start("UpdateModel")
                for i in range(len(structure)):
                    if verbose >= 2:
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
            if pseudoreg == "yes":
                logger.info("PSEUDOREGULARIZE")
                timelog.start("INTPSEUDOREG")
                structure = preg.regularize_frag(structure)
                timelog.end("INTPSEUDOREG")
            # U-isotropic refinement
            if refuiso or (postrefuiso and lastcyc):
                logger.info("REFINE U ISO")
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
                    if verbose >= 2:
                        shift_u.append([structure.atomList[i].temp_fac, du[i], db[i]])
                timelog.end("UpdateModel")
            temp_result = sf_util.ResultsByCycle(
                cycrr,
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
            if len(shift_u) != 0 and verbose >= 2:
                outusiocsv = "shiftuiso_u2b_{0}.csv".format(cyc + 1)
                fuiso = open(outusiocsv, "w")
                for j in range(0, len(shift_u)):
                    fuiso.write("{0}, {1}\n".format(j, shift_u[j]))
                fuiso.close()
            # end of cycle loop
        if pseudoreg == "postref":
            logger.info("PSEUDOREGULARIZE")
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
            ovl_mapf, ovl_mdlf = scorer.calculate_overlap_scores(
                mapin, cmap, mapin_t, cmap_t
            )
            timelog.end("Scoring")
            logger.info(f"End of refine-regularise cycle {cycrr+1}")
            logger.info("TEMPys scores :")
            logger.info(
                "Fraction of map overlapping with model: {0:.3f}".format(ovl_mapf)
            )
            logger.info(
                "Fraction of model overlapping with map: {0:.3f}".format(ovl_mdlf)
            )
        # end of psedo reg loop
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

    logger.info(f"Ended at {datetime.datetime.now()}")
    timelog.profile_log()


if __name__ == "__main__":
    # parser = SheetbendParser()
    # parser.get_args()
    # main(parser.args)
    main()
