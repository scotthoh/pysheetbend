# Utily functions for shiftfield code
# S.W.Hoh, University of York, 2020

from __future__ import annotations
import os
import sys
from timeit import default_timer as timer
from collections import OrderedDict
import numpy as np
import gemmi

"""
from TEMPy.Vector import Vector as Vector
from TEMPy.EMMap import Map
from TEMPy.StructureParser import mmCIFParser as cifp
from TEMPy.StructureParser import PDBParser as pdbp
from TEMPy.ProtRep_Biopy import BioPy_Structure as BPS
from TEMPy.MapParser import MapParser as mp
"""
"""
# from time import perf_counter
from TEMPy.math.vector import Vector
from TEMPy.maps.em_map import Map
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.protein.structure_parser import mmCIFParser as cifp
from TEMPy.protein.structure_parser import PDBParser as pdbp
from TEMPy.protein.prot_rep_biopy import BioPy_Structure as BPS


#from TEMPy.protein.structure_blurrer import StructureBlurrer
"""

try:
    import pyfftw

    pyfftw_flag = True
except ImportError:
    pyfftw_flag = False

if sys.version_info[0] > 2:
    from builtins import isinstance


def match_model_map_unitcell(model, map):
    model_cell = np.array(model.cell.parameters)
    map_cell = np.array(map.unit_cell.parameters)
    if not bool(np.asarray(model_cell == map_cell).all()):
        print("Match model map unit cell")
        model.cell.set(
            map.unit_cell.a,
            map.unit_cell.b,
            map.unit_cell.c,
            map.unit_cell.alpha,
            map.unit_cell.beta,
            map.unit_cell.gamma,
        )


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
    if (
        (densMap.x_size() >= x_pos >= 0)
        and (densMap.y_size() >= y_pos >= 0)
        and (densMap.z_size() >= z_pos >= 0)
    ):
        # search all points within radius of atom
        list_points = gridtree.query_ball_point([atom.x, atom.y, atom.z], radius)
        return list_points  # ,(x_pos, y_pos, z_pos)
    else:
        print("Warning, atom out of map box")
        return []


def make_atom_overlay_map1_rad(mapin, prot, gridtree, rad):
    """
    Returns a Map instance with atom locations recorded on
    the voxel within radius with a value of 1
    """
    densMap = mapin.copy()
    densMap.fullMap = densMap.fullMap * 0.0
    # use mapgridpositions to get the points within radius. faster and efficient
    # this works. resample map before assigning density
    for atm in prot:
        # get list of nearest points of an atom
        points = mapGridPositions_radius(densMap, atm, gridtree[0], rad)
        for ind in points:
            pos = gridtree[1][ind]
            p_z = int(pos[2] - densMap.apix / 2.0)
            p_y = int(pos[1] - densMap.apix / 2.0)
            p_x = int(pos[0] - densMap.apix / 2.0)
            densMap.fullMap[p_z, p_y, p_x] = 1.0
    return densMap


def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def calc_best_grid_apix(spacing, cellsize, verbose=0):
    """
    Return best nearest grid and pixel size to input
    Arguments:
        spacing: numpy 1D array of pixel size X,Y,Z dimension
        cellsize: numpy 1D array of X,Y,Z cell size Angstroms
        verbose: verbosity, default=0
    Return:
        grid_shape, pixel_size: 1D arrays of (X,Y,Z)
    """
    if not isinstance(spacing, np.ndarray):
        spacing = np.array((spacing, spacing, spacing), dtype=np.float32)

    out_grid = []
    grid_shape = (
        int(round(cellsize[0] / spacing[0])),
        int(round(cellsize[1] / spacing[1])),
        int(round(cellsize[2] / spacing[2])),
    )
    grid_same = True
    for dim in grid_shape:
        new_dim = dim
        if new_dim % 2:
            new_dim -= 1
        largest_prime = largest_prime_factor(new_dim)
        while largest_prime > 19:
            new_dim -= 2
            largest_prime = largest_prime_factor(new_dim)
        out_grid.append(new_dim)
        if new_dim != dim:
            grid_same = False

    if not grid_same:
        newapix = []
        for i in range(0, 3):
            newapix.append(float(cellsize[i]) / float(out_grid[i]))
        # if newapix[0] == newapix[1] == newapix[2]:
        #    newapix = newapix[0]
    else:
        newapix = spacing  # np.array((newapix, newapix, newapix), dtype=np.float32)
    # check if newapix is larger than apix
    # for i in range(0, 3):
    #    if newapix[i] < apix[i]:
    if verbose > 1:
        print(f"new apix, {newapix}")
        print(f"New grid shape, {out_grid}")
    return out_grid, newapix


def plan_fft_ifft(
    gridinfo=None,
    grid_shape=None,
    grid_reci=None,
    fft_in_dtype=np.float32,
    fft_out_dtype=np.complex64,
):
    """
    Returns fft and ifft objects. Plan fft and ifft.
    Arguments
        grid_dim: GridInfo object containing grid shape data etc.
        fft_input_dtype: input type for fft, output type for ifft, default=np.float32
        fft_output_dtype: input type for ifft, output type of fft, default=np.complex64
    Returns
        fft, ifft objects
    """
    if gridinfo is None:
        grid_shape = grid_shape
        grid_reci = grid_reci
    else:
        grid_shape = gridinfo.grid_shape
        grid_reci = gridinfo.grid_reci
    fft_out_dtype = np.complex64
    if fft_in_dtype == np.float64:
        fft_out_dtype = np.complex128
    try:
        if not pyfftw_flag:
            raise ImportError
        # plan fft
        fft_arrin = pyfftw.empty_aligned(
            shape=grid_shape,
            dtype=fft_in_dtype,
        )
        fft_arrout = pyfftw.empty_aligned(
            shape=grid_reci,
            dtype=fft_out_dtype,
        )
        fft = pyfftw.FFTW(
            input_array=fft_arrin,
            output_array=fft_arrout,
            direction="FFTW_FORWARD",
            axes=(0, 1, 2),
            flags=["FFTW_ESTIMATE"],
        )

        # plan ifft
        ifft_arrin = pyfftw.empty_aligned(
            shape=grid_reci,
            dtype=fft_out_dtype,
        )
        ifft_arrout = pyfftw.empty_aligned(
            shape=grid_shape,
            dtype=fft_in_dtype,
        )
        ifft = pyfftw.FFTW(
            input_array=ifft_arrin,
            output_array=ifft_arrout,
            direction="FFTW_BACKWARD",
            axes=(0, 1, 2),
            flags=["FFTW_ESTIMATE"],
        )
    except ImportError:
        print("pyfftw import error. Not running")
    return fft, ifft


def eightpi2():
    """
    Returns 8*pi*pi
    """
    return 8.0 * np.pi * np.pi


def u2b(u_iso):
    """
    Returns the isotropic B-value
    Converts Isotropic U-value to B-value
    Argument:
    *u_iso*
      isotropic U-value
    """
    return u_iso * eightpi2()


def b2u(b_iso):
    """
    Returns the isotropic U-value
    Converts Isotropic B-value to U-value
    Argument:
    *u_iso*
      isotropic U-value
    """
    return b_iso / eightpi2()


def limit_biso(b_iso, blo, bhi):
    """
    Checks and return b_iso values within limit
    Arguments:
    *b_iso*
        b isotropic value
    *blo*
        lower limit of b isotropic value
    *bhi*
        upper limit of b isotropic value
    """
    return bhi if b_iso > bhi else blo if b_iso < blo else b_iso

    return b_iso


class Cell:
    """
    Cell object
    """

    def __init__(self, a, b, c, alpha, beta, gamma):
        """
        Arguments:
        a,b,c : Cell dimensions in angstroms
        alpha, beta, gamma : Cell angles in radians or degrees
        (will convert automatically to radians during initialisation)
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if self.alpha > np.pi:
            self.alpha = np.deg2rad(alpha)
        if self.beta > np.pi:
            self.beta = np.deg2rad(beta)
        if self.gamma > np.pi:
            self.gamma = np.deg2rad(gamma)
        rad_90deg = np.deg2rad(90.0)

        if alpha == rad_90deg and gamma == rad_90deg and beta == rad_90deg:
            self.vol = a * b * c
            # deal with null
            if self.vol <= 0.0:
                raise ValueError
            self.orthmat = np.zeros((3, 3))
            self.orthmat[0, 0] = a
            self.orthmat[1, 1] = b
            self.orthmat[2, 2] = c
            self.fracmat = np.linalg.inv(self.orthmat)
            self.realmetric = (a * a, b * b, c * c, 0.0, 0.0, 0.0)
            self.recimetric = (1 / (a * a), 1 / (b * b), 1 / (c * c), 0.0, 0.0, 0.0)
        else:
            self.vol = (a * b * c) * np.sqrt(
                2.0 * np.cos(self.alpha) * np.cos(self.beta) * np.cos(self.gamma)
                - np.cos(self.alpha) * np.cos(self.alpha)
                - np.cos(self.beta) * np.cos(self.beta)
                - np.cos(self.gamma) * np.cos(self.gamma)
                + 1.0
            )
            # deal with null
            if self.vol <= 0.0:
                raise ValueError

            # orthogonalisation + fractionisation matrices
            self.orthmat = np.identity(3)
            self.orthmat[0, 0] = a
            self.orthmat[0, 1] = self.orthmat[1, 0] = b * np.cos(
                self.gamma
            )  # if 90deg = 0
            self.orthmat[0, 2] = self.orthmat[2, 0] = c * np.cos(
                self.beta
            )  # if 90deg = 0
            self.orthmat[1, 1] = b * np.sin(self.gamma)  # if 90deg = b
            self.orthmat[1, 2] = (
                -c * np.sin(self.beta) * np.cos(self.alpha_star())
            )  # if 90deg, 0
            self.orthmat[2, 1] = self.orthmat[1, 2]
            self.orthmat[2, 2] = (
                c * np.sin(self.beta) * np.sin(self.alpha_star())
            )  # if 90deg, c
            self.fracmat = np.linalg.inv(self.orthmat)

            # calculate metric_tensor
            self.realmetric = self.real_metric_tensor()
            self.recimetric = self.reci_metric_tensor()

    def alpha_star(self):
        return np.arccos(
            (np.cos(self.gamma) * np.cos(self.beta) - np.cos(self.alpha))
            / (np.sin(self.beta) * np.sin(self.gamma))
        )

    def beta_star(self):
        return np.arccos(
            (np.cos(self.alpha) * np.cos(self.gamma) - np.cos(self.beta))
            / (np.sin(self.gamma) * np.sin(self.alpha))
        )

    def gamma_star(self):
        return np.arccos(
            (np.cos(self.beta) * np.cos(self.alpha) - np.cos(self.gamma))
            / (np.sin(self.alpha) * np.sin(self.beta))
        )

    def a_star(self):
        return self.b * self.c * np.sin(self.alpha) / self.vol

    def b_star(self):
        return self.c * self.a * np.sin(self.beta) / self.vol

    def c_star(self):
        return self.a * self.b * np.sin(self.gamma) / self.vol

    def volume(self):
        return self.vol

    def geta(self):
        return self.a

    def getb(self):
        return self.b

    def getc(self):
        return self.c

    def getalpha(self):
        return self.alpha

    def getbeta(self):
        return self.beta

    def getgamma(self):
        return self.gamma

    def real_metric_tensor(self):
        m00 = self.a * self.a
        m11 = self.b * self.b
        m22 = self.c * self.c
        m01 = 2.0 * self.a * self.b * np.cos(self.gamma)
        m02 = 2.0 * self.a * self.c * np.cos(self.beta)
        m12 = 2.0 * self.b * self.c * np.cos(self.alpha)
        return (m00, m11, m22, m01, m02, m12)

    def reci_metric_tensor(self):
        m00 = self.a_star() * self.a_star()
        m11 = self.b_star() * self.b_star()
        m22 = self.c_star() * self.c_star()
        m01 = 2.0 * self.a_star() * self.b_star() * np.cos(self.gamma_star())
        m02 = 2.0 * self.a_star() * self.c_star() * np.cos(self.beta_star())
        m12 = 2.0 * self.b_star() * self.c_star() * np.cos(self.alpha_star())
        return (m00, m11, m22, m01, m02, m12)


def metric_reci_lengthsq(x, y, z, metric_tensor):
    return (
        x * (x * metric_tensor[0] + y * metric_tensor[3] + z * metric_tensor[4])
        + y * (y * metric_tensor[1] + z * metric_tensor[5])
        + z * (z * metric_tensor[2])
    )


def get_lowerbound_posinnewbox(xg, yg, zg):
    """
    Return lowerbound of the index, and index in new box
    Argument
    *x*
      Index in x direction
    *y*
      Index in y direction
    *z*
      Index in z direction
    """

    x0 = int(xg - 1)
    y0 = int(yg - 1)
    z0 = int(zg - 1)
    nbx = xg - x0
    nby = yg - y0
    nbz = zg - z0

    return [x0, y0, z0], [nbx, nby, nbz]


def maptree_zyx(densmap):
    """
    Return gridtree and indices (z,y,x) convention
    Argument
    *densmap*
      Input density map
    """
    origin = densmap.origin
    apix = densmap.apix
    nz, ny, nx = densmap.box_size()

    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    zgc = zg * apix + origin[2]
    ygc = yg * apix + origin[1]
    xgc = xg * apix + origin[0]

    indi = np.vstack([zgc.ravel(), ygc.ravel(), xgc.ravel()]).T

    try:
        from scipy.spatial import cKDTree

        gridtree = cKDTree(indi)
    except ImportError:
        try:
            from scipy.spatial import KDTree

            gridtree = KDTree(indi)
        except ImportError:
            return

    # zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    indi = np.vstack([zg.ravel(), yg.ravel(), xg.ravel()]).T
    return gridtree, indi


def mark_selected_residues(structure, selection):
    """Mark residues in selection with 's'
    Args:
        structure (Gemmi.structure): Structure to mark
        selection (list): List of selection strings
    """
    for select in selection:
        sel = gemmi.Selection(select)
        for model in sel.models(structure):
            for chain in sel.chains(model):
                for residue in sel.residues(chain):
                    residue.flag = "s"


# @dataclass #from dataclasses import dataclass
class ResultsByCycle:
    """
    Dataclass to stor results for each cycle.
    """

    def __init__(
        self,
        cyclerr,
        cycle,
        resolution,
        radius,
        mapmdlfrac,
        mdlmapfrac,
        fscavg,
    ):
        """
        Initialise results for the cycle
         cyclerr = nth macro cycle (regularize refine cycle)
         cycle = nth cycle of the refine cycle
         resolution = resolution of the current cycle
         radius = regression radius of the current cycle
         mapmdlfrac = TEMPy score for of map overlaping with model
         mdlmapfrac = TEMPy score for model overlaping with map
         fscavg = FSC average of model to map
        """
        self.cyclerr = cyclerr
        self.cycle = cycle
        self.resolution = resolution
        self.radius = radius
        self.mapmdlfrac = mapmdlfrac
        self.mdlmapfrac = mdlmapfrac
        self.fscavg = fscavg

    """
    cyclerr: int
    cycle: int
    resolution: float
    radius: float
    mapmdlfrac: float
    mapmdlfrac_reso: float
    mdlmapfrac: float
    mdlmapfrac_reso: float
    fscavg: float
    """

    def write_xml_results_header(self, f, oppdb, ippdb):
        """
        Write the starting lines for XML output
        Arguments:
            f = file object
            oppdb = output PDB filename
            ippdb = input PDB filename
        """
        f.write("<SheetbendResult>\n")
        f.write(" <Title>{0}</Title>\n".format(os.path.basename(ippdb)))
        f.write(" <RefinedPDB>{0}</RefinedPDB>\n".format(oppdb))
        f.write(' <RefineRegulariseCycles cycle="{0}">\n'.format(self.cyclerr + 1))

    def write_xml_results_start(self, f):
        """
        Write the refine regularise cycle start lines for XML output
        Arguments:
            f = file object
        """
        f.write(' <RefineRegulariseCycles cycle="{0}">\n'.format(self.cyclerr + 1))

    def write_xml_results_start_map(self, f, mapout, mapin):
        """
        Write the starting lines for XML output for map refinement
        Arguments:
            f = file object
            mapout = output map filename
            mapin = input map filename
        """
        f.write("<SheetbendResult>\n")
        f.write(" <Title>{0}</Title>\n".format(os.path.basename(mapin)))
        f.write(" <FinalMap>{0}</FinalMap>\n".format(mapout))
        f.write(" <Cycles>\n")

    def write_xml_results_end_macrocyc(self, f, map=False):
        """
        Write the ending lines of each macro cycle for XML output
        Arguments:
            f = file object
            map = boolean to indicate if it is map refinement, Default=False
        """
        if map:
            f.write(" </Cycles>\n")
        else:
            f.write(" </RefineRegulariseCycles>\n")

    def write_xml_results_final(self, f, map=False):
        """
        Write the final lines for XML output
        Arguments:
            f = file object
            map = boolean to indicate if it is map refinement, Default=False
        """
        f.write(" <Final>\n")
        if not map:
            f.write(
                "  <RegularizeNumber>{0}</RegularizeNumber>\n".format(self.cyclerr + 1)
            )
        f.write("  <Number>{0}</Number>\n".format(self.cycle + 1))
        f.write("  <Resolution>{0}</Resolution>\n".format(self.resolution))
        f.write("  <Radius>{0}</Radius>\n".format(self.radius))
        f.write("  <OverlapMap>{0}</OverlapMap>\n".format(self.mapmdlfrac))
        if not map:
            f.write("  <OverlapModel>{0}</OverlapModel>\n".format(self.mdlmapfrac))
        f.write(" </Final>\n")
        f.write("</SheetbendResult>\n")

    def write_xml_results_cyc(self, f, map=False):
        """
        Write the results of the cycle for XML output
        Arguments:
            f = file object
            map = boolean to indicate if it is map refinement, Default=False
        """
        f.write("  <Cycle>\n")
        f.write("   <Number>{0}</Number>\n".format(self.cycle + 1))
        f.write("   <Resolution>{0}</Resolution>\n".format(self.resolution))
        f.write("   <Radius>{0}</Radius>\n".format(self.radius))
        f.write("   <OverlapMap>{0}</OverlapMap>\n".format(self.mapmdlfrac))
        if not map:
            f.write("   <OverlapModel>{0}</OverlapModel>\n".format(self.mdlmapfrac))
        f.write("  </Cycle>\n")


class Profile:
    """
    Profiling class to measure elapsed time
    """

    def __init__(self):
        """
        initialize profiling class
        ID is the key in Ordered Dictionary class
        """
        self.prof = OrderedDict()
        self.start_T = 0.0
        self.end_T = 0.0
        self.id = None

    def start(self, id):
        """
        start timer record
        Argument:
            id = id name as key for dictionary
        """
        self.id = id
        self.start_T = timer()

    def end(self, id):
        """
        stop timer record and save in dictionary
        Argument:
            id = id name as key for dictionary
        """
        self.end_T = timer()
        prof_time = self.end_T - self.start_T
        if id in self.prof:
            self.prof[id] += prof_time
        else:
            self.prof[id] = prof_time

    def profile_log(self):
        """
        print out all records
        """
        if self.prof:
            for k, v in self.prof.items():
                print("{0} : {1:.6f} s".format(k, v))


# singleton for profile
# _profile = Profile()
# set_file = _profile.set_file
# start = _profile.start
# end = _profile.end
# write = _profile.write
# close = _profile.close
