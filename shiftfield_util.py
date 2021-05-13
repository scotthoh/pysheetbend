# Utily functions for shiftfield code
# S.W.Hoh, University of York, 2020

import math
import numpy as np
from timeit import default_timer as timer
'''from TEMPy.math.vector import Vector
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.protein.structure_blurrer import StructureBlurrer
'''

from TEMPy.Vector import Vector as Vector
from TEMPy.EMMap import Map

try:
    import pyfftw
    pyfftw_flag = True
except ImportError:
    pyfftw_flag = False

import sys
if sys.version_info[0] > 2:
    from builtins import isinstance


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
        list_points = gridtree.query_ball_point([atom.x,
                                                 atom.y,
                                                 atom.z],
                                                radius)
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
    densMap.fullMap = densMap.fullMap * 0.0
    # use mapgridpositions to get the points within radius. faster and efficient
    # this works. resample map before assigning density
    for atm in prot:
        # get list of nearest points of an atom
        points = mapGridPositions_radius(densMap, atm, gridtree[0], rad)
        for ind in points:
            pos = gridtree[1][ind]
            p_z = int(pos[2] - densMap.apix/2.0)
            p_y = int(pos[1] - densMap.apix/2.0)
            p_x = int(pos[0] - densMap.apix/2.0)
            densMap.fullMap[p_z, p_y, p_x] = 1.0
    return densMap


class grid_dim:
    """
    Grid dimensions object
    """
    def __init__(self, densMap):
        """
        Sets up grid size in real, reciprocal and half space
        Arguments
        *densMap*
          Input map
        """
        self.grid_sam = densMap.fullMap.shape
        self.g_reci = (densMap.z_size(), densMap.y_size(),
                       densMap.x_size()//2+1)
        self.g_real = (self.g_reci[0], self.g_reci[1],
                       int(self.g_reci[2]-1)*2)
        self.g_half = (densMap.z_size()//2, densMap.y_size()//2,
                       densMap.x_size()//2+1)


def plan_fft(grid_dim):
    """
    Returns fft object. Plan fft
    Arguments
    *grid_dim*
      grid shape of map
    """
    output_shape = grid_dim.g_reci
    # need to add scipy fft if pyfftw not imported
    try:
        if not pyfftw_flag:
            raise ImportError

        input_arr = pyfftw.empty_aligned(grid_dim.grid_sam,
                                         dtype='float64', n=16)
        output_arr = pyfftw.empty_aligned(output_shape,
                                          dtype='complex128', n=16)
        # fft planning
        fft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_FORWARD',
                          axes=(0, 1, 2), flags=['FFTW_ESTIMATE'])
    except ImportError:
        print('Not running')

    return fft


def plan_ifft(grid_dim):
    """
    Returns ifft object. Plan ifft
    Arguments
    *grid_dim*
      grid shape of map
    """
    output_shape = grid_dim.g_real
    # need to add scipy fft if pyfftw not imported
    try:
        if not pyfftw_flag:
            raise ImportError
        input_arr = pyfftw.empty_aligned(grid_dim.g_reci,
                                         dtype='complex128', n=16)
        output_arr = pyfftw.empty_aligned(output_shape,
                                          dtype='float64', n=16)
        # ifft planning,
        ifft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_BACKWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
    except ImportError:
        print('Not running')

    return ifft


def cor_mod(a, b):
    """
    Returns corrected remainder of division. If remainder <0,
    then adds value b to remainder.
    Arguments
    *a*
      Dividend
    *b*
      Divisor
    """
    c = np.fmod(a, b)
    if (c < 0):
        c += b
    return int(c)


def hkl_c(c, ch, g):
    """
    Returns the index 
    Arguments
    *c*
      Index h,k,l
    *ch*
      Half of the grid shape
    *g*
      Real space grid shape
    """

    # z, y, x; return vector(x, y ,z)
    cv = Vector(int(c[2]), int(c[1]), int(c[0]))
    chv = Vector(int(ch[2]), int(ch[1]), int(ch[0]))
    # cv = Vector(int(c[0]), int(c[1]), int(c[2]))
    # chv = Vector(int(ch[0]), int(ch[1]), int(ch[2]))
    v1 = cv + chv
    m1 = Vector(cor_mod(v1.x, g[2]),
                cor_mod(v1.y, g[1]),
                cor_mod(v1.z, g[0]))

    return m1 - chv


def eightpi2():
    """
    Returns 8*pi*pi
    """
    return (8.0 * np.pi * np.pi)


def u2b(u_iso):
    """
    Returns the isotropic B-value
    Converts Isotropic U-value to B-value
    Argument
    *u_iso*
      isotropic U-value
    """
    return u_iso * eightpi2()


def b2u(b_iso):
    """
    Returns the isotropic U-value
    Converts Isotropic B-value to U-value
    Argument
    *u_iso*
      isotropic U-value
    """
    return b_iso / eightpi2()


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
            print('deg2rad')
        if self.beta > np.pi:
            self.beta = np.deg2rad(beta)
        if self.gamma > np.pi:
            self.gamma = np.deg2rad(gamma)

        self.vol = a*b*c*np.sqrt(2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                                 - np.cos(alpha)*np.cos(alpha)
                                 - np.cos(beta)*np.cos(beta)
                                 - np.cos(gamma)*np.cos(gamma)+1.0)

        # deal with null
        if self.vol <= 0.0:
            return

        # orthogonalisation + fractionisation matrices
        self.orthmat = np.identity(3)
        self.orthmat[0, 0] = a
        self.orthmat[0, 1] = b*np.cos(gamma)
        self.orthmat[0, 2] = c*np.cos(beta)
        self.orthmat[1, 1] = b*np.sin(gamma)
        self.orthmat[1, 2] = -c*np.sin(beta)*np.cos(self.alpha_star())
        self.orthmat[2, 2] = c*np.sin(beta)*np.sin(self.alpha_star())
        self.fracmat = np.linalg.inv(self.orthmat)

        # calculate metric_tensor
        self.realmetric = self.metric_tensor(self.a, self.b, self.c,
                                             self.alpha, self.beta, self.gamma)
        self.recimetric = self.metric_tensor(self.a_star(), self.b_star(),
                                             self.c_star(), self.alpha_star(),
                                             self.beta_star(),
                                             self.gamma_star())

    def alpha_star(self):
        return np.arccos((np.cos(self.gamma)*np.cos(self.beta)-np.cos(self.alpha))
                         / (np.sin(self.beta)*np.sin(self.gamma)))

    def beta_star(self):
        return np.arccos((np.cos(self.alpha)*np.cos(self.gamma)-np.cos(self.beta))
                         / (np.sin(self.gamma)*np.sin(self.alpha)))

    def gamma_star(self):
        return np.arccos((np.cos(self.beta)*np.cos(self.alpha)-np.cos(self.gamma))
                         / (np.sin(self.alpha)*np.sin(self.beta)))

    def a_star(self):
        return self.b*self.c*np.sin(self.alpha)/self.vol

    def b_star(self):
        return self.c*self.a*np.sin(self.beta)/self.vol

    def c_star(self):
        return self.a*self.b*np.sin(self.gamma)/self.vol

    def vol(self):
        return self.vol

    def a(self):
        return self.a

    def b(self):
        return self.b

    def c(self):
        return self.c

    def alpha(self):
        return self.alpha

    def beta(self):
        return self.beta

    def gamma(self):
        return self.gamma

    def metric_tensor(self, a, b, c, alp, bet, gam):
        m00 = a*a
        m11 = b*b
        m22 = c*c
        m01 = 2.0*a*b*np.cos(gam)
        m02 = 2.0*a*c*np.cos(bet)
        m12 = 2.0*b*c*np.cos(alp)
        return (m00, m11, m22, m01, m02, m12)

    def metric_reci_lengthsq(self, x, y, z):
        return (x*(x*self.recimetric[0] + y*self.recimetric[3]
                   + z*self.recimetric[4])
                + y*(y*self.recimetric[1] + z*self.recimetric[5])
                + z*(z*self.recimetric[2]))


def crop_mapgrid_points(x0, y0, z0, densmap, k=4):
    """
    Return cropped map containing density data
    of surrounding 64 map grid points of atom_point
    """
    boxmap = Map(np.zeros((4, 4, 4)),
                 [0, 0, 0],
                 densmap.apix,
                 'mapname',)

    # problem: should the atom point be at the corner of the box or
    # the centre for interpolation
    # atm mapgridposition at corner
    for iz in range(0, 4):
        for iy in range(0, 4):
            for ix in range(0, 4):
                mx = ix + x0
                my = iy + y0
                mz = iz + z0
                boxmap.fullMap[iz, iy, ix] = densmap.fullMap[mz, my, mx]

    return boxmap  # cropped map, atm pos in cropped map


def get_lowerbound_posinnewbox(x, y, z):
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
    x0 = x - 1
    y0 = y - 1
    z0 = z - 1
    nbx = x - x0
    nby = y - y0
    nbz = z - z0

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


class radial_fltr:
    """
    Radial filter object
    """
    def __init__(self, radcyc, function, densmap):
        """
        Initialise filter object, sets the radius and function to use
        Arguments
        *radcyc*
          radius of the filter
        *function*
          0 = step, 1 = linear, 2 = quadratic
        *densmap*
          reference density map
        """
        self.verbose = 0
        self.radius = radcyc
        self.function = function
        '''if function == 'step':
            self.function = 0
        elif function == 'linear':
            self.function = 1
        elif function == 'quadratic':
            self.function = 2
        '''
        # function = step, linear, quadratic
        # determine effective radius of radial function
        self.nrad = 1000
        self.drad = 0.25
        self.sum_r = [0.0]*self.nrad
        self.gridshape = grid_dim(densmap)

        for i in range(0, self.nrad):
            r = self.drad * (float(i) + 0.5)
            self.sum_r[i] = r*r*math.fabs(self.fltr(r))

        for i in range(1, self.nrad):
            self.sum_r[i] += self.sum_r[i-1]
        for i in range(0, self.nrad):
            if self.sum_r[i] > 0.99*self.sum_r[self.nrad-1]:
                break
        self.rad = self.drad*(float(i)+1.0)
        self.fltr_data_r = np.zeros(self.gridshape.grid_sam, dtype='float64')
        f000 = 0.0
        # z,y,x convention
        origin = np.array([densmap.origin[2], densmap.origin[1],
                          densmap.origin[0]])
        if isinstance(densmap.apix, tuple):
            apix = np.array([densmap.apix[2], densmap.apix[1],
                            densmap.apix[0]])
        else:    
            apix = np.array([densmap.apix, densmap.apix, densmap.apix])

        #g_half = (g_real[0]//2, g_real[1]//2, g_real[0]//2+1)
        #SB = StructureBlurrer()
        #gt = SB.maptree(densmap)
        self.gt = maptree_zyx(densmap)
        # filpping indices made things wrong because the 
        # the chronology of indices has changed.
        # is this true? pt1 = 001 and pt1 = 100 different
        #start = timer() # flipping indices takes about 0.4 sec 
        #indi = np.flip(gt[1], 1) # gt[1] indices is x,y,z , flip become z,y,x
        #end = timer()
        #print('flip indices ', end-start)

        gh = np.array([self.gridshape.g_half])
        if self.verbose >= 1:
            start = timer()
        c = self.gt[1] + gh  # self.gridshape.g_half
        if self.verbose >= 1:
            end = timer()
            print('indi + halfgrid ', end-start)

        if self.verbose >= 1:
            start = timer()
        c1 = self.cor_mod1(c, self.gridshape.grid_sam) - gh
        if self.verbose >= 1:
            end = timer()
            print('cor mod ', end-start)

        if self.verbose >= 1:
            start = timer()
        pos = c1[:]*apix+origin
        r = np.sqrt(np.sum(np.square(pos), axis=1))
        if self.verbose >= 1:
            end = timer()
            print('indices get r ', end-start)
        #for i in range(len(r)):
        #    print(pos[i], c1[i], gt[1][i], r[i])
        if self.verbose >= 1:
            start = timer()
            print('self.rad ', self.rad)
        r_ind = np.nonzero(r < self.rad)
        #r = np.where(r < self.rad, self.fltr(r), 0.0)
        #r_ind = np.nonzero(r)
        if self.verbose >= 1:
            end = timer()
            print('nonzero transpose ', end-start)

        if self.verbose >= 1:
            start = timer()
        count = 0
        # fill the radial function map
        for i in r_ind[0]:
            rf = self.fltr(r[i])
            f000 += rf
            #print(gt[1][i][0], gt[1][i][1], gt[1][i][2], rf)
            count += 1
            self.fltr_data_r[self.gt[1][i][0], self.gt[1][i][1], self.gt[1][i][2]] = rf #[i]
        
        if self.verbose >= 1:
            end = timer()
            print('fill radial function map ', end-start)
            print('count ', count)
        '''
        count = 0
        print('self.rad ', self.rad)
        for ind in range(0,len(gt[1])):
            xyz = gt[1][ind]
            c = hkl_c((xyz[2], xyz[1], xyz[0]), self.gridshape.g_half, self.gridshape.grid_sam)
            pos = mapgridpos_to_coord((c[0],c[1],c[2]),densmap)
            r = Vector(pos[0],pos[1],pos[2]).mod()
            #r = Vector(cz, cy, cx).mod()
            if r < self.rad:
                print((pos[2],pos[1],pos[0]), (c[2],c[1],c[0]), (xyz[2], xyz[1], xyz[0]), r)
                r = self.fltr(r)
                f000 += r
                print(r)
                #self.fltr_data_r[xyz[2], xyz[1], xyz[0]] = r
                count+=1
        end=timer()
        '''
        # calc scale factor
        self.scale = 1.0/f000
        if self.verbose >= 1:
            print('scale, ', self.scale, ' f000, ', f000)

    def cor_mod1(self, a, b):
        """
        Returns corrected remainder of division. If remainder <0,
        then adds value b to remainder.
        Arguments
        *a*
          array of Dividend (z,y,x indices)
        *b*
          array of Divisor (z,y,x indices)
        """
        c = np.fmod(a, b)
        d = np.transpose(np.nonzero(c < 0))
        #d, e = np.nonzero(c<0)
        for i in d:  #range(len(d)):
            c[i[0], i[1]] += b[i[1]]
            #c[i, j] += b[i]
        return c

    def fltr(self, r):
        """
        Returns radius value from filter function
        Arguments
        *r*
          radius
        """
        if r < self.radius:
            if self.function == 2:
                return pow((1.0-r)/self.radius, 2)
            elif self.function == 1:
                return (1.0-r)/self.radius
            elif self.function == 0:
                return 1.0
        else:
            return 0.0

    def mapfilter(self, data_arr, fft_obj, ifft_obj):
        """
        Returns filtered data
        Argument
        *data_arr*
          array of data to be filtered
        *fft_obj*
          fft object
        *ifft_obj*
          ifft object
        """
        # copy map data and filter data
        data_r = np.zeros(self.gridshape.grid_sam, dtype='float64')
        data_r = data_arr.copy()
        fltr_input = np.zeros(self.gridshape.grid_sam, dtype='float64')
        fltr_input = self.fltr_data_r.copy()
        
        if self.verbose >= 1:
            start = timer()
        # create complex data array
        fltr_data_c = np.zeros(self.gridshape.g_reci, dtype='complex128')
        data_c = np.zeros(self.gridshape.g_reci, dtype='complex128')
        # fourier transform of filter data
        fltr_data_c = fft_obj(fltr_input, fltr_data_c)
        fltr_data_c = fltr_data_c.conjugate().copy()
        if self.verbose >= 1:
            end = timer()
            print('fft fltr_data : {0}s'.format(end-start))
        if self.verbose >= 1:
            start = timer()
        # fourier transform of map data
        data_c = fft_obj(data_r, data_c)
        data_c = data_c.conjugate().copy()
        if self.verbose >= 1:
            end = timer()
            print('fft data : {0}s'.format(end-start))
        # apply filter
        if self.verbose >= 1:
            start = timer()
        data_c[:, :, :] = self.scale*data_c[:, :, :]*fltr_data_c[:, :, :]
        if self.verbose >= 1:
            end = timer()
            print('Convolution : {0}s'.format(end-start))

        # inverse fft
        if self.verbose >= 1:
            start = timer()
        data_c = data_c.conjugate().copy()
        data_r = ifft_obj(data_c, data_r)
        if self.verbose >= 1:
            end = timer()
            print('ifft : {0}s'.format(end-start))

        return data_r


# @dataclass #from dataclasses import dataclass
class results_by_cycle:
    '''
    Dataclass to stor results for each cycle.
    '''
    def __init__(self, cyclerr, cycle, resolution, radius,
                 mapmdlfrac, mapmdlfrac_reso,
                 mdlmapfrac, mdlmapfrac_reso, fscavg):
        self.cyclerr = cyclerr
        self.cycle = cycle
        self.resolution = resolution
        self.radius = radius
        self.mapmdlfrac = mapmdlfrac
        self.mapmdlfrac_reso = mapmdlfrac_reso
        self.mdlmapfrac = mdlmapfrac
        self.mdlmapfrac_reso = mdlmapfrac_reso
        self.fscavg = fscavg
    '''
    cyclerr: int
    cycle: int
    resolution: float
    radius: float
    mapmdlfrac: float
    mapmdlfrac_reso: float
    mdlmapfrac: float
    mdlmapfrac_reso: float
    fscavg: float
    '''
    def write_xml_results_start(self, f):
        f.write('<SheetbendResult>\n')
        f.write(' <Title>{0}</Title>\n'.format('temp'))
        f.write(' <Cycles>\n')
        
    def write_xml_results_end(self, f):
        f.write(' </Cycles>\n')
        f.write(' <Final>\n')
        f.write('   <RegularizeNumber>{0}</RegularizeNumber>\n'
                .format(self.cyclerr+1))
        f.write('   <Number>{0}</Number>\n'.format(self.cycle+1))
        f.write('   <Resolution>{0}</Resolution>\n'.format(self.resolution))
        f.write('   <Radius>{0}</Radius>\n'.format(self.radius))
        f.write('   <OverlapMap>{0}</OverlapMap>\n'.format(self.mapmdlfrac))
        f.write('   <OverlapMapAtResolution>{0}</OverlapMapAtResolution>\n'
                .format(self.mapmdlfrac_reso))
        f.write('   <OverlapModel>{0}</OverlapModel>\n'
                .format(self.mdlmapfrac))
        f.write('   <OverlapModelAtResolution>{0}</OverlapModekAtResolution>\n'
                .format(self.mdlmapfrac_reso))
        f.write(' </Final>\n')
        f.write('</SheetbendResult\n')

    def write_xml_results_cyc(self, f):
        f.write('  <Cycle>\n')
        f.write('   <RegularizeNumber>{0}</RegularizeNumber>\n'
                .format(self.cyclerr+1))
        f.write('   <Number>{0}</Number>\n'.format(self.cycle+1))
        f.write('   <Resolution>{0}</Resolution>\n'.format(self.resolution))
        f.write('   <Radius>{0}</Radius>\n'.format(self.radius))
        f.write('   <OverlapMap>{0}</OverlapMap>\n'.format(self.mapmdlfrac))
        f.write('   <OverlapMapAtResolution>{0}</OverlapMapAtResolution>\n'
                .format(self.mapmdlfrac_reso))
        f.write('   <OverlapModel>{0}</OverlapModel>\n'
                .format(self.mdlmapfrac))
        f.write('   <OverlapModelAtResolution>{0}</OverlapModekAtResolution>\n'
                .format(self.mdlmapfrac_reso))
        f.write('  </Cycle>\n')
