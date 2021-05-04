#%%
# Class and methods to calculate map of electron density with b-factors
# for a given structure. Scattering factors of electrons for neutral
# atoms used in clipper, gemmi, ccp4
# S.W.Hoh, University of York, 2020

import math
#Local from downloaded TEMPy code
'''
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.maps.em_map import Map
from TEMPy.protein.structure_parser import PDBParser
from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.map_process.map_filters import Filter
from TEMPy.map_process.process import MapEdit

'''
# TEMPy code from CCPEM
from TEMPy.MapParser import MapParser as mp
from TEMPy.EMMap import Map
from TEMPy.StructureParser import PDBParser
from TEMPy.StructureBlurrer import StructureBlurrer
from TEMPy.mapprocess import Filter
from TEMPy.mapprocess import MapEdit

import numpy as np
import os
import sys
if sys.version_info[0] > 2:
    from builtins import isinstance
#else:
#    from __builtins__ import isinstance


# scattering factor of electrons for neutral atoms
ElecSF = {
    "H": ((0.034900, 0.120100, 0.197000, 0.057300, 0.119500),
	   (0.534700, 3.586700, 12.347100, 18.952499, 38.626900)),
    "He": ((0.031700, 0.083800, 0.152600, 0.133400, 0.016400),
	    (0.250700, 1.475100, 4.493800, 12.664600, 31.165300)),
    "Li": ((0.075000, 0.224900, 0.554800, 1.495400, 0.935400),
	    (0.386400, 2.938300, 15.382900, 53.554501, 138.733704)),
    "Be": ((0.078000, 0.221000, 0.674000, 1.386700, 0.692500),
	    (0.313100, 2.238100, 10.151700, 30.906099, 78.327301)),
    "B": ((0.090900, 0.255100, 0.773800, 1.213600, 0.460600),
	   (0.299500, 2.115500, 8.381600, 24.129200, 63.131401)),
    "C": ((0.089300, 0.256300, 0.757000, 1.048700, 0.357500),
	   (0.246500, 1.710000, 6.409400, 18.611300, 50.252300)),
    "N": ((0.102200, 0.321900, 0.798200, 0.819700, 0.171500),
	   (0.245100, 1.748100, 6.192500, 17.389400, 48.143101)),
    "O": ((0.097400, 0.292100, 0.691000, 0.699000, 0.203900),
	   (0.206700, 1.381500, 4.694300, 12.710500, 32.472599)),
    "F": ((0.108300, 0.317500, 0.648700, 0.584600, 0.142100),
	   (0.205700, 1.343900, 4.278800, 11.393200, 28.788099)),
    "Ne": ((0.126900, 0.353500, 0.558200, 0.467400, 0.146000),
	    (0.220000, 1.377900, 4.020300, 9.493400, 23.127800)),
    "Na": ((0.214200, 0.685300, 0.769200, 1.658900, 1.448200),
	    (0.333400, 2.344600, 10.083000, 48.303699, 138.270004)),
    "Mg": ((0.231400, 0.686600, 0.967700, 2.188200, 1.133900),
	    (0.327800, 2.272000, 10.924100, 39.289799, 101.974800)),
    "Al": ((0.239000, 0.657300, 1.201100, 2.558600, 1.231200),
	    (0.313800, 2.106300, 10.416300, 34.455200, 98.534401)),
    "Si": ((0.251900, 0.637200, 1.379500, 2.508200, 1.050000),
	    (0.307500, 2.017400, 9.674600, 29.374399, 80.473198)),
    "P": ((0.254800, 0.610600, 1.454100, 2.320400, 0.847700),
	   (0.290800, 1.874000, 8.517600, 24.343399, 63.299599)),
    "S": ((0.249700, 0.562800, 1.389900, 2.186500, 0.771500),
	   (0.268100, 1.671100, 7.026700, 19.537701, 50.388802)),
    "Cl": ((0.244300, 0.539700, 1.391900, 2.019700, 0.662100),
	    (0.246800, 1.524200, 6.153700, 16.668699, 42.308601)),
    "Ar": ((0.238500, 0.501700, 1.342800, 1.889900, 0.607900),
	    (0.228900, 1.369400, 5.256100, 14.092800, 35.536098)),
    "K": ((0.411500, 1.403100, 2.278400, 2.674200, 2.216200),
	   (0.370300, 3.387400, 13.102900, 68.959198, 194.432907)),
    "Ca": ((0.405400, 1.388000, 2.160200, 3.753200, 2.206300),
	    (0.349900, 3.099100, 11.960800, 53.935299, 142.389206)),
    "Sc": ((0.378700, 1.218100, 2.059400, 3.261800, 2.387000),
	    (0.313300, 2.585600, 9.581300, 41.768799, 116.728203)),
    "Ti": ((0.382500, 1.259800, 2.000800, 3.061700, 2.069400),
	    (0.304000, 2.486300, 9.278300, 39.075100, 109.458298)),
    "V": ((0.387600, 1.275000, 1.910900, 2.831400, 1.897900),
	   (0.296700, 2.378000, 8.798100, 35.952801, 101.720100)),
    "Cr": ((0.404600, 1.369600, 1.894100, 2.080000, 1.219600),
	    (0.298600, 2.395800, 9.140600, 37.470100, 113.712097)),
    "Mn": ((0.379600, 1.209400, 1.781500, 2.542000, 1.593700),
	    (0.269900, 2.045500, 7.472600, 31.060400, 91.562202)),
    "Fe": ((0.394600, 1.272500, 1.703100, 2.314000, 1.479500),
	    (0.271700, 2.044300, 7.600700, 29.971399, 86.226501)),
    "Co": ((0.411800, 1.316100, 1.649300, 2.193000, 1.283000),
	    (0.274200, 2.037200, 7.720500, 29.968000, 84.938301)),
    "Ni": ((0.386000, 1.176500, 1.545100, 2.073000, 1.381400),
	    (0.247800, 1.766000, 6.310700, 25.220400, 74.314598)),
    "Cu": ((0.431400, 1.320800, 1.523600, 1.467100, 0.856200),
	    (0.269400, 1.922300, 7.347400, 28.989201, 90.624603)),
    "Zn": ((0.428800, 1.264600, 1.447200, 1.829400, 1.093400),
	    (0.259300, 1.799800, 6.750000, 25.586000, 73.528397)),
    "Ga": ((0.481800, 1.403200, 1.656100, 2.460500, 1.105400),
	    (0.282500, 1.978500, 8.754600, 32.523800, 98.552299)),
    "Ge": ((0.465500, 1.301400, 1.608800, 2.699800, 1.300300),
	    (0.264700, 1.792600, 7.607100, 26.554100, 77.523804)),
    "As": ((0.451700, 1.222900, 1.585200, 2.795800, 1.263800),
	    (0.249300, 1.643600, 6.815400, 22.368099, 62.039001)),
    "Se": ((0.447700, 1.167800, 1.584300, 2.808700, 1.195600),
	    (0.240500, 1.544200, 6.323100, 19.461000, 52.023300)),
    "Br": ((0.479800, 1.194800, 1.869500, 2.695300, 0.820300),
	    (0.250400, 1.596300, 6.965300, 19.849199, 50.323299)),
    "Kr": ((0.454600, 1.099300, 1.769600, 2.706800, 0.867200),
	    (0.230900, 1.427900, 5.944900, 16.675200, 42.224300)),
    "Rb": ((1.016000, 2.852800, 3.546600, -7.780400, 12.114800),
	    (0.485300, 5.092500, 25.785101, 130.451508, 138.677505)),
    "Sr": ((0.670300, 1.492600, 3.336800, 4.460000, 3.150100),
	    (0.319000, 2.228700, 10.350400, 52.329102, 151.221603)),
    "Y": ((0.689400, 1.547400, 3.245000, 4.212600, 2.976400),
	   (0.318900, 2.290400, 10.006200, 44.077099, 125.012001)),
    "Zr": ((0.671900, 1.468400, 3.166800, 3.955700, 2.892000),
	    (0.303600, 2.124900, 8.923600, 36.845798, 108.204903)),
    "Nb": ((0.612300, 1.267700, 3.034800, 3.384100, 2.368300),
	    (0.270900, 1.768300, 7.248900, 27.946501, 98.562401)),
    "Mo": ((0.677300, 1.479800, 3.178800, 3.082400, 1.838400),
	    (0.292000, 2.060600, 8.112900, 30.533600, 100.065804)),
    "Tc": ((0.708200, 1.639200, 3.199300, 3.432700, 1.871100),
	    (0.297600, 2.210600, 8.524600, 33.145599, 96.637703)),
    "Ru": ((0.673500, 1.493400, 3.096600, 2.725400, 1.559700),
	    (0.277300, 1.971600, 7.324900, 26.689100, 90.558098)),
    "Rh": ((0.641300, 1.369000, 2.985400, 2.695200, 1.543300),
	    (0.258000, 1.772100, 6.385400, 23.254900, 85.151703)),
    "Pd": ((0.590400, 1.177500, 2.651900, 2.287500, 0.868900),
	    (0.232400, 1.501900, 5.159100, 15.542800, 46.821301)),
    "Ag": ((0.637700, 1.379000, 2.829400, 2.363100, 1.455300),
	    (0.246600, 1.697400, 5.765600, 20.094299, 76.737198)),
    "Cd": ((0.636400, 1.424700, 2.780200, 2.597300, 1.788600),
	    (0.240700, 1.682300, 5.658800, 20.721901, 69.110901)),
    "In": ((0.676800, 1.658900, 2.774000, 3.183500, 2.132600),
	    (0.252200, 1.854500, 6.293600, 25.145700, 84.544800)),
    "Sn": ((0.722400, 1.961000, 2.716100, 3.560300, 1.897200),
	    (0.265100, 2.060400, 7.301100, 27.549299, 81.334900)),
    "Sb": ((0.710600, 1.924700, 2.614900, 3.832200, 1.889900),
	    (0.256200, 1.964600, 6.885200, 24.764799, 68.916801)),
    "Te": ((0.694700, 1.869000, 2.535600, 4.001300, 1.895500),
	    (0.245900, 1.854200, 6.441100, 22.173000, 59.220600)),
    "I": ((0.704700, 1.948400, 2.594000, 4.152600, 1.505700),
	   (0.245500, 1.863800, 6.763900, 21.800699, 56.439499)),
    "Xe": ((0.673700, 1.790800, 2.412900, 4.210000, 1.705800),
	    (0.230500, 1.689000, 5.821800, 18.392799, 47.249599)),
    "Cs": ((1.270400, 3.801800, 5.661800, 0.920500, 4.810500),
	    (0.435600, 4.205800, 23.434200, 136.778305, 171.756104)),
    "Ba": ((0.904900, 2.607600, 4.849800, 5.160300, 4.738800),
	    (0.306600, 2.436300, 12.182100, 54.613499, 161.997803)),
    "La": ((0.840500, 2.386300, 4.613900, 5.151400, 4.794900),
	    (0.279100, 2.141000, 10.340000, 41.914799, 132.020401)),
    "Ce": ((0.855100, 2.391500, 4.577200, 5.027800, 4.511800),
	    (0.280500, 2.120000, 10.180800, 42.063301, 130.989304)),
    "Pr": ((0.909600, 2.531300, 4.526600, 4.637600, 4.369000),
	    (0.293900, 2.247100, 10.826600, 48.884201, 147.602005)),
    "Nd": ((0.880700, 2.418300, 4.444800, 4.685800, 4.172500),
	    (0.280200, 2.083600, 10.035700, 47.450600, 146.997604)),
    "Pm": ((0.947100, 2.546300, 4.352300, 4.478900, 3.908000),
	    (0.297700, 2.227600, 10.576200, 49.361900, 145.358002)),
    "Sm": ((0.969900, 2.583700, 4.277800, 4.457500, 3.598500),
	    (0.300300, 2.244700, 10.648700, 50.799400, 146.417892)),
    "Eu": ((0.869400, 2.241300, 3.919600, 3.969400, 4.549800),
	    (0.265300, 1.859000, 8.399800, 36.739700, 125.708900)),
    "Gd": ((0.967300, 2.470200, 4.114800, 4.497200, 3.209900),
	    (0.290900, 2.101400, 9.706700, 43.426998, 125.947403)),
    "Tb": ((0.932500, 2.367300, 3.879100, 3.967400, 3.799600),
	    (0.276100, 1.951100, 8.929600, 41.593700, 131.012207)),
    "Dy": ((0.950500, 2.370500, 3.821800, 4.047100, 3.445100),
	    (0.277300, 1.946900, 8.886200, 43.093800, 133.139603)),
    "Ho": ((0.924800, 2.242800, 3.618200, 3.791000, 3.791200),
	    (0.266000, 1.818300, 7.965500, 33.112900, 101.813904)),
    "Er": ((1.037300, 2.482400, 3.655800, 3.892500, 3.005600),
	    (0.294400, 2.079700, 9.415600, 45.805599, 132.772003)),
    "Tm": ((1.007500, 2.378700, 3.544000, 3.693200, 3.175900),
	    (0.281600, 1.948600, 8.716200, 41.841999, 125.031998)),
    "Yb": ((1.034700, 2.391100, 3.461900, 3.655600, 3.005200),
	    (0.285500, 1.967900, 8.761900, 42.330399, 125.649902)),
    "Lu": ((0.992700, 2.243600, 3.355400, 3.781300, 3.099400),
	    (0.270100, 1.807300, 7.811200, 34.484901, 103.352600)),
    "Hf": ((1.029500, 2.291100, 3.411000, 3.949700, 2.492500),
	    (0.276100, 1.862500, 8.096100, 34.271198, 98.529503)),
    "Ta": ((1.019000, 2.229100, 3.409700, 3.925200, 2.267900),
	    (0.269400, 1.796200, 7.694400, 31.094200, 91.108902)),
    "W": ((0.985300, 2.116700, 3.357000, 3.798100, 2.279800),
	   (0.256900, 1.674500, 7.009800, 26.923401, 81.390999)),
    "Re": ((0.991400, 2.085800, 3.453100, 3.881200, 1.852600),
	    (0.254800, 1.651800, 6.884500, 26.723400, 81.721497)),
    "Os": ((0.981300, 2.032200, 3.366500, 3.623500, 1.974100),
	    (0.248700, 1.597300, 6.473700, 23.281700, 70.925400)),
    "Ir": ((1.019400, 2.064500, 3.442500, 3.491400, 1.697600),
	    (0.255400, 1.647500, 6.596600, 23.226900, 70.027199)),
    "Pt": ((0.914800, 1.809600, 3.213400, 3.295300, 1.575400),
	    (0.226300, 1.381300, 5.324300, 17.598700, 60.017101)),
    "Au": ((0.967400, 1.891600, 3.399300, 3.052400, 1.260700),
	    (0.235800, 1.471200, 5.675800, 18.711901, 61.528599)),
    "Hg": ((1.003300, 1.946900, 3.439600, 3.154800, 1.418000),
	    (0.241300, 1.529800, 5.800900, 19.452000, 60.575298)),
    "Tl": ((1.068900, 2.103800, 3.603900, 3.492700, 1.828300),
	    (0.254000, 1.671500, 6.350900, 23.153099, 78.709900)),
    "Pb": ((1.089100, 2.186700, 3.616000, 3.803100, 1.899400),
	    (0.255200, 1.717400, 6.513100, 23.917000, 74.703903)),
    "Bi": ((1.100700, 2.230600, 3.568900, 4.154900, 2.038200),
	    (0.254600, 1.735100, 6.494800, 23.646400, 70.377998)),
    "Po": ((1.156800, 2.435300, 3.645900, 4.406400, 1.717900),
	    (0.264800, 1.878600, 7.174900, 25.176600, 69.282097	)),
    "At": ((1.090900, 2.197600, 3.383100, 4.670000, 2.127700),
	    (0.246600, 1.670700, 6.019700, 20.765699, 57.266300)),
    "Rn": ((1.075600, 2.163000, 3.317800, 4.885200, 2.048900),
	    (0.240200, 1.616900, 5.764400, 19.456800, 52.500900)),
    "Fr": ((1.428200, 3.508100, 5.676700, 4.196400, 3.894600),
	    (0.318300, 2.688900, 13.481600, 54.386600, 200.832108)),
    "Ra": ((1.312700, 3.124300, 5.298800, 5.389100, 5.413300),
	    (0.288700, 2.289700, 10.827600, 43.538898, 145.610901)),
    "Ac": ((1.312800, 3.102100, 5.338500, 5.961100, 4.756200),
	    (0.286100, 2.250900, 10.528700, 41.779598, 128.297302)),
    "Th": ((1.255300, 2.917800, 5.086200, 6.120600, 4.712200),
	    (0.270100, 2.063600, 9.305100, 34.597698, 107.919998)),
    "Pa": ((1.321800, 3.144400, 5.437100, 5.644400, 4.010700),
	    (0.282700, 2.225000, 10.245400, 41.116199, 124.444901)),
    "U": ((1.338200, 3.204300, 5.455800, 5.483900, 3.634200),
	   (0.283800, 2.245200, 10.251900, 41.725101, 124.902298)),
    "Np": ((1.519300, 4.005300, 6.532700, -0.140200, 6.748900),
	    (0.321300, 2.820600, 14.887800, 68.910301, 81.725700)),
    "Pu": ((1.351700, 3.293700, 5.321300, 4.646600, 3.571400),
	    (0.281300, 2.241800, 9.995200, 42.793900, 132.173904)),
    "Am": ((1.213500, 2.796200, 4.754500, 4.573100, 4.478600),
	    (0.248300, 1.843700, 7.542100, 29.384100, 112.457901)),
    "Cm": ((1.293700, 3.110000, 5.039300, 4.754600, 3.503100),
	    (0.263800, 2.034100, 8.710100, 35.299198, 109.497200)),
    "Bk": ((1.291500, 3.102300, 4.930900, 4.600900, 3.466100),
	    (0.261100, 2.002300, 8.437700, 34.155899, 105.891098)),
    "Cf": ((1.208900, 2.739100, 4.348200, 4.004700, 4.649700),
	    (0.242100, 1.748700, 6.726200, 23.215300, 80.310799))
}


def eightpi2():
    """
    Returns 8*pi*pi
    """
    return (8.0 * np.pi * np.pi)


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
    list_points = gridtree.query_ball_point([atom.x, atom.y, atom.z], radius)
    #return list_points  # ,(x_pos, y_pos, z_pos)
    return list_points  # ,(x_pos, y_pos, z_pos)
  else:
    print('Warning, atom out of map box')
    return []


def u2b(u_iso):
    """
    Returns the isotropic B-value
    Argument:
    *u_iso*
      isotropic U-value
    """
    return u_iso * eightpi2()


def mapgridpos_to_coord(pos, densMap, ccpem_tempy=False):
    """
    Returns the 3D coordinates of the map grid position given
    Argument:
    *pos*
      map grid position
    *densMap*
      Map instance the atom is placed on
    """

    origin = densMap.origin
    apix = densMap.apix
    midapix = apix/2.0

    #if not ccpem_tempy:
    x_coord = pos[0] * apix + origin[0]
    y_coord = pos[1] * apix + origin[1]
    z_coord = pos[2] * apix + origin[2]
    #else:
    #    x_coord = (pos[0] - midapix) * apix + origin[0]
    #    y_coord = (pos[1] - midapix) * apix + origin[1]
    #    z_coord = (pos[2] - midapix) * apix + origin[2]

    return (x_coord, y_coord, z_coord)


def mapgridpos_to_coord_arr(pos, densMap, ccpem_tempy=False):
    """
    Returns the array of 3D coordinates of the map grid position given
    Argument:
    *pos*[[x1,y1,z1],[x2,y2,z2],...]
      an array of map grid position 
    *densMap*
      Map instance the atom is placed on
    """
    origin = densMap.origin
    if isinstance(densMap.apix, tuple):
      apix = np.array([densMap.apix[0], densMap.apix[1], densMap.apix[2]]) #x,y,z
    else:
      apix = np.array([densMap.apix, densMap.apix, densMap.apix])

    xyz_arr = pos * apix + origin

    return xyz_arr

def mod_bfact_all_atoms(atomlist, smooth):
    """
    Returns atomlist with increased/decreased bfact
    Argument:
    *atomlist*
      list of atoms
    *smooth*
      True : smoothen by adding bfact value by 20
      False : sharpen by dividing bfact value by 2
    """
    if smooth:
	  for atm in atomlist:
	    atm.temp_fac += 20.0
    else:
	    for atm in atomlist:
	      atm.temp_fac = (atm.temp_fac)/2.0

    return atomlist


def maptree(densMap):
    origin = densMap.origin
    apix = densMap.apix
    nz, ny, nx = densMap.fullMap.shape

    # convert to real coordinates
    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    # to get indices in real coordinates
    zg = zg * apix + origin[2] + apix / 2.0
    yg = yg * apix + origin[1] + apix / 2.0
    xg = xg * apix + origin[0] + apix / 2.0
    indi = np.vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T
    try:
	    from scipy.spatial import cKDTree
	    gridtree = cKDTree(indi)
    except ImportError:
	  try:
	    from scipy.spatial import KDTree
	    gridtree = KDTree(indi)
	  except ImportError:
	    return
    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    indi = np.vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T
    return gridtree, indi


class AtomShapeFn:
  """
  Atom shape function class adapted from clipper::AtomShapeFn class.
  The atomic scattering factor is instantiated for each atom in turn,
  giving the parameters: position, element, occupancy and the
  isotropic or anisotropic U-value. The methods in the class is then
  called to return calculated density in real space.
  """
  def __init__(self, atom, is_iso):
    """
    The atom is initialised.
    Arguments:
      *atom*
        BioPyAtom object from TEMPy
      *is_iso*
        True to initialise atom as isotropic
        False to initialise atom as anisotropic
    """
    if atom == []:  # if not atom
      return

    self.occ = atom.occ

    self.x = atom.x
    self.y = atom.y
    self.z = atom.z
    self.xyz = atom.get_pos_mass()[:3] #np.array([atom.x, atom.y, atom.z], dtype='float64')
    self.temp_fac = atom.temp_fac
    # self.anisou = atom.anisou
    self.is_iso = is_iso
    try:
        self.elem = atom.elem
    except:
        self.elem = ''
    self.res_no = atom.res_no

    atype = ""
    nalpha = 0
    for a in self.elem:
      if a.isalpha():
        nalpha += 1
        if nalpha == 1:
          atype += a.upper()
        else:
          atype += a.lower()
      elif not self.elem.isspace():
        atype += a
    self.elem = atype
    # calculate coefficients
    self.a = [float('nan')] * 5
    self.b = [float('nan')] * 5
    self.aw = [float('nan')] * 5
    self.bw = [float('nan')] * 5
    self.calc_coeff()

  def calc_coeff(self):
    """
    Calculate and store coefficients for initialised atom
    """
    #print(self.elem, self.res_no,self.x, self.y, self.z)
    # store coeffs and derived info
    for i in range(5):
        self.a[i] = ElecSF[self.elem][0][i]
        self.b[i] = ElecSF[self.elem][1][i]
        self.bw[i] = (-4.0*np.pi*np.pi) / (self.b[i] + self.temp_fac)
        #try:
        self.aw[i] = self.a[i] * pow(-self.bw[i]/np.pi, 1.5)
        #except ValueError:
        #  print('{0}, {1}, {2}'.format(self.temp_fac, self.b[i], self.bw[i]))

  def print_coeff(self):
    """
    Print out coefficients
    """
    print('a, b:-')
    for i in range(5):
        print(self.a[i], self.b[i])

    print('####\naw, bw:-')
    for i in range(5):
        print(self.aw[i], self.bw[i])
  
  def rho(self, pos, apix):
    """
    Returns the electron density at given coordinates
    Re-coded from clipper atomsf.cpp
    Argument:
      (x,y,z) coordinates
    """
    dx = pos[0] - self.x
    dy = pos[1] - self.y
    dz = pos[2] - self.z
    # needs to subtract half pixel size to correct the mapgrid position
    # when ccpem tempy is used
    # dx = pos[0] - apix/2.0 - self.x
    # dy = pos[1] - apix/2.0 - self.y
    # dz = pos[2] - apix/2.0 - self.z
    rsq = dx*dx + dy*dy + dz*dz

    return (self.occ * (self.aw[0]*np.exp(self.bw[0]*rsq) +
      self.aw[1]*np.exp(self.bw[1]*rsq) +
      self.aw[2]*np.exp(self.bw[2]*rsq) +
      self.aw[3]*np.exp(self.bw[3]*rsq) +
      self.aw[4]*np.exp(self.bw[4]*rsq)))
  

  def rho_arr(self, pos):
    """
    Returns the array of electron density for the 
    given array of coordinates.
    Re-coded from clipper atomsf.cpp
    Argument:
      [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]] array of coordinates shape (n,3)
    """
    dxyz = pos - self.xyz
    # needs to subtract half pixel size to correct the mapgrid position
    # when ccpem tempy is used
    # dx = pos[0] - apix/2.0 - self.x
    # dy = pos[1] - apix/2.0 - self.y
    # dz = pos[2] - apix/2.0 - self.z
    #rsq = dx*dx + dy*dy + dz*dz
    rsq = np.sum(np.square(dxyz), axis=1)


    rho_array = self.occ * (self.aw[0]*np.exp(self.bw[0]*rsq) +
                self.aw[1]*np.exp(self.bw[1]*rsq) +
                self.aw[2]*np.exp(self.bw[2]*rsq) +
                self.aw[3]*np.exp(self.bw[3]*rsq) +
                self.aw[4]*np.exp(self.bw[4]*rsq))

    return rho_array

  
def main(mapin, structure_instance):
  # create new map initialised with 0 density value
  # based on input map's origin and grid spacing.
  # calculate new grid spacing and update
  apix = mapin.apix
  print(apix)
  #if isinstance(apix, tuple):
  #    if(apix[0] == apix[1] == apix[2]):
  #        apix = apix[0]
  '''
  
  x_s = int(mapin.x_size() * apix[0])
  y_s = int(mapin.y_size() * apix[1])
  z_s = int(mapin.z_size() * apix[2])
  newMap = Map(np.zeros((z_s, y_s, x_s)),
      mapin.origin,
      apix,
      'mapname',)
  apix_x = (apix[0] * mapin.x_size()) / x_s
  apix_y = (apix[1] * mapin.x_size()) / y_s
  apix_z = (apix[2] * mapin.x_size()) / z_s
  apix = (apix_x, apix_y, apix_z)
  elif isinstance(apix, float):
  '''
  
  x_s = int(mapin.x_size() * apix)
  y_s = int(mapin.y_size() * apix)
  z_s = int(mapin.z_size() * apix)
  newMap = Map(np.zeros((z_s, y_s, x_s)),
    mapin.origin,
    apix,
    'mapname',)
  newMap.apix = (apix * mapin.x_size()) / x_s
  
  '''x_s = int(round(mapin.x_size() * mapin.apix))
  y_s = int(round(mapin.y_size() * mapin.apix))
  z_s = int(round(mapin.z_size() * mapin.apix))'''
  
  newMap.update_header()
  print('starting esf ', newMap.origin)
  # Calculate electron density and set the value to newmap
  # TEMPy's fullmap index z,y,x
  #struc_id = os.path.basename(filename).split('.')[0]
  #structure_instance = PDBParser.read_PDB_file(struc_id,
  #                                             filename,
  #                                             hetatm=True,
  #                                             water=False)
  blurrer = StructureBlurrer()

  count = 0  # for checking total no. of atoms
  print('Atom list size : {0}'.format(len(structure_instance.atomList)))
  atomList = structure_instance.atomList
  # get KDTree of coordinates
  gridtree = blurrer.maptree(newMap)
  # gridtree = maptree(newMap):q!:

  for atm in atomList:
  # get list of nearest points of an atom
  #print(atm.write_to_PDB())
    sf = AtomShapeFn(atm, is_iso=True)
    points = mapGridPositions_radius(newMap, atm, gridtree[0], 2.5)
    for ind in points:
      pos = gridtree[1][ind]  # real coordinates of the index
      # initialise AtomShapeFn object with atom
      #sf = AtomShapeFn(atm, is_iso=True)
      #if count >= 1295:
      #  sf.print_coeff()
      # get 3D coordinates from map grid position
      coord_pos = mapgridpos_to_coord(pos, newMap, False)
      # calculate electron density from 3D coordinates and
      # set to the map grid position
      # p_z=int(pos[2]-(newMap.apix/2.0))
      # p_y=int(pos[1]-(newMap.apix/2.0))
      # p_x=int(pos[0]-(newMap.apix/2.0))
      p_z = pos[2]
      p_y = pos[1]
      p_x = pos[0]
      #try:
      newMap.fullMap[p_z, p_y, p_x] += sf.rho(coord_pos, newMap.apix)
      #print('{0}, {1}, {2}, {3}'.format(atm.serial, atm.x, atm.y, atm.z))
      
    count += 1

  print('after calculate ED {0} atoms'.format(count))
  # resample to fit the input map
  #print('before downsample ', newMap.origin)
  newMap = newMap.downsample_map(apix, grid_shape=mapin.fullMap.shape)
  newMap.update_header()
  #print('in esf, {0}, {1}, {2}'.format(newMap.box_size(),
  #            newMap.apix,
  #            str(newMap.fullMap.dtype)))                                         
  #newMap.write_to_MRC_file('/home/swh514/Projects/testing_ground/example_esf_maps/python_esf_r2p5_emd3488_pptmap1.map')
  #print('end of esf ', newMap.origin)
  return newMap


def main_opt(mapin, structure_instance):
  # create new map initialised with 0 density value
  # based on input map's origin and grid spacing.
  # calculate new grid spacing and update
  apix = mapin.apix
  print(apix)
  #if isinstance(apix, tuple):
  #    if(apix[0] == apix[1] == apix[2]):
  #        apix = apix[0]
  '''
  x_s = int(mapin.x_size() * apix[0])
  y_s = int(mapin.y_size() * apix[1])
  z_s = int(mapin.z_size() * apix[2])
  newMap = Map(np.zeros((z_s, y_s, x_s)),
      mapin.origin,
      apix,
      'mapname',)
  apix_x = (apix[0] * mapin.x_size()) / x_s
  apix_y = (apix[1] * mapin.x_size()) / y_s
  apix_z = (apix[2] * mapin.x_size()) / z_s
  apix = (apix_x, apix_y, apix_z)
  elif isinstance(apix, float):
  '''
  x_s = int(mapin.x_size() * apix)
  y_s = int(mapin.y_size() * apix)
  z_s = int(mapin.z_size() * apix)
  newMap = Map(np.zeros((z_s, y_s, x_s)),
    mapin.origin,
    apix,
    'mapname',)
  newMap.apix = (apix * mapin.x_size()) / x_s
  '''x_s = int(round(mapin.x_size() * mapin.apix))
  y_s = int(round(mapin.y_size() * mapin.apix))
  z_s = int(round(mapin.z_size() * mapin.apix))'''
  
  newMap.update_header()
  print('starting esf ', newMap.origin)
  # Calculate electron density and set the value to newmap
  # TEMPy's fullmap index z,y,x
  #struc_id = os.path.basename(filename).split('.')[0]
  #structure_instance = PDBParser.read_PDB_file(struc_id,
  #                                             filename,
  #                                             hetatm=True,
  #                                             water=False)
  blurrer = StructureBlurrer()

  count = 0  # for checking total no. of atoms
  print('Atom list size : {0}'.format(len(structure_instance.atomList)))
  atomList = structure_instance.atomList
  # get KDTree of coordinates
  gridtree = blurrer.maptree(newMap)
  # gridtree = maptree(newMap):q!:

  for atm in atomList:
  # get list of nearest points of an atom
  #print(atm.write_to_PDB())
    points = mapGridPositions_radius(newMap, atm, gridtree[0], 2.5)
    for pt in range(len(points)):
      if pt == 0:
        pos = np.array([gridtree[1][points[pt]]]) # [x,y,z]
      else:
        pos = np.append(pos, [gridtree[1][points[pt]]], axis=0)
    
    sf = AtomShapeFn(atm, is_iso=True)
    
    coord_pos = mapgridpos_to_coord_arr(pos, newMap, False)
    rho_array = sf.rho_arr(coord_pos)
    #pos_zyx = np.flip(pos, 1)
    for i in range(len(pos)):
      newMap.fullMap[pos[i][2], pos[i][1], pos[i][0]] += rho_array[i]
    '''for ind in points:
      pos = gridtree[1][ind]  # real coordinates of the index
      # initialise AtomShapeFn object with atom
      sf = AtomShapeFn(atm, is_iso=True)
      #if count >= 1295:
      #  sf.print_coeff()
      # get 3D coordinates from map grid position
      
      # calculate electron density from 3D coordinates and
      # set to the map grid position
      # p_z=int(pos[2]-(newMap.apix/2.0))
      # p_y=int(pos[1]-(newMap.apix/2.0))
      # p_x=int(pos[0]-(newMap.apix/2.0))
      p_z = pos[2]
      p_y = pos[1]
      p_x = pos[0]
      #try:
      newMap.fullMap[p_z, p_y, p_x] += sf.rho(coord_pos, newMap.apix)
      #print('{0}, {1}, {2}, {3}'.format(atm.serial, atm.x, atm.y, atm.z))
    '''
    count += 1

  print('after calculate ED {0} atoms'.format(count))
  # resample to fit the input map
  #print('before downsample ', newMap.origin)
  newMap = newMap.downsample_map(apix, grid_shape=mapin.fullMap.shape)
  newMap.update_header()
  #print('in esf, {0}, {1}, {2}'.format(newMap.box_size(),
  #            newMap.apix,
  #            str(newMap.fullMap.dtype)))                                         
  #newMap.write_to_MRC_file('/home/swh514/Projects/testing_ground/example_esf_maps/python_esf_r2p5_emd3488_pptmap1.map')
  #print('end of esf ', newMap.origin)
  return newMap

if __name__ == '__main__':
    mapin = mp.readMRC('/home/swh514/Projects/data/EMD-3488/map/emd_3488.map')
    filename = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1.ent'
    #main(mapin, filename)


# %%

