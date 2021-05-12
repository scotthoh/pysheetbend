import argparse
from _version import VERSION as __version__


class sheetbendParser(object):

    def __init__(self, args=None):
        self.args = args

    def get_args(self):
        parser = argparse.ArgumentParser(prog='sheetbend')

        infiles = parser.add_argument_group('Input Files')
        infiles.add_argument(
            '-m',
            '--mapin',
            help='Reference input map',
            metavar='Input_map',
            type=str,
            dest='mapin',
            default=None,
            required=False
        )

        infiles.add_argument(
            '-m1',
            '--mapin1',
            help='Input map to be refined against reference map',
            metavar='Input_map1',
            type=str,
            dest='mapin1',
            default=None,
            required=False
        )

        infiles.add_argument(
            '-p',
            '--pdbin',
            help='Input model to be refined.',
            metavar='Input_PBD',
            type=str,
            default=None,
            dest='pdbin',
            required=False
        )

        outfiles = parser.add_argument_group('Output Files')
        outfiles.add_argument(
            '-mapout',
            '--mapout',
            help='Output map filename',
            metavar='Output_map',
            type=str,
            dest='mapout',
            default='sheetbend_mapout_result.map',
            required=False
        )

        outfiles.add_argument(
            '-pout',
            '--pdbout',
            help='Output model filename.',
            metavar='Output_PBD',
            type=str,
            default='sheetbend_pdbout_result.pdb',
            dest='pdbout',
            required=False
        )

        outfiles.add_argument(
            '-xml',
            '--xmlout',
            help='Prints xml output of the calculation. Default=program.xml',
            metavar='Xmlout',
            type=str,
            default='program.xml',
            dest='xmlout',
            required=False
        )

        res = parser.add_argument_group('Refinement Parameters')
        res.add_argument(
            '-r',
            '--resolution',
            help='''Resolution (Angs) for the calculation. All data is truncated.
                 If resolution by cycle are not supplied, this resolution is
                 applied for every cycle.''',
            metavar='Resolution',
            type=float,
            default=-1.0,
            dest='res',
            required=False
        )

        res.add_argument(
            '-rcyc',
            '--res-by-cycle',
            nargs='+',
            help='''Set the resolution for each cycle of multi-cycle calculation.
                 Each resolution is separated by space. If there are fewer
                 resolutions than cycles, linear interpolation is used to
                 fill in the remaining values.''',
            metavar='Resolutions_by_cycle',
            type=float,
            default=None,
            dest='res_by_cyc',
            required=False
        )
        
        cyc = parser.add_argument_group()
        cyc.add_argument(
            '-c',
            '--cycles',
            help='''Number of cycles to perform. For coordinates, 10-20 is probably
                 reasonable, (Default=1)''',
            metavar='Cycles',
            type=int,
            default=1,
            dest='cycle',
            required=False
        )

        radius = parser.add_argument_group()
        radius.add_argument(
            '-rad',
            '--radius',
            help='''Radius (Angstrom) of the regression refinement sphere.
                 This controls the size of the regions which are 'dragged'
                 by the morphing calculation. Larger radii lead to bulk
                 changes, smaller radii allow smaller features to move
                 independently, at a cost of messing up the geometry.
                 Avoid radius < 2.5*resolution.
                 Default = radius_scale * resolution of the cycle''',
            metavar='Radius',
            type=float,
            default=-1.0,
            dest='rad',
            required=False
        )

        radius.add_argument(
            '-radsc',
            '--radius-scale',
            help='''Automatically set the radius in proportion to the resolution
                 for the current cycle. The resolution is multiplied by this
                 factor to get the radius. Overridden by radius.
                 (Default=5.0)''',
            metavar='Radius_scale',
            type=float,
            default=5.0,
            dest='radscl',
            required=False
        )

        refine = parser.add_argument_group()
        refine.add_argument(
            '-coord',
            '--coord',
            help='''Perform coordinate refinement. If no refinement option is
                 specified, coordinate refinement is enabled by default.''',
            action='store_true',
            dest='refxyz',
            required=False
        )

        refine.add_argument(
            '-u-iso',
            '--u-iso',
            help='''Perform B-factor refinement. (Default=False)''',
            action='store_true',
            dest='refuiso',
            required=False
        )
        
        postref = parser.add_argument_group()
        postref.add_argument(
            '-postrefine-coord',
            '--postrefine-coord',
            help='''Perform post-refinement on coordinates. (Default=False)''',
            action='store_true',
            dest='postrefxyz',
            required=False
        )
        postref.add_argument(
            '-postrefine-u-iso',
            '--postrefine-u-iso',
            help='''Perform post-refinement on B-factors. (Default=False)''',
            action='store_true',
            dest='postrefuiso',
            required=False
        )

        postref.add_argument(
            '-pseudo-regularise',
            '--pseudo-regularise',
            help='Pseudo regularise the model at the end of the calculation.',
            action='store_true',
            dest='pseudoreg',
            required=False,
        )
        bisorange = parser.add_argument_group()
        bisorange.add_argument(
            '-b-iso-range',
            '--b-iso-range',
            help='''Sets the lower and upper bound for B-isotropic value
                  refinement. (Default: 0.1 999.9)''',
            metavar=('blo', 'bhi'),
            type=float,
            nargs=2,
            default=[0.1, 999.9],
            dest='biso_range',
            required=False
        )

        prog_version = parser.add_argument_group()
        prog_version.add_argument(
            '-v',
            '--version',
            action='version',
            version='%(prog)s {version}'.format(version=__version__)
            )

        verbose = parser.add_argument_group()
        verbose.add_argument(
            '-verbose',
            '--verbose',
            help='Sets verbosity of terminal output. Default=0',
            metavar='Verbosity',
            type=int,
            default=0,
            dest='verbose',
            required=False
        )

        self.args = parser.parse_args()

    def print_args(self):
        print('Input parameters: \n')
        for arg in vars(self.args):
            arg_val = getattr(self.args, arg)
            if isinstance(arg_val, list):
                print(arg, arg_val)
            if isinstance(arg_val, float):
                if (arg_val > 0.0) and (not None):
                    print(arg, arg_val)
            if isinstance(arg_val, int):
                if arg_val > 0:
                    print(arg, arg_val)
            if isinstance(arg_val, str):
                if arg_val is not None:
                    print(arg, arg_val)
