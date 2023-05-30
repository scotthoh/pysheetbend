"""
Author: Soon Wen Hoh, Kevin Cowtan
York Structural Biology Laboratory

License: ?
"""
import argparse
import platform

from pysheetbend import __version__


class SheetbendParser:
    """
    Sheetbend commandline parser
    """

    def __init__(self, args=None):
        self.args = args

    def add_args(self, parser):
        """
        Parse input arguments
        """
        # parser = argparse.ArgumentParser()
        parser.prog = "pysheetbend"
        parser.description = """Shift-field refinement of macromolecular
        atomic models for cryo-EM data."""
        # parser = parser.add_parsers(dest="command")
        # parser.add_parser("refine_model_to_map")
        # parser.add_parser("refine_map_to_map")

        infiles = parser.add_argument_group("Input Files")
        infiles.add_argument(
            "--mapin",
            type=str,
            dest="mapin",
            help="Reference input map",
            metavar="Input_map",
            default=None,
            required=True,
        )

        infiles.add_argument(
            "--mapin2",
            type=str,
            dest="mapin2",
            help="Input map to be refined against reference map",
            metavar="Input_map2",
            default=None,
            required=False,
        )

        infiles.add_argument(
            "--pdbin",
            "--model",
            type=str,
            dest="pdbin",
            help="Input model to be refined.",
            metavar="Input_PBD",
            default=None,
            required=False,
        )

        infiles.add_argument(
            "-no-ligands",
            "--no-ligands",
            action="store_true",
            dest="no_ligands",
            help="Remove ligands.",
            required=False,
        )

        infiles.add_argument(
            "-no-h2o",
            "--no-waters",
            action="store_true",
            dest="no_water",
            help="Remove water.",
            required=False,
        )

        infiles.add_argument(
            "-maskin",
            "--maskin",
            type=str,
            dest="maskin",
            help="Input mask.",
            metavar="Input_mask",
            default=None,
            required=False,
        )

        infiles.add_argument(
            "--nomask",
            action="store_true",
            dest="nomask",
            help="No mask.",
            required=False,
        )

        outfiles = parser.add_argument_group("Output Files")

        outfiles.add_argument(
            "--pdbout",
            type=str,
            dest="pdbout",
            help="Output model filename.",
            metavar="Output_PBD",
            default=None,
            required=False,
        )

        outfiles.add_argument(
            "--mapout",
            type=str,
            dest="mapout",
            help="Output map filename.",
            metavar="Output_map",
            default="sheetbend_mapout_result.mrc",
            required=False,
        )

        outfiles.add_argument(
            "--xmlout",
            type=str,
            dest="xmlout",
            help=(
                "XML filename. Prints XML output of the calculation."
                "(Default: program.xml)"
            ),
            metavar="Xmlout",
            default="program.xml",
            required=False,
        )

        outfiles.add_argument(
            "--no-xmlout",
            action="store_true",
            dest="no_xml",
            help="If set, will not write XML outfile.",
            required=False,
        )

        outfiles.add_argument(
            "--intermediate",
            action="store_true",
            dest="intermediate",
            help="Output model files every cycle.",
            required=False,
        )

        ref = parser.add_argument_group("Refinement Parameters")
        ref.add_argument(
            "--resolution",
            type=float,
            dest="res",
            help=(
                "Resolution (Angstrom) of the input map."
                "This will be used to calculate grid spacing"
                "for resampling grids."
                "If resolutions by cycle are not supplied,"
                "the program will run at this resolution only."
            ),
            metavar="Resolution",
            required=True,
        )

        ref.add_argument(
            "--res-by-cycle",
            type=float,
            nargs="+",
            dest="res_by_cyc",
            help=(
                "Set the resolution for each cycle of multi-cycle calculation."
                "Each resolution is separated by space. If there are fewer"
                "resolutions than cycles, linear interpolation is used to"
                "fill in the remaining values."
                "If the final resolution in --res-by-cycle is smaller in value"
                "than --resolution (if provided), it will be replaced by the"
                "value from --resolution."
            ),
            metavar="Resolutions_by_cycle",
            default=None,
            required=False,
        )

        ref.add_argument(
            "--cycles",
            type=int,
            dest="cycle",
            help=(
                "Number of cycles to perform. For coordinates, 10-20 is"
                "probably reasonable, (Default=1)"
            ),
            metavar="Cycles",
            default=1,
            required=False,
        )

        ref.add_argument(
            "--macro-cycles",
            type=int,
            dest="cycle_regularise",
            help=(
                "Number of refine-regularise (macro) cycles to perform. (Default = 1)"
            ),
            metavar="CyclesRR",
            default=1,
            required=False,
        )

        ref.add_argument(
            "--radius",
            type=float,
            dest="rad",
            help=(
                "Radius (Angstrom) of the regression refinement sphere."
                "This controls the size of the regions which are 'dragged'"
                "by the morphing calculation. Larger radii lead to bulk"
                "changes, smaller radii allow smaller features to move"
                "independently, at a cost of messing up the geometry."
                "Avoid radius < 2.5*resolution."
                "(Default = radius_scale * resolution of the cycle)"
            ),
            metavar="Radius",
            default=-1.0,
            required=False,
        )

        ref.add_argument(
            "--radius-scale",
            type=float,
            dest="radscl",
            help=(
                "Automatically set the radius in proportion to the resolution"
                "for the current cycle. The resolution is multiplied by this"
                "factor to get the radius. Overridden by radius."
                "(Default = 4.0)"
            ),
            metavar="Radius_scale",
            default=4.0,
            required=False,
        )

        ref.add_argument(
            "--coor-tol",
            type=float,
            dest="coor_tol",
            help=(
                "Coordinates RMSD tolerance for checking convergence."
                "(Default = 0.01)"
            ),
            metavar="Coord_tol",
            default=0.01,
            required=False,
        )

        ref.add_argument(
            "--coord",
            action="store_true",
            dest="refxyz",
            help=(
                "Perform coordinate refinement. If no refinement option is"
                "specified, coordinate refinement is enabled by default."
            ),
            required=False,
        )

        ref.add_argument(
            "--uiso",
            action="store_true",
            dest="refuiso",
            help="Perform B-factor refinement. (Default = False)",
            required=False,
        )

        ref.add_argument(
            "-sel",
            "--selection",
            dest="selection",
            type=str,
            nargs="*",
            help=(
                "Selected parts of the model will be refined."
                "Use Gemmi's selection syntax."
            ),
            required=False,
        )

        ref.add_argument(
            "--postrefine-u-iso",
            action="store_true",
            dest="postrefuiso",
            help="Perform post-refinement on B-factors. (Default = False)",
            required=False,
        )

        ref.add_argument(
            "--pseudo-regularise",
            choices=["no", "yes", "postref"],
            dest="pseudoreg",
            help=(
                "Pseudo-regularise the model. (Default = postref)"
                "no: turn off pseudo-regularise,"
                "yes: run at the end of the every cycle."
                "postref: run only at the very end of refinement."
            ),
            default="postref",
            required=False,
        )

        ref.add_argument(
            "--b-iso-range",
            type=float,
            nargs=2,
            dest="biso_range",
            help=(
                "Sets the lower and upper bound for B-isotropic value"
                "refinement. (Default = 0.1 999.9)"
            ),
            metavar=("blo", "bhi"),
            default=[0.1, 999.9],
            required=False,
        )

        misc = parser.add_argument_group("Others")
        misc.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s {pysheetbend} with Python {python}".format(
                pysheetbend=__version__, python=platform.python_version()
            ),
        )

        misc.add_argument(
            "--verbose",
            help="Sets verbosity of terminal output. Default = 0",
            type=int,
            dest="verbose",
            metavar="Verbosity",
            default=0,
            required=False,
        )

        misc.add_argument(
            "--write_cubic_map",
            help="Write internally made cubic map if input is not cubic",
            action="store_true",
            dest="cubicmap",
            required=False,
        )

    def parse_args(self, arg_list):
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        self.args = parser.parse_args(arg_list)
        return self.args

    def print_args(self):
        """
        Print input arguments
        """
        print("Input parameters: \n")
        for arg in vars(self.args):
            arg_val = getattr(self.args, arg)
            if isinstance(arg_val, list):
                print(f" {arg}: {arg_val}")
            if isinstance(arg_val, float):
                if (arg_val > 0.0) and (not None):
                    print(f" {arg}: {arg_val}")
            if isinstance(arg_val, int):
                if arg_val > 0:
                    print(f" {arg}: {arg_val}")
            if isinstance(arg_val, str):
                if arg_val is not None:
                    print(f" {arg}: {arg_val}")
        print("\n")

    def show_args(self):
        """
        Show input arguments
        """
        message = ""
        message += "Input parameters: \n"
        for arg in vars(self.args):
            arg_val = getattr(self.args, arg)
            if isinstance(arg_val, list):
                message += f" {arg}: {arg_val}\n"
            if isinstance(arg_val, float):
                if (arg_val > 0.0) and (not None):
                    message += f" {arg}: {arg_val}\n"
            if isinstance(arg_val, int):
                if arg_val > 0:
                    message += f" {arg}: {arg_val}\n"
            if isinstance(arg_val, str):
                if arg_val is not None:
                    message += f" {arg}: {arg_val}\n"
        message += "\n"
        return message
