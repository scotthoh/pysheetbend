from __future__ import absolute_import, print_function, generators
import pipes
import platform
import sys
import argparse
import logging
import datetime
from pysheetbend import __version__
from pysheetbend.utils.logger import log2file
from pysheetbend.sheetbend_cmdln_parser import SheetbendParser
import pysheetbend.refine_xyz_to_map_resoloop
import pysheetbend.refine_map_to_map


def set_logger():
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    logging.basicConfig(
        level=logging.DEBUG,
        filename="pysheetbend.log",
        filemode="a",
        format="%(message)s",
    )
    log = logging.getLogger(__name__)
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    sys.stdout = log2file(log, logging.INFO, orig_stdout)
    sys.stderr = log2file(log, logging.ERROR, orig_stderr)


def dependencies_version():
    import gemmi
    import numpy
    import numba
    import scipy
    import pyfftw

    return dict(
        gemmi=gemmi.__version__,
        numpy=numpy.version.full_version,
        numba=numba.version_info.string,
        scipy=scipy.version.full_version,
        pyfftw=pyfftw.__version__,
    )


def write_header():
    print("{0}".format("#" * 15))
    print(f"# Pysheetbend vers. {__version__} Python {platform.python_version()}")
    print(
        f"# Libraries: {', '.join([x[0]+' '+x[1] for x in dependencies_version().items()])}"  # noqa:E501
    )
    print(f"# Job started : {datetime.datetime.now()}")
    print("# Command and arguments given :")
    print(f'# {" ".join(map(lambda x: pipes.quote(x), sys.argv[1:]))}')


def main():
    sb_parser = SheetbendParser()

    parser = argparse.ArgumentParser(
        prog="pysheetbend",
        description="Shift-field refinement of macromolecular atomic models for cryo-EM data",  # noqa=E501
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {pysheetbend} with Python {python}".format(
            pysheetbend=__version__, python=platform.python_version()
        ),
    )
    subparser = parser.add_subparsers(dest="command")
    # subparser.add_parser("refine_model_to_map")
    # subparser.add_parser("refine_map_to_map")
    # print(len(sys.argv))
    # print(sys.argv[1])
    # command = str(sys.argv[1])

    modes = dict(
        refine_model_to_map=pysheetbend.refine_xyz_to_map_resoloop,
        refine_map_to_map=pysheetbend.refine_map_to_map,
        # refine_model_to_map2=pysheetbend.refine_xyz_to_map_resoloop,
    )
    for n in modes:
        p = subparser.add_parser(n)
        sb_parser.add_args(p)

    args = parser.parse_args()

    if args.command in modes:
        set_logger()
        write_header()
        modes[args.command].main(args)
        print(f"# Job finished : {datetime.datetime.now()}")
    else:
        print("\nPlease specify subcommand!")
        # print(modes.keys())
        print(f"Subcommand: {', '.join([x for x in modes.keys()])}\n")


if __name__ == "__main__":
    main()
