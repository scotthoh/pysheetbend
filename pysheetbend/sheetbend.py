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

# from . import __version__
import pysheetbend.refine_map_to_map
import pysheetbend.refine_xyz_to_map
import pysheetbend.refine_xyz_to_map_resoloop

orig_stdout = sys.stdout
orig_stderr = sys.stderr
logging.basicConfig(
    level=logging.DEBUG, filename="pysheetbend.log", filemode="a", format="%(message)s"
)
log = logging.getLogger(__name__)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
sys.stdout = log2file(log, logging.INFO, orig_stdout)
sys.stderr = log2file(log, logging.ERROR, orig_stderr)


def main():
    sb_parser = SheetbendParser()

    parser = argparse.ArgumentParser(
        prog='PySheetbendEM',
        description='Shift-field refinement of macromolecular atomic models for cryo-EM data',
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
        refine_model_to_map=pysheetbend.refine_xyz_to_map,
        refine_map_to_map=pysheetbend.refine_map_to_map,
        refine_model_to_map2=pysheetbend.refine_xyz_to_map_resoloop,
    )
    for n in modes:
        p = subparser.add_parser(n)
        sb_parser.add_args(p)

    args = parser.parse_args()
    print('{0}'.format('#' * 15))
    print(f'# Job started : {datetime.datetime.now()}')
    print('# Command and arguments given :')
    print(f'# {" ".join(map(lambda x: pipes.quote(x), sys.argv[1:]))}')
    if args.command == 'refine_model_to_map':
        modes[args.command].main(args)  # sb_args)
    if args.command == 'refine_map_to_map':
        modes[args.command].main(args)  # sb_args)
    if args.command == 'refine_model_to_map2':
        modes[args.command].main(args)  # sb_args)
    print(f"# Job finished : {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
