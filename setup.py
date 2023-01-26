from __future__ import absolute_import, print_function
import setuptools
import pysheetbend
import shutil
import os

if os.path.exists("build") is True:
    print("build exists, removing directory and rebuild...")
    shutil.rmtree("./build")

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="pysheetbend",
    version=pysheetbend.__version__,
    author="Soon Wen Hoh and Kevin Cowtan",
    author_email="soonwen.hoh@york.ac.uk",
    description="Shift-field refinement of macromolecular atomic models for cryo-EM data",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/scotth2o/pysheetbend",
    packages=setuptools.find_packages(),
    # include_package_data=True,
    license="LGPL-2.1",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires="~=3.8",
    install_requires=[
        "numpy>1.15",
        "gemmi>=0.5.7",
        "numba>=0.53.1",
        "scipy",
        "ccpem-utils",
        "pyfftw",
    ],
    entry_points={
        "console_scripts": ["pysheetbend = pysheetbend.sheetbend:main"],
    },
)
