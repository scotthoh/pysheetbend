from __future__ import absolute_import, print_function
import setuptools
import pysheetbend
import shutil
import os

if os.path.exists("build") is True:
    print("build exists")
    shutil.rmtree("./build")

setuptools.setup(
    name="pysheetbend",
    version=pysheetbend.__version__,
    author="Soon Wen Hoh and Kevin Cowtan",
    url="n",
    description="Shift-field refinement of macrocmolecular atomic models for cryo-EM data",
    license="LGPL-2.1",
    packages=setuptools.find_packages(),
    python_requires="~=3.8",
    install_requires=["numpy>=1.15", "scipy", "gemmi>=0.5.3"],
    entry_points={
        "console_scripts": ["pysheetbend = pysheetbend.sheetbend:main"],
    },
)

# package_dir={"pysheetbend": "pysheetbend"},
