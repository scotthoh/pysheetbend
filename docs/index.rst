.. pysheetbend documentation master file, created by
   sphinx-quickstart on Wed Sep  7 13:30:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
PYSHEETBEND
=======================================
A python implementation of SHEETBEND for cryo-EM data
=====================================================
Sheetbend - model morphing by shift field refinement
----------------------------------------------------
Description:
------------
*Sheetbend* performs model morphing by shift field refinement. This is a very fast form of preliminary refinement which can be applied
as a precursor to conventional refinement. It can be applied at any resolutions, and when used at low resolution is particularly suited
to correcting large shifts in secondary structure elements in molecular replacement. It can also be used to refine B-factors.

What is needed to run sheetbend for cryo-EM cases:
--------------------------------------------------
A structure file (PDB/mmCIF) and a map are required. The parameters in the structure file will be updated to better match the observations.
The performance of the calculation is controlled by two parameters, the resolution and the radius. For details, see:-

Cowtan, K., & Agirre, J. (2018) *Acta Cryst. D*\ **74**, 125-131. doi: `10.1107/S2059798320013170 <https://doi.org/10.1107/S2059798320013170>`_

For coordinate refinement, the calculation would normally start at a low resolution (e.g. >=6 Angstroms) and then increase to 3 or 4 Angstroms before moving to regular refinement.
B-value refinement may be usefully performed at higher resolutions, depending on the speed and fit required. The resolutions may be set for the whole calculation, or a list of resolutions may be provided describing how the resolution will change over multiple macro cycles.
If the radius is not specified, a default value will be calculated from 4x the resolution.
While little or no optimization has been performed the calculation is already fast, especially when run in low resolutions. Calculation time is dominated by the model to map and difference maps calculations, which is inefficient by modern standards. 

**Example for coordinate and B-value refinement**

.. code-block:: console

   $ pysheetbend refine_model_to_map2 --mapin MAP_PATH \
     --pdbin PDB_PATH --radius-scale 4.0 \
     --resolution RESOLUTION --res-by-cyc 6.0 3.0 \
     --cycles 12 --macro-cycles 3 --coord --uiso \
     --pseudo-regularise postref

* ``--radius-scale`` is the scale factor to calculate the radius value from the resolution
* ``--resolution`` is the resolution of the map
* ``--res-by-cycle`` is the starting and ending or list of resolutions for refinement. If ending resolution given is higher than --resolution, then the value from --resolution will be used as ending resolution.
* ``--cycles`` is the number of refinement cycles within each resolution cycle.
* ``--macro-cycles`` is the number of refinement cycles. The refinement resolution changes in each macro cycle.
* ``--coord`` to turn on coordinate refinement
* ``--uiso`` to turn on B-value refinement which runs at the end of each macro cycle
* ``--pseudo-regularise postref`` is to pseudo-regularise the model at the end of each macro cycle. Other options are *yes* to pseudo-regularise the model every cycle, or *no* for none.

Contents:
-----------------
.. toctree::
   :maxdepth: 2
   
   Introduction <self>
   installation
   shiftfield

