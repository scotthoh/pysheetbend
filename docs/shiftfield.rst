Running shift-field refinement
==============================

**Example for coordinate only**
The following command and keywords will run shift-field coordinate refinement on the input model.
The resulting model of the refinement will have the default filename of **shiftfield_refined_final.pdb**.

.. code-block:: console

   $ pysheetbend refine_model_to_map2 --mapin MAP_PATH \
     --pdbin PDB_PATH --radius-scale 4.0 \
     --resolution RESOLUTION --res-by-cyc 6.0 3.0 \
     --cycle 12 --macro-cycles 3 --coord \
     --pseudo-regularise postref

* ``--radius-scale`` is the scale factor to calculate the radius value from the resolution
* ``--resolution`` is the resolution of the map
* ``--res-by-cycle`` is the starting and ending or list of resolutions for refinement. If ending resolution given is higher than --resolution, then the value from --resolution will be used as ending resolution.
* ``--cycles`` is the number of refinement cycles within each resolution cycle.
* ``--macro-cycles`` is the number of refinement cycles. The refinement resolution changes in each macro cycle.
* ``--coord`` to turn on coordinate refinement
* ``--pseudo-regularise postref`` is to pseudo-regularise the model at the end of each macro cycle. Other options are *yes* to pseudo-regularise the model every cycle, or *no* for none.

**Example for B-values only**
The following command and keywords will run shift-field atomic B-value refinement on the input model.
The resulting model of the refinement will have the default filename of **shiftfield_refined_final.pdb**.

.. code-block:: console

   $ pysheetbend refine_model_to_map2 --mapin MAP_PATH \
     --pdbin PDB_PATH --radius-scale 4.0 \
     --resolution RESOLUTION --res-by-cyc 6.0 3.0 \
     --macro-cycles 3 --uiso

* ``--radius-scale`` is the scale factor to calculate the radius value from the resolution
* ``--resolution`` is the resolution of the map
* ``--res-by-cycle`` is the starting and ending or list of resolutions for refinement. If ending resolution given is higher than --resolution, then the value from --resolution will be used as ending resolution.
* ``--macro-cycles`` is the number of refinement cycles. The refinement resolution changes in each macro cycle.
* ``--uiso`` to turn on B-value refinement

Keyworded input:
----------------
**Input files and related keywords**

* ``--mapin`` Input map file
* ``--pdbin`` Input PDB/mmCIF file containing the model to be refined
* ``--maskin`` [optional] Input mask.
* ``--nomask`` [optional] Do not mask input map. If specified, program will not mask input map for refinement. 
* ``--no-ligands`` [optional] Remove ligands. If specified, ligands will not be read from input file.
* ``--no-waters`` [optional] Remove waters. If specified, waters will not be read from input file.

**Output files and related keywords**

* ``--pdbout`` [optional] Output filename. Will use the given filename as prefix to the final output filename. Default: shiftfield.pdb
* ``--xmlout`` [optional] XML filename. Default: program.xml
* ``--no-xmlout`` [optional] Do not write XML output. If specified no XML output will be written. 
* ``--intermediate`` [optional] Output model files every cycle.

**Refinement parameters**

* ``--resolution`` Resolution of the input map. This value will be used to calculate values of sample rate and grid spacing used for calculating resampling grids. If **res-by-cycle** is not given, refinement will run at this resolution only.
* ``--res-by-cycle`` Set the resolution for each macro cycle. Resolutions are separated by space. If there are fewer resolutions than macro-cycles, linear interpolation is used to fill in the remaining values.\
* ``--cycles`` Number of cycles within each resolution/macro cycle. Default: 1
* ``--macro-cycles`` Number of resolution/macro cycles. Default: 1
* ``--radius`` *radius* 