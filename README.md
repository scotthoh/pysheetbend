# PYSHEETBEND
## A python implementation of SHEETBEND.
### Sheetbend - model morphing by shift field refinement

### Description:
*Sheetbend* performs model morphing by shift field refinement. This is a very fast form of preliminary refinement which can be applied
as a precursor to conventional refinement. It can be applied at any resolutions, and when used at low resolution is particularly suited
to correcting large shifts in secondary structure elements in molecular replacement. It can also be used to refine B-factors.

### How to run sheetbend for cryo-EM cases:
A structure file (PDB/mmCIF) and a map are required. The parameters in the structure file will be updated to better match the observations.

For coordinate refinement, the calculation would normally start at a low resolution (e.g. ~6 Angstroms) and then increase to 3 or 4 Angstroms before moving to regular refinement. B-value refinement may be usefully performed at higher resolutions, depending on the speed and fit required. The resolutions may be set for the whole calculation, or a list of resolutions may be provided describing how the resolution will change over multiple cycles. If the radius is not specified, a default value will be calculated from 4x the resolution.

While little or no optimization has been performed the calculation is already fast, especially when run in low resolutions. Calculation time is dominated by the linear algebra calculation to get the shifts (in the case for this python code), electron density map and difference maps calculations. 


### Authors:
Kevin Cowtan (sheetbend)
Paul Bond (sheetbend)
Stephen Metcalfe (sheetbend)
Soon Wen Hoh (sheetbend - python code)

### References:
Cowtan, K., & Agirre, J. (2018) Acta Cryst. D74, 125-131.
Cowtan, K., Metcalfe, S. & Bond, P. (2020). Acta Cryst. D76, 1192-1200.

Copyright 2018-2021 Kevin Cowtan & University of York all rights reserved

Updated - 4th Feb 2022 (SW Hoh)
