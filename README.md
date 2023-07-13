## Python virtual acoustic room simulator


### Python re-implementation of a MATLAB virtual acoustic room simulator (revised and ported by msaddler 2023/07)
```
simulator.py
```
Required Python packages: `numpy`, `scipy`, `soundfile`


### Python interface for directly calling the MATLAB virtual acoustic room simulator by Mike O'Connell and Jay Desloge.
```
simulator_matlab.py
```
Requires the MATLAB engine for Python to use (Conda environment: `/om2/user/msaddler/.conda/envs/matlab`)

MATLAB source code was copied from Andrew Francl: `/om/user/francl/Room_Simulator_20181115_Rebuild`. MATLAB files were minimally modified by msaddler to standardize formatting and enable interfacing with Python (the `meas_files` variable was changed from a char array to a cell array such that it could be passed to MATLAB as a list of strings).


### HRTF measurements of a KEMAR dummy-head microphone

The `kemar_hrtfs` included as the default option in this repository were measured by Bill Gardner and Keith Martin (Copyright 1994 by the MIT Media Laboratory). The compact wav files and documentation were downloaded from [https://sound.media.mit.edu/resources/KEMAR.html](https://sound.media.mit.edu/resources/KEMAR.html).
