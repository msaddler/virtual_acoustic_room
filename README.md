## Python virtual acoustic room simulator

### Python re-implementation of a MATLAB virtual acoustic room simulator (revised and ported by msaddler 2023/07)
```
simulator.py
```
Required Python packages: `numpy`, `scipy`, `soundfile`

### Python interface for directly calling the MATLAB virtual acoustic room simulator by Mike O'Connell and Jay Desloge.
```
simulator_matlab_engine.py
```
Requires the MATLAB engine for Python to use (Conda environment: `/om2/user/msaddler/.conda/envs/matlab`)

MATLAB source code and HRTFs were copied from Andrew Francl: `/scratch2/weka/mcdermott/francl/Room_Simulator_20181115_Rebuild`. MATLAB files were minimally modified by msaddler to standardize formatting and enable interfacing with Python (one variable was changed from a char array to a cell array of strings).
