## MATLAB source code for virtual acoustic room simulator

MATLAB virtual acoustic room simulator code (Mike O'Connell and Jay Desloge):
```
room_impulse_hrtf.m
impulse_generate_hrtf.m
shapedfilter_hrtf.m
acoeff_hrtf.m
```

MATLAB script for generating a set of binaural room impulse responses (Andrew Francl):
```
vary_stim_env_hrir_sweep.m
```

All MATLAB source code was copied from Andrew Francl's files on Openmind: `/om/user/francl/Room_Simulator_20181115_Rebuild`. MATLAB files were minimally modified by msaddler to standardize formatting and enable interfacing with Python (e.g., the `meas_files` variable was changed from a char array to a cell array such that it could be passed to MATLAB from Python as a list of strings).
