## Python virtual acoustic room simulator

This virtual acoustic room simulator uses the [Image-Source Method](https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-image-source-method-and-hrtf-interpolation.html#mw_rtc_RoomImpulseResponseImageSourceExample_M_FDE78C42) to render binaural room impulse responses (BRIRs) for spatializing audio in simple "shoebox" (cuboid) rooms.


### Contents
```
|__ src (MATLAB source code by Mike O'Connell and Jay Desloge)
    |__ acoeff_hrtf.m
    |__ impulse_generate_hrtf.m
    |__ shapedfilter_hrtf.m
    |__ vary_stim_env_hrir_sweep.m (script for generating set of BRIRs by Andrew Francl)

|__ simulator.py (Python implementation -- revised and ported by Mark Saddler, 2023/07)

|__ simulator_matlab.py (Python interface for calling MATLAB simulator -- Mark Saddler, 2023/07)

|__ DEMO.ipynb (START HERE)

|__ kemar_hrtfs (HRTF measurements of a KEMAR dummy-head microphone -- Gardner & Martin, 1994)
```


### Requirements

- The pure Python implementation (`simulator.py`) requires: `numpy`, `scipy`, `soundfile`

- The Python interface for the MATLAB implementation (`simulator_matlab.py`) requires the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
    - Existing Conda environment on Openmind: `/om2/user/msaddler/.conda/envs/matlab`


### HRTF measurements of a KEMAR dummy-head microphone

The `kemar_hrtfs` included in this repository were measured by Bill Gardner and Keith Martin (Copyright 1994 by the MIT Media Laboratory). The compact wav files and documentation were downloaded from [https://sound.media.mit.edu/resources/KEMAR.html](https://sound.media.mit.edu/resources/KEMAR.html).
