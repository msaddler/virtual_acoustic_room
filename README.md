## Python virtual acoustic room simulator

This virtual acoustic room simulator uses the [Image-Source Method](https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-image-source-method-and-hrtf-interpolation.html#mw_rtc_RoomImpulseResponseImageSourceExample_M_FDE78C42) to render binaural room impulse responses (BRIRs) for spatializing audio in simple "shoebox" (cuboid) rooms. This is a Python port of [MATLAB code](src) by Mike O'Connell and Jay Desloge. We used this Python version to generate a large set of BRIRs for training [deep neural network models of human sound localization](https://doi.org/10.1038/s41467-024-54700-5).


### Requirements

- The Python implementation (`simulator.py`) requires only: `numpy`, `pandas`, `scipy`, `soundfile`

- The Python interface for calling the MATLAB implementation (`simulator_matlab.py`) requires the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)


### Conventions

- The `room_dim_xyz` argument specifies the `[X, Y, Z]` dimensions of a room with one corner at the origin.
- The `x-y` plane with `z=0` corresponds to the room's floor and `z=Z` corresponds to the room's ceiling.
- The `room_materials` argument is an ordered list of 6 integers corresponding to 4 walls, a floor, and a ceiling: `[x=0 wall, x=X wall, y=0 wall, y=Y wall, z=0 floor, z=Z ceiling]`. The integers correspond to materials defined in [`materials_original.csv`](materials_original.csv).
- 0 degrees azimuth is defined as parallel to a vector along the x-axis: `[1, 0, 0]`.
- **Positive azimuths indicate counter-clockwise rotation away from the x-axis in the `x-y` plane**. Note this azimuth convention is opposite to that used by Gardner & Martin to label the KEMAR HRTFs. The `simulator.get_brir` function handles this mismatch internally. When using the function, a source azimuth of +90 degrees is to the listener's left and -90 degrees is to the listener's right.
- Positive elevations indicate upward rotation away from the floor (`z=0` plane).


### Example usage

```
import simulator

room_materials = [1, 1, 1, 1, 15, 16]
room_dim_xyz = [10.0, 10.0, 3.0]  # X, Y, and Z dimensions of room in m
head_pos_xyz = [5.0, 5.0, 2.0]  # X, Y, and Z coordinates of head in m
head_azim = 0  # Head azimuth in degrees
src_azim = 75  # Source azimuth in degrees
src_elev = 0  # Source elevation in degrees
src_dist = 1.4  # Source distance in m

sr = 44100  # Sampling rate in Hz to match KEMAR HRTFs
dur = 0.5  # Duration of BRIR in seconds

brir = simulator.get_brir(
    room_materials=room_materials,
    room_dim_xyz=room_dim_xyz,
    head_pos_xyz=head_pos_xyz,
    head_azim=head_azim,
    src_azim=src_azim,
    src_elev=src_elev,
    src_dist=src_dist,
    sr=sr,
    dur=dur,
)
print(brir.shape)  # --> [22050 timesteps, 2 channels = left and right ear]
```


### Contents

```
|__ DEMO.ipynb (START HERE)

|__ simulator.py (Python implementation -- revised and ported by Mark Saddler, 2023/07)

|__ simulator_matlab.py (Python interface for calling MATLAB simulator -- Mark Saddler, 2023/07)

|__ kemar_hrtfs (HRTF measurements of a KEMAR dummy-head microphone -- Gardner & Martin, 1994)

|__ materials_original.csv (absorption coefficients for different wall materials)

|__ src (MATLAB source code by Mike O'Connell and Jay Desloge)
    |__ acoeff_hrtf.m
    |__ impulse_generate_hrtf.m
    |__ shapedfilter_hrtf.m
    |__ vary_stim_env_hrir_sweep.m (script for generating set of BRIRs by Andrew Francl)

|__ generate_brir_manifest.py
|__ generate_brir_dataset.py      (code for Saddler & McDermott, 2024 Nature Communications)
|__ generate_brir_dataset_job.sh
```


### HRTF measurements of a KEMAR dummy-head microphone

The `kemar_hrtfs` included in this repository were measured by Bill Gardner and Keith Martin (Copyright 1994 by the MIT Media Laboratory). The compact wav files and documentation were downloaded from [https://sound.media.mit.edu/resources/KEMAR.html](https://sound.media.mit.edu/resources/KEMAR.html).


### Contact

Mark R. Saddler (msaddler@mit.edu / marksa@dtu.dk)
