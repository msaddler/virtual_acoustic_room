import os
import sys
import time
import scipy.io
import numpy as np
import matlab.engine


"""
The MATLAB function `acoeff_hrtf.m` maps materials to acoustic absorption coefficients.
This dictionary maps integer codes to materials for which coefficients are available.
"""
map_int_to_material = {
    # WALLS
    1: 'Brick',
    2: 'Concrete, painted',
    3: 'Window Glass',
    4: 'Marble',
    5: 'Plaster on Concrete',
    6: 'Plywood',
    7: 'Concrete block, coarse',
    8: 'Heavyweight drapery',
    9: 'Fiberglass wall treatment, 1 in',
    10: 'Fiberglass wall treatment, 7 in',
    11: 'Wood panelling on glass fiber blanket',
    # FLOORS
    12: 'Wood parquet on concrete',
    13: 'Linoleum',
    14: 'Carpet on concrete',
    15: 'Carpet on foam rubber padding',
    # CEILINGS
    16: 'Plaster, gypsum, or lime on lath',
    17: 'Acoustic tiles, 0.625", 16" below ceiling',
    18: 'Acoustic tiles, 0.5", 16" below ceiling',
    19: 'Acoustic tiles, 0.5" cemented to ceiling',
    20: 'Highly absorptive panels, 1", 16" below ceiling',
    # OTHERS
    21: 'Upholstered seats',
    22: 'Audience in upholstered seats',
    23: 'Grass',
    24: 'Soil',
    25: 'Water surface',
    26: 'Anechoic',
    27: 'Uniform (0.6) absorbtion coefficient',
    28: 'Uniform (0.2) absorbtion coefficient',
    29: 'Uniform (0.8) absorbtion coefficient',
    30: 'Uniform (0.14) absorbtion coefficient',
    31: 'Artificial - absorbs more at high freqs',
    32: 'Artificial with absorption higher in middle ranges',
    33: 'Artificial - absorbs more at low freqs',
}


def room_impulse_hrtf(
        src_loc,
        head_center,
        head_azimuth,
        walls,
        wtypes,
        meas_sym=1,
        f_samp=44100,
        c_snd=344.5,
        num_taps=22050,
        log_dist=0,
        jitter=1,
        highpass=1,
        dsply=0,
        eng=None):
    """
    Python wrapper function around MATLAB function `room_impulse_hrtf.m`.
    
    TODO:   `meas_locs` and `meas_files` are hard-coded here, which currently
            limits this room simulator to use a single set of measured HRTFs.
    """
    meas_locs = eng.load('HRTFs/data_locs.mat')['locs_gardnermartin']
    meas_files = eng.cellstr(scipy.io.loadmat('HRTFs/file_names.mat')['gardnermartin_file'].tolist())
    meas_delay = (np.sqrt(np.sum(np.square(src_loc - head_center))) / c_snd) * np.ones(len(meas_files))
    h_out, lead_zeros = eng.room_impulse_hrtf(
        matlab.double(src_loc.reshape([1, 3]).tolist()),
        matlab.double(head_center.reshape([1, 3]).tolist()),
        matlab.double([head_azimuth]),
        meas_locs,
        meas_files,
        matlab.double(meas_delay.reshape([len(meas_files), 1]).tolist()),
        matlab.double([meas_sym]),
        matlab.double(walls.reshape([1, 3]).tolist()),
        matlab.double(np.array([wtypes]).tolist()),
        matlab.double([f_samp]),
        matlab.double([c_snd]),
        matlab.double([num_taps]),
        matlab.double([log_dist]),
        matlab.double([jitter]),
        matlab.double([highpass]),
        matlab.double([dsply]),
        nargout=2)
    h_out = np.array(h_out)
    lead_zeros = np.array(lead_zeros)
    return h_out, lead_zeros


def is_valid_position(point, walls, buffer=0):
    """
    Helper function to check if a position is inside the room walls.
    """
    assert len(point) == len(walls)
    for p, w in zip(point, walls):
        if p < buffer or p > w - buffer:
            return False
    return True


def get_brir(room_materials=[26, 26, 26, 26, 26, 26],
             room_dim_xyz=[10, 10, 3],
             head_pos_xyz=[5, 5, 1.5],
             head_azim=0,
             src_azim=0,
             src_elev=0,
             src_dist=1,
             buffer_pos=0,
             sr=50000,
             dur=0.5,
             kwargs_room_impulse_hrtf={},
             incorporate_lead_zeros=True,
             verbose=True,
             eng=None):
    """
    Main function to generate binaural room impulse response (BRIR) from
    a room description, a listener position, and a source position.
    """
    assert head_azim >= -135, "head_azim < -135 degrees cause unexpected behavior"
    room_materials = np.array(room_materials)
    assert room_materials.shape == (6,), "room_materials shape: [wall_x0, wall_x, wall_y0, wall_y, floor, ceiling]"
    room_dim_xyz = np.array(room_dim_xyz)
    assert room_dim_xyz.shape == (3,), "room_dim_xyz shape: [x_len (length), y_len (width), z_len (height)]"
    head_pos_xyz = np.array(head_pos_xyz)
    assert head_pos_xyz.shape == (3,), "head_pos_xyz shape: [x_head, y_head, z_head]"
    src_pos_xyz = np.array([
        src_dist * np.cos(np.deg2rad(src_elev)) * np.cos(np.deg2rad(src_azim + head_azim)) + head_pos_xyz[0],
        src_dist * np.cos(np.deg2rad(src_elev)) * np.sin(np.deg2rad(src_azim + head_azim)) + head_pos_xyz[1],
        src_dist * np.sin(np.deg2rad(src_elev)) + head_pos_xyz[2],
    ])
    assert is_valid_position(head_pos_xyz, room_dim_xyz, buffer=buffer_pos), "Invalid head position"
    assert is_valid_position(src_pos_xyz, room_dim_xyz, buffer=buffer_pos), "Invalid source position"
    if verbose:
        print("[room simulator] head_pos: {}, src_pos: {}, room_dim: {}".format(
            head_pos_xyz.tolist(),
            src_pos_xyz.tolist(),
            room_dim_xyz.tolist()))
    t0 = time.time()
    h_out, lead_zeros = room_impulse_hrtf(
        src_loc=src_pos_xyz,
        head_center=head_pos_xyz,
        head_azimuth=-head_azim, # `room_impulse_hrtf` convention is positive azimuth = clockwise
        walls=room_dim_xyz,
        wtypes=room_materials,
        f_samp=sr,
        num_taps=int(dur * sr),
        **kwargs_room_impulse_hrtf,
        eng=eng)
    if verbose:
        print(f'[room simulator] time elapsed: {time.time() - t0} seconds')
    if incorporate_lead_zeros:
        lead_zeros = int(np.round(lead_zeros))
        print(f'[room simulator] incorporated {lead_zeros} leading zeros')
        if lead_zeros >= 0:
            h_out = np.pad(h_out, ((lead_zeros, 0), (0, 0)))
            brir = h_out[:int(dur * sr)]
        else:
            h_out = np.pad(h_out, ((0, -lead_zeros), (0, 0)))
            brir = h_out[-lead_zeros:]
    else:
        brir = h_out
    return brir


def sample_room_parameters(
        p_outdoor=0.25,
        p_outdoor_wall=0.25,
        range_room_x=[3, 30],
        range_room_y=[3, 30],
        range_room_z=[2.5, 10],
        list_material_outdoor_floor=[1, 7, 23, 24, 25],
        list_material_outdoor_wall=[1, 2, 3, 4, 5, 6, 7, 23, 24],
        list_material_indoor_floor=[1, 2, 4, 6, 12, 13, 14, 15],
        list_material_indoor_wall=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        list_material_indoor_ceiling=[3, 6, 16, 17, 18, 19, 20],
        list_n_unique_wall_materials=[1, 2, 3, 4],
        verbose=True):
    """
    Helper function for randomly sampling room parameters (dimensions and materials).
    """
    ANECHOIC = 26
    is_outdoor = np.random.choice([1, 0], p=[p_outdoor, 1-p_outdoor])
    x_len = np.exp(np.random.uniform(low=np.log(range_room_x[0]), high=np.log(range_room_x[1]), size=None))
    y_len = np.exp(np.random.uniform(low=np.log(range_room_y[0]), high=np.log(range_room_y[1]), size=None))
    z_len = np.exp(np.random.uniform(low=np.log(range_room_z[0]), high=np.log(range_room_z[1]), size=None))
    if is_outdoor:
        material_floor = np.random.choice(list_material_outdoor_floor)
        material_ceiling = np.random.choice([ANECHOIC])
        material_wall = np.random.choice(
            [-1, ANECHOIC],
            size=4,
            replace=True,
            p=[p_outdoor_wall, 1-p_outdoor_wall])
        IDX_outdoor_wall = material_wall < 0
        material_wall[IDX_outdoor_wall] = np.random.choice(
            list_material_outdoor_wall,
            size=IDX_outdoor_wall.sum(),
            replace=True)
    else:
        material_floor = np.random.choice(list_material_indoor_floor)
        material_ceiling = np.random.choice(list_material_indoor_ceiling)
        n_unique_wall_materials = np.random.choice(list_n_unique_wall_materials)
        material_wall = np.random.choice(
            list_material_indoor_wall,
            size=n_unique_wall_materials,
            replace=False)
        if len(material_wall < 4):
            material_wall = np.concatenate([
                material_wall,
                np.random.choice(material_wall, size=4-len(material_wall), replace=True),
            ])
    room_parameters = {
        'room_materials': list(material_wall) + [material_floor, material_ceiling],
        'room_dim_xyz': [x_len, y_len, z_len],
        'is_outdoor': is_outdoor,
        'material_wall_xmin': map_int_to_material[material_wall[0]],
        'material_wall_xmax': map_int_to_material[material_wall[1]],
        'material_wall_ymin': map_int_to_material[material_wall[2]],
        'material_wall_ymax': map_int_to_material[material_wall[3]],
        'material_wall_zmin': map_int_to_material[material_floor],
        'material_wall_zmax': map_int_to_material[material_ceiling],
    }
    if verbose:
        print(f"[sample_room_parameters]")
        for k in room_parameters.keys():
            print(f"|__ {k}: {room_parameters[k]}")
    return room_parameters


def sample_head_parameters(
        room_dim_xyz=[10, 10, 3],
        buffer_pos=1.4,
        range_src_elev=[-60, 60],
        range_head_azim=[0, 90],
        range_head_z=[1.2, 2.4],
        verbose=True):
    """
    Helper function for randomly sampling head parameters (position and azimuth).
    """
    min_z = buffer_pos * np.sin(np.deg2rad(np.max(np.abs(range_src_elev))))
    min_z = max(range_head_z[0], min_z)
    max_z = min(range_head_z[1], room_dim_xyz[2] - min_z)
    assert (min_z <= max_z) and (min_z >= 0), "invalid range of z values for head position"
    head_pos_xyz = np.array([
        np.random.uniform(low=buffer_pos, high=room_dim_xyz[0]-buffer_pos),
        np.random.uniform(low=buffer_pos, high=room_dim_xyz[1]-buffer_pos),
        np.random.uniform(low=min_z, high=max_z),
    ])
    head_azim = np.random.uniform(low=range_head_azim[0], high=range_head_azim[1])
    head_parameters = {
        'head_pos_xyz': head_pos_xyz,
        'head_azim': head_azim,
    }
    if verbose:
        print(f"[sample_head_parameters] (head_z sampled uniformly from {[min_z, max_z]})")
        for k in head_parameters.keys():
            print(f"|__ {k}: {head_parameters[k]}")
    return head_parameters


def distance_to_wall(room_dim_xyz, head_pos_xyz, head_azim, src_azim, src_elev):
    """
    Helper function to find maximum possible source distance given room dimensions,
    head position, head azimuth, source azimuth (relative to head), and source
    elevation (relative to head).
    """
    azim = head_azim + src_azim
    elev = src_elev
    while azim < 0:
        azim += 360
    azim = azim % 360
    quadrant = int(azim / 90) + 1
    if quadrant == 1:
        rx = (room_dim_xyz[0] - head_pos_xyz[0]) / (np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
        ry = (room_dim_xyz[1] - head_pos_xyz[1]) / (np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
    elif quadrant == 2:
        rx = -head_pos_xyz[0] / (np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
        ry = (room_dim_xyz[1] - head_pos_xyz[1]) / (np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
    elif quadrant == 3:
        rx = -head_pos_xyz[0] / (np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
        ry = -head_pos_xyz[1] / (np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
    elif quadrant == 4:
        rx = (room_dim_xyz[0] - head_pos_xyz[0]) / (np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
        ry = -head_pos_xyz[1] / (np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)))
    else:
        raise ValueError('INVALID ANGLE')
    if elev > 0:
        rz = (room_dim_xyz[2] - head_pos_xyz[2]) / np.sin(np.deg2rad(elev))
    elif elev < 0:
        rz = -head_pos_xyz[2] / np.sin(np.deg2rad(elev))
    else:
        rz = np.inf
    return min(rx, ry, rz)
