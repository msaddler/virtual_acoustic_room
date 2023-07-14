import os
import sys
import functools
import multiprocessing
import numpy as np
import pandas as pd

import simulator


def sample_room_parameters(
        p_outdoor=0.25,
        p_outdoor_wall=0.25,
        range_room_x=[3, 30],
        range_room_y=[3, 30],
        range_room_z=[2.2, 10],
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
        'material_x0': simulator.map_int_to_material[material_wall[0]],
        'material_x1': simulator.map_int_to_material[material_wall[1]],
        'material_y0': simulator.map_int_to_material[material_wall[2]],
        'material_y1': simulator.map_int_to_material[material_wall[3]],
        'material_z0': simulator.map_int_to_material[material_floor],
        'material_z1': simulator.map_int_to_material[material_ceiling],
    }
    if verbose:
        print(f"[sample_room_parameters]")
        for k in room_parameters.keys():
            print(f"|__ {k}: {room_parameters[k]}")
    return room_parameters


def sample_head_parameters(
        room_dim_xyz=[3, 3, 2.2],
        buffer=1.45,
        range_src_elev=[-40, 60],
        range_head_azim=[0, 90],
        range_head_z=[0, 2],
        verbose=True):
    """
    Helper function for randomly sampling head parameters (position and azimuth).
    """
    assert room_dim_xyz[0] >= 2 * buffer, "invalid range of x values for head position"
    assert room_dim_xyz[1] >= 2 * buffer, "invalid range of y values for head position"
    min_z = max(range_head_z[0], buffer * np.sin(np.deg2rad(-range_src_elev[0])))
    max_z = min(range_head_z[1], room_dim_xyz[2] - buffer * np.sin(np.deg2rad(range_src_elev[1])))
    assert (min_z <= max_z) and (min_z >= 0), "invalid range of z values for head position"
    head_pos_xyz = np.array([
        np.random.uniform(low=buffer, high=room_dim_xyz[0]-buffer),
        np.random.uniform(low=buffer, high=room_dim_xyz[1]-buffer),
        np.random.uniform(low=min_z, high=max_z),
    ])
    head_azim = np.random.uniform(low=range_head_azim[0], high=range_head_azim[1])
    head_parameters = {
        'head_pos_xyz': head_pos_xyz,
        'head_azim': head_azim,
    }
    if verbose:
        print(f"[sample_head_parameters]")
        print(f"|__ head_pos_x sampled uniformly from {[buffer, room_dim_xyz[0]-buffer]})")
        print(f"|__ head_pos_y sampled uniformly from {[buffer, room_dim_xyz[1]-buffer]})")
        print(f"|__ head_pos_z sampled uniformly from {[min_z, max_z]})")
        print(f"|__ head_azim sampled uniformly from {range_head_azim})")
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


def get_df_brir(
        dfi_room,
        list_src_dst=[1.4, None],
        list_src_azim=np.arange(0, 360, 5),
        list_src_elev=np.arange(-40, 61, 10)):
    """
    """
    list_d = []
    index_brir = 0
    for src_dist in list_src_dst:
        for src_azim in list_src_azim:
            for src_elev in list_src_elev:
                if src_dist is None:
                    src_dist_to_wall = distance_to_wall(
                        room_dim_xyz=dfi_room.room_dim_xyz,
                        head_pos_xyz=dfi_room.head_pos_xyz,
                        head_azim=dfi_room.head_azim,
                        src_azim=src_azim,
                        src_elev=src_elev)
                    min_src_dist = 1.0
                    max_src_dist = src_dist_to_wall - 0.1
                    assert max_src_dist >= min_src_dist
                    r = np.random.uniform(low=min_src_dist, high=max_src_dist)
                else:
                    r = src_dist
                head_azim = dfi_room.head_azim
                head_pos_xyz = dfi_room.head_pos_xyz
                src_pos_xyz = np.array([
                    r * np.cos(np.deg2rad(src_elev)) * np.cos(np.deg2rad(src_azim + head_azim)) + head_pos_xyz[0],
                    r * np.cos(np.deg2rad(src_elev)) * np.sin(np.deg2rad(src_azim + head_azim)) + head_pos_xyz[1],
                    r * np.sin(np.deg2rad(src_elev)) + head_pos_xyz[2],
                ])
                assert simulator.is_valid_position(head_pos_xyz, dfi_room.room_dim_xyz, buffer=0)
                assert simulator.is_valid_position(src_pos_xyz, dfi_room.room_dim_xyz, buffer=0)
                d = {
                    'index_brir': index_brir,
                    'index_room': dfi_room.index_room,
                    'room_materials': dfi_room.room_materials,
                    'room_dim_xyz': dfi_room.room_dim_xyz,
                    'head_pos_xyz': dfi_room.head_pos_xyz,
                    'head_azim': dfi_room.head_azim,
                    'src_azim': src_azim,
                    'src_elev': src_elev,
                    'src_dist': r,
                    'buffer': 0,
                    'sr': 44100,
                    'c': 344.5,
                    'dur': 0.5,
                    'use_hrtf_symmetry': True,
                    'use_log_distance': False,
                    'use_jitter': True,
                    'use_highpass': True,
                    'incorporate_lead_zeros': True,
                }
                list_d.append(d)
                index_brir = index_brir + 1
    df_brir = pd.DataFrame(list_d)
    return df_brir


def main(
        fn_manifest_room, 
        fn_manifest_brir,
        n_room=2000,
        list_src_dist=[1.4, None],
        processes=20):
    """
    """
    print(f"Sampling {n_room} rooms / head positions")
    list_df_room = []
    for index_room in range(n_room):
        # Use `index_room` as random seed for sampling room and head parameters
        np.random.seed(index_room)
        room_parameters = sample_room_parameters(verbose=False)
        head_parameters = sample_head_parameters(room_dim_xyz=room_parameters['room_dim_xyz'], verbose=False)
        d = {'index_room': index_room}
        d.update(room_parameters)
        d.update(head_parameters)
        list_df_room.append(d)
    df_room = pd.DataFrame(list_df_room).sort_index(axis=1)
    df_room.to_pickle(fn_manifest_room)
    print(f"Wrote `df_room` ({len(df_room)} rooms / head positions):\n{fn_manifest_room}")
    
    print(f"Sampling BRIR metadata for {n_room} rooms / head positions")
    f = functools.partial(
        get_df_brir,
        list_src_dst=list_src_dist,
        list_src_azim=np.arange(0, 360, 5),
        list_src_elev=np.arange(-40, 61, 10))
    list_dfi_room = [df_room.iloc[_] for _ in range(len(df_room))] 
    with multiprocessing.Pool(processes=processes) as p:
        list_df_brir = p.map(f, list_dfi_room)
    df_brir = pd.concat(list_df_brir).reset_index(drop=True).sort_index(axis=1)
    df_brir.to_pickle(fn_manifest_brir)
    print(f"Wrote `df_brir` ({len(df_brir)} BRIRs):\n{fn_manifest_brir}")


if __name__ == "__main__":
    main(
        fn_manifest_room="/om2/user/msaddler/spatial_audio_pipeline/assets/brir/v00/manifest_room.pdpkl", 
        fn_manifest_brir="/om2/user/msaddler/spatial_audio_pipeline/assets/brir/v00/manifest_brir.pdpkl",
        n_room=2000,
        list_src_dist=[1.4, None])
