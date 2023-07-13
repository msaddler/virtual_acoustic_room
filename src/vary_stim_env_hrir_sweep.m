function [] = vary_stim_hrir_sweep(wall_xdim_str, wall_ydim_str, wall_zdim_str, wall_material_str, floor_material_str, cieling_material_str, head_x_str, head_y_str, head_z_str, speaker_dist_str)
    load data_locs
    load file_names
    wall_xdim = str2num(wall_xdim_str);
    wall_ydim = str2num(wall_ydim_str);
    wall_zdim = str2num(wall_zdim_str);
    walls = [wall_xdim, wall_ydim, wall_zdim];
    head_azimuth = 0;
    %set postion so that speak never touches room wall given the speaker is
    %"speaker_dist"  meters from the listener
    head_x = str2num(head_x_str);
    head_y = str2num(head_y_str);
    head_z = str2num(head_z_str);
    speaker_dist = str2num(speaker_dist_str);
    head_center = [head_x, head_y, head_z];
    meas_locs = locs_gardnermartin;
    meas_files = gardnermartin_file;

    c_snd = 344.5;
    meas_sym = 1;
    wall_material = str2num(wall_material_str);
    floor_material = str2num(floor_material_str);
    cieling_material = str2num(cieling_material_str);
    wtypes = [wall_material, wall_material, wall_material, wall_material, floor_material, cieling_material]; % [x=0 x=L y=0 y=W z=0 z=H]

    f_samp_Hz = 44100;
    num_taps = 22050;
    dsply = 0;
    jitter = 1;
    log_dist = 0;
    highpass = 1;

    new_dir = sprintf('./Expanded_HRIRdist%d-5deg_elev_az_room%dx%dy%dz_materials%dwall%dfloor%dciel', round(speaker_dist * 100), wall_xdim, wall_ydim, wall_zdim, wall_material, floor_material, cieling_material);

    if ~exist(new_dir, 'dir')
        mkdir(new_dir);
    end

    for elev = -20:10:60

        for az = 0:5:355
            x = speaker_dist * cosd(elev) .* cosd(az) + head_x;
            y = speaker_dist * cosd(elev) .* sind(az) + head_y;
            z = speaker_dist * sind(elev) +head_z;
            src_loc = [x, y, z]
            %moved from parameter space for varying head/source locations
            d = sqrt(sum((src_loc - head_center) .^ 2));
            meas_delay = (d / c_snd) * ones(size(gardnermartin_file, 1), 1);

            out_l = sprintf('./Expanded_HRIRdist%d-5deg_elev_az_room%dx%dy%dz_materials%dwall%dfloor%dciel/%delev_%daz_%.2fx%.2fy%.2fz_l.wav', round(100 * speaker_dist), wall_xdim, wall_ydim, wall_zdim, wall_material, floor_material, cieling_material, elev, az, head_x, head_y, head_z);
            out_r = sprintf('./Expanded_HRIRdist%d-5deg_elev_az_room%dx%dy%dz_materials%dwall%dfloor%dciel/%delev_%daz_%.2fx%.2fy%.2fz_r.wav', round(100 * speaker_dist), wall_xdim, wall_ydim, wall_zdim, wall_material, floor_material, cieling_material, elev, az, head_x, head_y, head_z);

            if exist(out_l, 'file')
                disp('File exists!')
                disp(wtypes)
                continue
            end

            disp(wtypes)
            tic
            [h_out, lead_zeros] = room_impulse_hrtf(src_loc, head_center, ...
                head_azimuth, meas_locs, meas_files, meas_delay, meas_sym, walls, ...
                wtypes, f_samp_Hz, c_snd, num_taps, log_dist, jitter, highpass, dsply);
            toc

            audiowrite(out_r, h_out(:, 2), 44100, 'BitsPerSample', 32);
            audiowrite(out_l, h_out(:, 1), 44100, 'BitsPerSample', 32);
        end
    end
end
