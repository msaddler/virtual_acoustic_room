function [h_out, lead_zeros] = room_impulse_hrtf(src_loc, head_center, ...
        head_azimuth, meas_locs, meas_files, meas_delay, meas_sym, walls, ...
        wtypes, f_samp, c_snd, num_taps, log_dist, jitter, highpass, dsply);
    %
    % ROOM_IMPULSE_HRTF Impulse response to simulated head using real HRTF
    %	measurements.
    %
    %  	[H_OUT,LEAD_ZEROS]= ROOM_IMPULSE_HRTF(SRC_LOC,HEAD_CENTER, ...
    %			HEAD_AZIMUTH,MEAS_LOCS,MEAS_FILES,MEAS_DELAY,MEAS_SYM,WALLS, ...
    %			WTYPES,F_SAMP,C_SND,NUM_TAPS,LOG_DIST,JITTER,HIGHPASS,DSPLY)
    %
    %	  	returns a two-channel impulse response H_OUT of for source to ear
    %		behavior in a reverberent rectangular room.  These impulse responses
    %		are generated using a measured set of anechoic HRTF impulse responses
    %		in order to create realistic acoustic effects.  Since this simulation
    %		involves the use of measured values, the simulated reverberent
    %		impulse responses contain some error.  Simulation is performed in the
    %		following way:
    %
    %			1. The image locations of the reverberated source are generated
    %				together with the frequency-shaping associated with
    %				each source (due to the wall reflections).
    %
    %			2. For each source (true and image), the (x,y,z)
    %				coordinates are calculated relative to the head center and
    %				orientation.
    %
    %			3.	The location is determined of the measured HRTF impulse
    %				response that lies nearest to the propagation source location.
    %				The flag variable log_dist below allows for this determination
    %				to be made using log distances relative to the head.  This
    %				biases the source locations to be related to the measurement
    %				locations nearer to the head.
    %
    %			4. An impulse response that describes the additional travel time
    %				of the source relative to the measurement location is generated
    %				This impulse response also incorporates all wall reflection
    %				effects.  If the source is closer than the measurement location,
    %				a negative propagation delay is generated.
    %
    %			5. The impulse response from Step 4 is convolved with the
    %				measured impulse responses from the measurement location to form
    %				the source contribution to the final, reverberent impulse
    %				response.
    %
    %			6. All contributing impulse responses from the real and all image
    %				sources are combined into the final H_OUT.
    %
    %		Since direct-wave propagation delay is not always necessary, this
    %		simulation separates this delay from the final H_OUT.  This delay
    %		is contained in the output LEAD_ZEROS.  By appending a matrix of
    %		zeros(LEAD_ZEROS,2) to the beginning of H_OUT, a realistic impulse
    %		response that accounts for direct-wave propagation delay will be formed.
    %
    %		The function inputs are described as follows:
    %
    %		SRC_LOC is a 1 by 3 matrix of the source location in x-y-z space.
    %
    %		HEAD_CENTER is a 1 by 3 matrix describing the center of the head
    %		in x-y-z cartesian space.
    %
    %		HEAD_AZIMUTH is a scalar describing the azimuth (in degrees) of the head.
    %		this azimuth is measured relative to the +x-axis direction and is
    %		assumed to increase in the negative-y direction (as head rotates to the
    %		right, based on a right-hand coordingate system).
    %
    %		MEAS_LOCS is an L by 3 matrix containing the (r,az,elev) polar
    %		coordinates of the measured HRTF impulse responses relative to both
    %		head center location and the head zero-azimuth.  This means that
    %		(1,0,0) would be 1 distance unit in directly in front of the head,
    %		(1,0,90) would be one unit directly to the right, (1,90,0) would be one
    %		unit directly overhead and so forth.  There are L measurements
    %		in this set.
    %
    %		MEAS_FILES is an L-row string matrix where row l contains the
    %		file name of a 2-channel *.wav file containing the [left,right]
    %		measured HRTF impulse responses corresponding to the source
    %		location in row l of MEAS_LOCS.  NOTE: all impulse responses
    %		are assumed to be of an identical length.
    %
    %		MEAS_DELAY is an L by 1 vector that gives any additional delay to be
    %		incorporated into the measured impulse responses.  Specifically,
    %		this delay would be the travel time from source l to the receivers.
    %		In this way, the impulse response files may be smaller, because they
    %		do not contain unecessary delay samples.
    %
    %		MEAS_SYM is a scalar flag vector.  If equal to 1, then the impulse
    %		responses are assumed to be left right symmetric.  In this case,
    %		the simulation only uses impulse responses from 0 to 180 degrees
    %		azimuth.  The 180 to 360 degree values are obtained by transposing
    %		the left and right channels of the corresponding 0 to 180 degree
    %		measurment.  If MEAS_SYM is equal to 0, then no symmetry is assumed,
    %		and 360 degree measurments are required.
    %
    %		WALLS is a 1 by 3 matrix of the [X Y Z] room dimensions.  The
    %		filter gain is normalized for gain 1 distance unit from source.
    %
    %		WTYPES is a 1 by 6 matrix of surface material numbers.  See HELP
    %		ACOEFF for the material types (e.g. 26 = anec, 27 = uniform (0.6)
    %		absorp coeff, 28 = uniform (0.2) absorp coeff, and 29 = uniform
    %		(0.8) absorp coeff).  Specifically, WTYPES(1) is the reflection
    %		coefficient for the wall in the plane of X=0, WTYPE(2) for the
    %		wall at X=WALLS(1), WTYPES(3) for the wall at Y=0, WTYPES(4) for
    %		the wall at Y=WALLS(4), etc.
    %		NOTE: WTYPES can be a scalar, in which case all walls are
    %		assumed to be identical.
    %
    %		F_SAMP is the FIR filter sample rate (default = 10000 Hz).
    %
    %		C_SND is the propagation velocity (default = 344.5).
    %
    %		NUM_TAPS is the number of taps to put in the impulse response
    %		(excluding the leading zeros). (default = 2000).
    %
    %		LOG_DIST is a flag variable.  If equal to 1, then the 'nearest
    %		HRTF measurment location' determined in Step 3 above is formed
    %		using a log scale of distance.  In particular, this applies when
    %		a source location lies between HRTF measurements at two different
    %		distances from the center of the head (for which the head
    %		shadowing and delay may be significantly different).  The
    %		distance parameters of all three locations (source and two measurement
    %		locations) with respect to the head center are converted onto a
    %		log scale, and the source location is affiliated with the measurment
    %		location whose log distance is closest to the log distance of the
    %		source.  If LOG_FLAG is 0, then linear distance applies.  Once the
    %		source location is affiliated with a measurement location, the
    %		impulse response simulation continues with Step 4 above.
    %
    %		JITTER is a flag variable.  If equal to 1, all image sources that
    %		pass through 5 or more walls have a 1 percent standard-deviation
    %		jitter incorporated into the source distance from the microphone.
    %		Note that the jitter is constant for all microphones and is based
    %		upon an array center-of-gravity (i.e., average) microphone.
    %
    %		HIGHPASS is a flag that is 1 if the output impulse responses
    %		are to be highpass filtered to elimilate the DC component,
    %		otherwise no filtering is done.  Filter used is a two-zero,
    %		two-pole (z=1,1 and p=0.9889 +- j0.0110) butterworth.
    %		(default = 0, no highpass)
    %
    %		DSPLY is a toggle that controls the progress display for the
    %		simulation. 1 mens display is shown and 0 means no display.
    %		(default = 0, no display)
    %
    %		ROOM_IMPULSE uses the following MATLAB files:
    %
    %			IMPULSE_GENERATE_HRTF.M - Actually generates image source impulses
    %			                      and inserts into output impulse response.
    %			SHAPEDFILTER_HRTF.M - Used by IMPULSE_GENERATE_HRTF to create the
    %			                 		 non-integer impulse delay and to account
    %			                 		 for wall type.

    % Original by Mike O'Connell
    % modified 9/96 by Jay Desloge
    % Revised 2/2001 J. Desloge

    %
    % Define variables as global to be used by IMPULSE_GENERATE
    %
    clear global
    global h src h_cent h_az s_locations s_reflects
    global m_locs m_locs_xyz m_locs_xyz_logdist
    global m_files m_delay m_sym fs c taps
    global ctap ctap2 fgains nfreq lead_z l_dist jittr dsply L
    src = src_loc;
    h_cent = head_center;
    h_az = head_azimuth;
    m_locs = meas_locs;
    m_files = meas_files;
    m_delay = meas_delay;
    m_sym = meas_sym;
    fs = f_samp;
    c = c_snd;
    taps = num_taps;
    dspy = dsply;
    jittr = jitter;
    l_dist = log_dist;
    L = length(meas_locs(:, 1));

    %
    % Check inputs and assign default values
    %
    if nargin < 9
        error('Insufficient inputs specified');
    end

    if nargin < 15,
        dspy = 0,
    end

    if nargin < 14,
        highpass = 0,
    end

    if nargin < 13,
        jitter = 0,
    end

    if nargin < 12
        taps = 2000; % Default output impulse response length
    end

    if nargin < 11
        c = 345; % Default sound propagation velocity
    end

    if nargin < 10
        fs = 10000; % Default sampling rate
        disp('room2src: assuming a 10000 Hz sampling rate.');
    end

    if length(wtypes) == 1
        wtypes = wtypes * ones(1, 6);
    end

    %
    % find frequency dependent reflection coefficients for each wall
    %
    uniform_walls = 1;

    for k = 1:6
        [alpha, freq] = acoeff_hrtf(wtypes(k)); % alpha = wall power absorp
        % freq = frequencies
        fgains(k, :) = sqrt(1 - alpha); % fgains = wall reflection

        if sum(abs(fgains(k, :) - fgains(k, 1))) > 0, % If fgains depends on freq,
            uniform_walls = 0; % set uniform_walls flag to 0.
        end

    end

    nfreq = freq / fs; % freq as fraction of fs

    %
    % BEGIN CALCULATIONS
    %

    %
    % Part I: Initialization
    %

    % Set some useful values
    ctap = 11; % Center tap of lowpass to create
    % non-integer delay impulse
    % (as in Peterson)

    if (uniform_walls == 1), % Center tap of filter to account
        ctap2 = 1; % for freq dep wall reflects
    else % If walls are uniform, use
        ctap2 = 33; % single tap filter (gain)
    end

    num_rec = 2; % number ears

    % Convert measured HRTF locations into room (xyz) coordinates
    %	Also form log distance locations
    m_locs_xyz = (m_locs(:, 1) * ones(1, 3)) .* ...
        [cos((m_locs(:, 2) + h_az) * pi / 180) .* cos(m_locs(:, 3) * pi / 180), ...
         -sin((m_locs(:, 2) + h_az) * pi / 180) .* cos(m_locs(:, 3) * pi / 180), ...
         sin(m_locs(:, 3) * pi / 180)];
    m_locs_xyz = m_locs_xyz + ones(L, 1) * h_cent;
    m_locs_xyz_logdist = ((log(m_locs(:, 1)) - log(0.05)) * ones(1, 3)) .* ...
        [cos((m_locs(:, 2) + h_az) * pi / 180) .* cos(m_locs(:, 3) * pi / 180), ...
         -sin((m_locs(:, 2) + h_az) * pi / 180) .* cos(m_locs(:, 3) * pi / 180), ...
         sin(m_locs(:, 3) * pi / 180)];
    m_locs_xyz_logdist = m_locs_xyz_logdist + ones(L, 1) * h_cent;

    % Calculate the number of lead zeros to strip.
    [m, i] = min(sqrt(((ones(L, 1) * src - m_locs_xyz) .^ 2) * ones(3, 1)));
    src_mloc = m_locs_xyz(i, :); %Nearest measured loc or direct path
    rel_dist = norm(src - h_cent, 2) - norm(src_mloc - h_cent, 2);
    lead_zeros = meas_delay(i) + floor(fs * rel_dist / c);
    lead_z = lead_zeros;

    % Initialize output matrix (will later truncate to exactly taps length).
    %Old code read: ht = wavread(deblank(meas_files(1,:)));
    %wavread depreciated --> audioread, also changed to deal with path
    %backslash for unix(AFF 07/2017)
    ht = audioread(strrep(deblank(meas_files(1, :)), '\', filesep));
    %  ht = wavread(deblank(eval(meas_files(1,:))));
    h = zeros(taps + ctap + ctap2 + length(ht(:, 1)), num_rec);

    %
    % Part II: determine source image locations and corresponding impulse
    % response contribution from each source.  To speed up process yet ease
    % the computational burden, for every 10000 source images, break off and
    % determine impulse response.
    %

    % The for determining source images is as follows.
    %
    %	1. Calculate maximum distance which provides relevant sources
    %		(i.e., those that arrive within the imp_resp duration)
    % 	2. By looping through the X dimension, generate images of
    %		the (0,0,0) corner of the room, restricting the
    %		distance below the presecribed level.
    %	3. Use the coordinates of each (0,0,0) image to generate 8
    %		source images
    %	4. Generate corresponding number of reflections from each wall
    %	 	for each source image.

    dmax = ceil((taps + lead_zeros) * c / fs + max(walls));
    % maximum source distance to be in
    % impulse response

    s_locations = ones(20000, 3); % Initialize locations and
    s_reflects = ones(20000, 6); % reflections matrices

    src_pts = [1 1 1; 1 1 -1; 1 -1 1; 1 -1 -1; -1 1 1; -1 1 -1; -1 -1 1; -1 -1 -1] .* ...
        (ones(8, 1) * src); % vector to get locations from
    % the (0,0,0) corner images

    Nx = ceil(dmax / (2 * walls(1))); % Appropriate number of (0,0,0)
    % images in either the +x of -x
    % directions to generate images
    % within dmax.

    loc_num = 0; % initialize location number index

    for nx = Nx:-1:0, % loop through the images of (0,0,0)

        if dspy,
            disp(sprintf('Stage %i', nx));
        end

        if nx < Nx,
            ny = ceil(sqrt(dmax * dmax - (nx * 2 * walls(1)) ^ 2) / (2 * walls(2)));
            nz = ceil(sqrt(dmax * dmax - (nx * 2 * walls(1)) ^ 2) / (2 * walls(3)));
        else % Determine the number of y and z
            ny = 0; nz = 0; % so that we contain dmax
        end

        X = nx * ones((2 * ny + 1) * (2 * nz + 1), 1); % Form images of (0,0,0)
        Y = [-ny:ny]' * ones(1, 2 * nz + 1); Y = Y(:);
        Z = ones(2 * ny + 1, 1) * [-nz:nz]; Z = Z(:);

        if nx ~= 0, % if nx~=0, do both +nx and -nx
            X = [-X; X]; Y = [Y; Y]; Z = [Z; Z]; % images of (0,0,0).
        end

        Xw = 2 * walls(1) * X;
        Yw = 2 * walls(2) * Y;
        Zw = 2 * walls(3) * Z;

        for k = 1:8, % for each image of (0,0,0), get
            % the eight source images and
            % number of relfects at each wall
            s_locs = zeros(length(X), 3);
            s_refls = zeros(length(X), 6);
            s_locs = [Xw, Yw, Zw] + ones(size(Xw)) * src_pts(k, :);
            s_refls(:, 1) = (src_pts(k, 1) > 0) * abs(X) + (src_pts(k, 1) < 0) * abs(X - 1);
            s_refls(:, 2) = abs(X);
            s_refls(:, 3) = (src_pts(k, 2) > 0) * abs(Y) + (src_pts(k, 2) < 0) * abs(Y - 1);
            s_refls(:, 4) = abs(Y);
            s_refls(:, 5) = (src_pts(k, 3) > 0) * abs(Z) + (src_pts(k, 3) < 0) * abs(Z - 1);
            s_refls(:, 6) = abs(Z);

            while (loc_num + length(s_locs(:, 1))) > 20000, % If the current source
                m = 20000 - loc_num; % image matrix has more than
                s_locations((loc_num + [1:m]), :) = s_locs(1:m, :); % 20000 images, process to
                s_reflects((loc_num + [1:m]), :) = s_refls(1:m, :);
                % get impulse response
                done = impulse_generate_hrtf;
                % contributions.

                loc_num = 0; % Reset loc_num counter
                s_locs = s_locs(m + 1:length(s_locs(:, 1)), :); % and continue
                s_refls = s_refls(m + 1:length(s_refls(:, 1)), :);
            end

            s_locations((loc_num + [1:length(s_locs(:, 1))]), :) = s_locs;
            s_reflects((loc_num + [1:length(s_locs(:, 1))]), :) = s_refls;
            loc_num = loc_num + length(s_locs(:, 1)); % If current locations matrix
        end % has < 20000 images, keep

    end % building

    s_locations = s_locations(1:loc_num, :); % When all locations have
    s_reflects = s_reflects(1:loc_num, :); % been generated, process
    done = impulse_generate_hrtf; % the final ones.

    %
    % Part III: Finalize output
    %
    if highpass,
        [bhp, ahp] = butter(2, 0.005, 'high'); % Highpass filter, if desired

        for k = 1:length(h(1, :)),
            h(:, k) = filter(bhp, ahp, h(:, k));
        end

    end

    h_out = h(1:taps, :); % Restrict h to 'taps' length.
