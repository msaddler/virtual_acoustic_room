function hout = shapedfilter_hrtf(sdelay, freq, gain, fs, ctap, ctap2);
    %
    % SHAPEDFILTER	shaped filter design for non-integer delay
    %
    %               SHAPEDFILTER(SDELAY,FREQ,GAIN,CTAP,CTAP2) is to
    %		be used by IMPULSE_GENERATE to generate the
    %		(2*CTAP-1)+(2*CTAP2-1)-1 long impulse responses (in each
    %		row) that produce the non-integer delay (SDELAY) and
    %		freq-dependent wall-reflection effects for each source
    %		image location.
    %
    %		The inputs are:
    %
    %		SDELAY a vector of non-integer delays for each source image.
    %
    %		FREQ, GAIN are vectors specifying the frequency dependent
    %		wall reflection effects for each source image.
    %
    %		FS is the sampling frequency.
    %		CTAP is the center tap of the filter for non-integer delay
    %		CTAP2 is the center tap of filter to account for the
    %		      frequency-dependent wall reflections and
    %		      sphere diffraction.

    % Originally by Mike O'Connell
    % Revised by Jay Desloge 9/19/96
    % Revised J. Desloge 2/2001

    %
    % Parse inputs
    %
    if nargin < 6
        error('Insufficient input arguments supplied!');
    end

    if sdelay < 0
        error('The sample delay must be positive');
    end

    %
    % design the non-integer delay filter
    %
    ntaps = 2 * ctap - 1;
    N = ctap - 1;
    fc = 0.9;
    h = 0.5 * fc * (1 + ...
        cos(pi * (ones(size(sdelay)) * [-N:N] - sdelay * [0, ones(1, ntaps - 1)]) / N)) .* ...
        sinc(fc * (ones(size(sdelay)) * [-N:N] - sdelay * [0, ones(1, ntaps - 1)]));

    %
    % make sure that freq are is vectors
    %
    freq = freq(:).';

    %
    % Design and incorporate the wall filter if it is needed (i.e., is ctap2>1).
    % Otherwise, just scale impulse resp. appropriately.
    %

    if ctap2 > 1,
        df = [0:ctap2 - 1] * (pi / (ctap2 - 1)); % Determine FFT points

        freq = [-eps 2 * pi * freq pi]; % frequencies at which gain is defined
        gain = [gain(:, 1) gain gain(:, length(gain(1, :)))];
        G = interp1(freq.', gain.', df(:)).';
        % Interpolate reflection frequency-dependence
        % to get gains at FFT points.

        %
        % Combine the non-integer delay filter and the wall/sphere filter
        %
        G(:, ctap2) = real(G(:, ctap2));
        G = [G, fliplr(conj(G(:, 2:ctap2 - 1)))]; % Transform into appropriate
        % wall into transfer function.
        gt = real(ifft(G.')); % IFFT to get imp-resp

        g = [0.5 * gt(ctap2, :); gt(ctap2 + 1:2 * ctap2 - 2, :); ...
               gt(1:ctap2 - 1, :); 0.5 * gt(ctap2, :); ...
               zeros(2 * ctap - 2, length(gt(1, :)))];
        G = fft(g); % Zero-pad and FFT

        H = fft([h.'; zeros(2 * ctap2 - 2, length(h(:, 1)))]);
        % Zero-pad and FFT delay filter

        HOUT = H .* G; % Convolve filters
        hout = real(ifft(HOUT)).'; % Obtain total impulse response

    else
        hout = h .* (gain(:, 1) * ones(size(h(1, :))));
        % Scale impulse response only if
        % Wall reflects are freq-indep and
        % if sphere not present.
    end
