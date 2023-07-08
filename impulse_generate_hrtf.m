function done=impulse_generate_hrtf
%
% IMPULSE_GENERATE_HRTF
%
%	IMPULSE_GENERATE_HRTF is called by the function ROOM_IMPULSE_HRTF.
%	It take a set of source images (S_LOCATIONS) and generates
%	the corresponding impulse response component between each image
%	and a simulated head and inserts this into the overall
%	impulse response for each ear (H).
%
%	All variables are passed as global variables from ROOM_IMPULSE.
%
%	H = simulated impulse responses that are being built.
%
%	H_CENT = XYZ coordinate location for the center of the simulated 
%	head
%
%	H_AZ = the azimuth orientation of the simulated head.
%
%	S_LOCATIONS = (XYZ) coordinates of sources (direct and image) within 
%	the space
%
%	S_REFLECTS = the number of reflections at each wall experienced by
%	each source image.
%
%	L = number of measured HRTF impulse responses in use.
%
% 	M_LOCS = (r,elev,azimuth) locations of measured HRTF impulse responses.
%	Coordinates relative to head and its affiliated azimuth.
%
%	M_LOCS_XYZ = measurement locations in the XYZ coordinate system of 
% 	the room
%
%	M_DELAY = propagation delay associated with, but not included in, 
%	the measured HRTF impulse response data.
%
%	M_SYM = symmetry flag variable indicating whether lft/right symmetry
%	is assumed.  If equal to 1, symmetry is assumed, and simulation uses
%	only right-side measured data.  If equal to 0, no symmetry is assumed
%	and full space data is required.
%
%	FS = sampling frequency.
%
%	C = sound propoagation velocity.
%
%	TAPS = desired length of output impulse response.
%
%	CTAP = center tap of filter to approximate the non-integer sample
%	       delay when travelling from source to receiver.
%
%	CTAP2 = center tap of filter to approximate the combined effects of
%		freq-dependent wall reflections and diffraction about sphere.
%
%	LEAD_Z = the number of leading zeros to strip from the impulse 
%		response.
%
%	FGAINS, NFREQ = the wall reflection freq-dependent gains and
%	corresponding frequencies.
%
% 	JITTR is a flag.  If equal to 1, jitter is incorporated into
%	further reflections to elimiate funny structural artifacts
%	that enter the simulation.  If equal to 0, no jitter is used.
%
%	DSPY if a flag that controls the display of the simulation process.

% J. Desloge 9/19/96
% Revised 2/2001 J. Desloge

% 
% Global variables from ROOM_IMPULSE
%
 global h src h_cent h_az s_locations s_reflects 
 global L m_locs m_locs_xyz m_locs_xyz_logdist 
 global m_files m_delay m_sym fs c taps l_dist
 global ctap ctap2 fgains nfreq lead_z jittr dspy
 jitter_reflects = 5;

%
% Part I: Form variables to be used in impulse response generation
%

% Determine the overall source gains (based on number of reflections
% through each wall) for each source location.
 gains = ones(length(s_locations(:,1)),length(nfreq));		
 for wall = 1:6
  gains = gains .* ...
          ((ones(size(s_locations(:,1)))*fgains(wall,:)).^ ...
             (s_reflects(:,wall)*ones(size(nfreq)))); 
 end	
 
% If m_sym is active, convert 180-360 sources to 0-180 sources
 s_locations_relh = s_locations - ones(size(s_locations(:,1)))*h_cent ;
 s_locations_pol = zeros(size(s_locations));
 s_locations_pol(:,1) = sqrt((s_locations_relh.^2)*ones(3,1));
 s_locations_pol(:,2) = (180/pi)*angle(s_locations_relh(:,1)-j*s_locations_relh(:,2)) ...
    - h_az;
 s_locations_pol(:,3) = (180/pi)*asin(s_locations_relh(:,3)./s_locations_pol(:,1));
 flip = zeros(size(s_locations(:,1)));
 if m_sym,
 	 flip = (s_locations_pol(:,2)<0);
    s_locations_pol(:,2) = abs(s_locations_pol(:,2));
    s_locations = (s_locations_pol(:,1)*ones(1,3)).* ...
    	[cos((s_locations_pol(:,2)+h_az)*pi/180).*cos(s_locations_pol(:,3)*pi/180), ...
       -sin((s_locations_pol(:,2)+h_az)*pi/180).*cos(s_locations_pol(:,3)*pi/180), ...
       sin(s_locations_pol(:,3)*pi/180)];
 	 s_locations = s_locations + ones(size(s_locations(:,1)))*h_cent;
 else
    flip = zeros(size(s_locations(:,1)));
 end
 
 % If log_dist is active, form s_locations_logdist
 if l_dist,
    s_locations_logdist = ((log(s_locations_pol(:,1))-log(0.05))*ones(1,3)).* ...
    	[cos((s_locations_pol(:,2)+h_az)*pi/180).*cos(s_locations_pol(:,3)*pi/180), ...
       -sin((s_locations_pol(:,2)+h_az)*pi/180).*cos(s_locations_pol(:,3)*pi/180), ...
       sin(s_locations_pol(:,3)*pi/180)];
 	 s_locations_logdist = s_locations_logdist + ones(size(s_locations_logdist(:,1)))*h_cent;
 end
 
% For each source, determine the closest measurement spot
 if ~l_dist,
   Dx = ones(length(m_locs_xyz(:,1)),1)*s_locations(:,1).' - ...
   	m_locs_xyz(:,1)*ones(1,length(s_locations(:,1)));
   Dy = ones(length(m_locs(:,1)),1)*s_locations(:,2).' - ...
   	m_locs_xyz(:,2)*ones(1,length(s_locations(:,1)));
 	Dz = ones(length(m_locs(:,1)),1)*s_locations(:,3).' - ...
   	m_locs_xyz(:,3)*ones(1,length(s_locations(:,1)));
 else
   Dx = ones(length(m_locs_xyz_logdist(:,1)),1)*s_locations_logdist(:,1).' - ...
   	m_locs_xyz_logdist(:,1)*ones(1,length(s_locations_logdist(:,1)));
   Dy = ones(length(m_locs_xyz_logdist(:,1)),1)*s_locations_logdist(:,2).' - ...
   	m_locs_xyz_logdist(:,2)*ones(1,length(s_locations_logdist(:,1)));
 	Dz = ones(length(m_locs_xyz_logdist(:,1)),1)*s_locations_logdist(:,3).' - ...
   	m_locs_xyz_logdist(:,3)*ones(1,length(s_locations_logdist(:,1)));   
 end  
 [m,near_m_loc] = min(sqrt(Dx.^2+Dy.^2+Dz.^2)); 
 clear Dx Dy Dz
 
%
% Part II: Based on the center of the head, introduce a
% 1 percent jitter to add into all source-to-mic
% distances that are reflected by more than 5 walls.
% Add jitter only if jitter flag == 1.
%
 no_jitt = find((s_reflects*ones(6,1))<jitter_reflects);
 jitt = ((0.01*s_locations_pol(:,1))).*randn(size(s_locations_pol(:,1)));
 jitt(no_jitt) = 0;
 if jittr,
    s_locations_pol(:,1) = s_locations_pol(:,1) + jitt;
 end
 
 
% Calculate the relative additional distance between each
% (jittered) source and the corresponding measurement location.
 rel_dist = s_locations_pol(:,1) - m_locs(near_m_loc(:),1);
 
%
% Part III: For each measurement location, generate impulse
% response from corresponding sources to meas loc.  Then
% incorporate HRTFs.  Treat flips and no flips accordingly.
%

[h1, h2, h3 ] =size(h);
hrtf_temp = zeros([L h1 h2]);
numColsInB = size(hrtf_temp,2);
 parfor l = 1:L,
    % Determine sources in this cell for no flip and flip.
    cell_l = find(near_m_loc==l);
    if length(cell_l)>0,
	 	ind_noflip = cell_l(flip(cell_l)==0);
  	 	ind_flip = cell_l(flip(cell_l)==1);
      
		% Initialize
  	 	h_flip = zeros(size(h));
      h_noflip = zeros(size(h));
      
      % Treat non-flipped sources
      if length(ind_noflip)>0,
          
       	% Get sample delays to the measured location
       	thit = ctap + ctap2 - lead_z + m_delay(l) + ...
				rel_dist(ind_noflip)*fs/c;% number of samples delay to travel to rec
       	ihit = floor(thit);				% round down	
         fhit = thit - ihit;				% non-integer extra samples
         gains_noflip = gains(ind_noflip,:);
         
		 	% Get scale factors to account for distance traveled       
       	m_sc = 1./m_locs(near_m_loc(ind_noflip),1);
       	s_sc = 1./s_locations_pol(ind_noflip,1);
       	rel_sc = s_sc./m_sc;
       
		 	% eliminate locations that are too far away to enter into impulse response
   	 	v=find(ihit<=taps+ctap+ctap2);
          
         if length(v)>0,
             
       		% Initialize temporary impulse response vector
	    		ht = zeros(length(h(:,1))+ctap+1+ctap2+1,1);
   
   	 		ht_ind = (ihit(v)*ones(1,2*ctap-1+2*ctap2-1-1))+ ...
					(ones(size(ihit(v)))*[-ctap-ctap2+1+1:ctap+ctap2-1-1]);
					% Indices into ht.  Each row corresonds to 
					% one source image location, with the center
					% determined by ihit. Within a row, there are
					% (2*ctap-1)+(2*ctap2-1)-1 values
					% that account for non-integer delay, fhit and
					% for the freq-dep wall reflections/sphere
					% diffraction.

        		% For each source location, determine the impulse response.
       		h_temp = (rel_sc(v)*ones(1,2*ctap-1+2*ctap2-1-1)).* ...
	 				(shapedfilter_hrtf(fhit(v),nfreq,gains_noflip(v,:),fs,ctap,ctap2));
					% form filter to incorporate frequency gains,
					% non-integer delay and scattering off of 
					% rigid sphere.

	  		 	% Add the impules response segments into the overall impulse response.
   		 	for k=1:length(v),
      	    	ht(ht_ind(k,:),1)=ht(ht_ind(k,:),1)+h_temp(k,:)';
	    		end

	       	% Incorporate HRTF impulse response and
   	    	% add into overall impulse response matrix
      	 	%hrtf = wavread(deblank(m_files(l,:)));
            %wavread depreciated--> audioread
            hrtf = audioread(strrep(deblank(m_files(l,:)),'\',filesep));
   	 		%h(:,1)=h(:,1)+fftfilt(hrtf(:,1),ht(1:length(h(:,1)))); 
   	 		%h(:,2)=h(:,2)+fftfilt(hrtf(:,2),ht(1:length(h(:,1))));
            %following code added to enable parallelization
            newVals = [fftfilt(hrtf(:,1),ht(1:length(h(:,1)))), fftfilt(hrtf(:,2),ht(1:length(h(:,1))))];
            hrtf_temp(l,:,:) =  hrtf_temp(l,:,:) + reshape(newVals, 1, numColsInB, 2);
 	 		end	   
   	end  
    	if length(ind_flip)>0,
       	% Get sample delays to the measured location
       	thit = ctap + ctap2 - lead_z + m_delay(l) + ...
				rel_dist(ind_flip)/c*fs;% number of samples delay to travel to rec
       	ihit = floor(thit);				% round down	
       	fhit = thit - ihit;				% non-integer extra samples
         gains_flip = gains(ind_flip,:);
       
		 	% Get scale factors to account for distance traveled       
       	m_sc = 1./m_locs(near_m_loc(ind_flip),1);
       	s_sc = 1./s_locations_pol(ind_flip,1);
       	rel_sc = s_sc./m_sc;
       
		 	% eliminate locations that are too far away to enter into impulse response
   	 	v=find(ihit<=taps+ctap+ctap2);
          
         if length(v)>0, 
            
            % Initialize temporary impulse response vector
		    	ht = zeros(length(h(:,1))+ctap+1+ctap2+1,1);
   
   		 	ht_ind = (ihit(v)*ones(1,2*ctap-1+2*ctap2-1-1))+ ...
					(ones(size(ihit(v)))*[-ctap-ctap2+1+1:ctap+ctap2-1-1]);
				% Indices into ht.  Each row corresonds to 
				% one source image location, with the center
				% determined by ihit. Within a row, there are
				% (2*ctap-1)+(2*ctap2-1)-1 values
				% that account for non-integer delay, fhit and
				% for the freq-dep wall reflections/sphere
				% diffraction.
            
	         % For each source location, determine the impulse response.
   	    	h_temp = (rel_sc(v)*ones(1,2*ctap-1+2*ctap2-1-1)).* ...
	 				(shapedfilter_hrtf(fhit(v),nfreq,gains_flip(v,:),fs,ctap,ctap2));
				% form filter to incorporate frequency gains,
				% non-integer delay and scattering off of 
				% rigid sphere.

	  		 	% Add the impules response segments into the overall impulse response.   
            for k=1:length(v),
         		 ht(ht_ind(k,:),1)=ht(ht_ind(k,:),1)+h_temp(k,:)';
	    		end
       
       		% Incorporate HRTF impulse response and
       		% add into overall impulse response matrix
            %hrtf = wavread(deblank(m_files(l,:)));
            %wavread depreciated --> audioread
            hrtf = audioread(strrep(deblank(m_files(l,:)),'\',filesep));
            % hrtf = wavread(deblank(eval(m_files(l,:))));
   	 		%h(:,1)=h(:,1)+fftfilt(hrtf(:,2),ht(1:length(h(:,1)))); 
   	 		%h(:,2)=h(:,2)+fftfilt(hrtf(:,1),ht(1:length(h(:,1))));
            %following code added to enable parallelization
            newVals = [fftfilt(hrtf(:,2),ht(1:length(h(:,1)))), fftfilt(hrtf(:,1),ht(1:length(h(:,1))))];
            hrtf_temp(l,:,:) =  hrtf_temp(l,:,:) + reshape(newVals, 1, numColsInB, 2);
             
          end
       end   
    end
 end
 h = h + squeeze(sum(hrtf_temp,1));
done = 1;
 	  
 
