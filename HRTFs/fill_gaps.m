% Generate gap locations
elev_angles = [-90:10:-50]';
elev_angles_meas = [-40:10:90]';
az_divs = [1;12;24;36;45];
az_divs_meas = [56;60;72;72;72;72;72;60;56;45;36;24;12;1];
half_az_divs = floor(az_divs/2 + 1);
elev = [];
azimuth = [];
for k=1:length(half_az_divs);
   elev = [elev;elev_angles(k)*ones(half_az_divs(k),1)];
   azimuth = [azimuth;[0:360/az_divs(k):180]'];
end   

% Generate file extension names
file_ext = [];
elev_dir = [];
for k=1:length(elev),
   if azimuth(k)<10, z = '00';
   elseif azimuth(k)<100, z = '0';
   else z = '';
   end
   file_ext = strvcat(file_ext,[int2str(elev(k)),'e',z,int2str(azimuth(k)),'a.wav']);
	elev_dir = strvcat(elev_dir,['HRTFs\elev',int2str(elev(k)),'\']);   
end


walls = [3 3 3];
sp_cent = walls/2;
sp_rad = 0.08;
recs = [1;1]*sp_cent+[0 -1.0001*sp_rad 0;0 1.0001*sp_rad 0];
src_dist = 1.4;
src_locs = ones(size(elev))*sp_cent+src_dist*[cos(azimuth*pi/180).*cos(elev*pi/180), ...
      					sin(azimuth*pi/180).*cos(elev*pi/180), sin(elev*pi/180)];


for k = 1:length(elev),
   k,
	[h_out,lead_zeros]= room_impulse(src_locs(k,:),recs,walls,26,sp_cent,sp_rad,44100,345,128,0,1,1);
   
   file = [elev_dir(k,:),'SP',file_ext(k,:)];
   wavwrite(h_out,file);
   
   file = [elev_dir(k,:),'SP-3dB',file_ext(k,:)];
   wavwrite(sqrt(0.5)*h_out,file);
   
	[h_out,lead_zeros]= room_impulse(src_locs(k,:),recs,walls,26,sp_cent,0,44100,345,128,0,1,1);
   
   file = [elev_dir(k,:),'FF',file_ext(k,:)];
   wavwrite(h_out,file);
   
   file = [elev_dir(k,:),'FF-3dB',file_ext(k,:)];
   wavwrite(sqrt(0.5)*h_out,file);
   
	h_out = zeros(128,2);   
   
   file = [elev_dir(k,:),'ZE',file_ext(k,:)];
   wavwrite(h_out,file);
end

   
   