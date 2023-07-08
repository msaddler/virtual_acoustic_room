% Generate gap locations
elev_angles = [-90:10:-50]';
elev_angles_meas = [-40:10:90]';
az_divs = [1;12;24;36;45];
az_divs_meas = [56;60;72;72;72;72;72;60;56;45;36;24;12;1];
half_az_divs = floor(az_divs/2 + 1);
half_az_divs_meas = floor(az_divs_meas/2 + 1);
elev = [];
azimuth = [];
elev_meas = [];
azimuth_meas = [];
for k=1:length(half_az_divs);
   elev = [elev;elev_angles(k)*ones(half_az_divs(k),1)];
   azimuth = [azimuth;[0:360/az_divs(k):180]'];
end   
for k=1:length(half_az_divs_meas);
   elev_meas = [elev_meas;elev_angles_meas(k)*ones(half_az_divs_meas(k),1)];
   azimuth_meas = [azimuth_meas;[0:360/az_divs_meas(k):180]'];
end   

% Generate file extension names
sp_file = [];
sp_h_file = [];
ff_file = [];
ff_h_file = [];
ze_file = [];
for k=1:length(elev),
   if azimuth(k)<10, z = '00';
   elseif azimuth(k)<100, z = '0';
   else z = '';
   end
   sp_file = strvcat(sp_file,['HRTFs\elev',int2str(elev(k)),'\SP', ...
         int2str(elev(k)),'e',z,int2str(azimuth(k)),'a.wav']);
   sp_h_file = strvcat(sp_h_file,['HRTFs\elev',int2str(elev(k)),'\SP-3dB', ...
         int2str(elev(k)),'e',z,int2str(azimuth(k)),'a.wav']);
   ff_file = strvcat(ff_file,['HRTFs\elev',int2str(elev(k)),'\FF', ...
         int2str(elev(k)),'e',z,int2str(azimuth(k)),'a.wav']);
   ff_h_file = strvcat(ff_h_file,['HRTFs\elev',int2str(elev(k)),'\FF-3dB', ...
         int2str(elev(k)),'e',z,int2str(azimuth(k)),'a.wav']);
   ze_file = strvcat(ze_file,['HRTFs\elev',int2str(elev(k)),'\ZE', ...
         int2str(elev(k)),'e',z,int2str(azimuth(k)),'a.wav']);   
end
meas_file = [];
for k=1:length(elev_meas),
   if azimuth_meas(k)<10, z = '00';
   elseif azimuth_meas(k)<100, z = '0';
   else z = '';
   end
   meas_file = strvcat(meas_file,['HRTFs\elev',int2str(elev_meas(k)),'\H', ...
         int2str(elev_meas(k)),'e',z,int2str(azimuth_meas(k)),'a.wav']);
end

save file_names sp_file sp_h_file ff_file ff_h_file ze_file meas_file

locs = [1.4*ones(size(elev)), azimuth, elev];
locs_meas = [1.4*ones(size(elev_meas)), azimuth_meas, elev_meas];
save data_locs locs locs_meas