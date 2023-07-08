function [alpha,freq] = acoeff(material,freq)
%
% ACOEFF frequency dependent material absorbtion coefficients
%
%        ACOEFF(MAT,FREQ) returns a vector of acoustic absorbtion 
%        coefficients equal in size to FREQ.  The coefficients are
%        taken from "Noise Control: handbook of principles and
%        practices," Lipscomb, D. M. and Taylor, A. C., Van Nostrand
%        Reinhold, 1978.  The data come from table 6-4, p. 143.
%
%        If no FREQ vector is given a default frequecy vector is used
%        and returned as a second output vector.
%
%        The MAT argument specifies the material type with:
%
% Walls
%   MAT =  1     Brick
%          2     Concrete, painted
%          3     Window Glass
%          4     Marble
%          5     Plaster on Concrete
%          6     Plywood
%          7     Concrete block, coarse
%          8     Heavyweight drapery
%          9     Fiberglass wall treatment, 1 in
%          10    Fiberglass wall treatment, 7 in
%          11    Wood panelling on glass fiber blanket

%
% Floors
%          12    Wood parquet on concrete
%          13    Linoleum
%          14    Carpet on concrete
%          15    Carpet on foam rubber padding
%
% Ceilings
%          16    Plaster, gypsum, or lime on lath
%          17    Acoustic tiles, 0.625", 16" below ceiling
%          18    Acoustic tiles, 0.5", 16" below ceiling
%          19    Acoustic tiles, 0.5" cemented to ceiling
%          20    Highly absorptive panels, 1", 16" below...
%
% Others
%          21    Upholstered seats
%          22    Audience in upholstered seats
%          23    Grass
%          24    Soil
%          25    Water surface
%          26    Anechoic
%          27    Uniform (0.6) absorbtion coefficient
%          28    Uniform (0.2) absorbtion coefficient
%          29    Uniform (0.8) absorbtion coefficient
%          30    Uniform (0.14) absorbtion coefficient
%          31    Artificial - absorbs more at high freqs
%          32    Artificial with absorption higher in middle ranges
%          33    Artificial  - absorbs more at low freqs
% Easy one: if 0 < material < 1, walls are set to uniform absorption
% with a coefficient value of 'material'.

freqtable = [ 125 250 500 1000 2000 4000];

if nargin < 2
    freq = freqtable;
end

if nargin < 1
    error('You must specify a material type');
end

%
% absorption coefficients
%

freqtable = [ 125 250 500 1000 2000 4000];

walls	= [	0.03	0.03	0.03	0.04	0.05	0.07;	...
			0.10	0.05	0.06	0.07	0.09	0.08;	...
			0.35	0.25	0.18	0.12	0.07	0.04;	...
			0.01	0.01	0.01	0.01	0.02	0.02;	...
			0.12	0.09	0.07	0.05	0.05	0.04;	...
			0.28	0.22	0.17	0.09	0.10	0.11;	...
			0.36	0.44	0.31	0.29	0.39	0.25;	...
			0.14	0.35	0.55	0.72	0.70	0.65;	...
			0.08	0.32	0.99	0.76	0.34	0.12;	...
			0.86	0.99	0.99	0.99	0.99	0.99;	...
			0.40	0.90	0.80	0.50	0.40	0.30];

floors	= [	0.04	0.04	0.07	0.06	0.06	0.07;	...
			0.02	0.03	0.03	0.03	0.03	0.02;   ...
			0.02	0.06	0.14	0.37	0.60	0.65;	...
			0.08	0.24	0.57	0.69	0.71	0.73];	

ceilings= [	0.14	0.10	0.06	0.05	0.04	0.03;	...
			0.25	0.28	0.46	0.71	0.86	0.93;	...
			0.52	0.37	0.50	0.69	0.79	0.78;	...
			0.10	0.22	0.61	0.66	0.74	0.72;	...
			0.58	0.88	0.75	0.99	1.00	0.96];

others	= [	0.19	0.37	0.56	0.67	0.61	0.59;	...
			0.39	0.57	0.80	0.94	0.92	0.87;	...
			0.11	0.26	0.60	0.69	0.92	0.99;	...
			0.15	0.25	0.40	0.55	0.60	0.60;	...
			0.01	0.01	0.01	0.02	0.02	0.03;	...
			1.00	1.00	1.00	1.00	1.00	1.00;	...
			0.60	0.60	0.60	0.60	0.60	0.60;	...
			0.20	0.20	0.20	0.20	0.20	0.20;   ...
			0.80	0.80	0.80	0.80	0.80	0.80;   ...
			0.14	0.14	0.14	0.14	0.14	0.14;   ...
            0.08	0.08	0.10	0.10	0.12    0.12;   ...
            0.05	0.05	0.20	0.20	0.10	0.10;
            0.12	0.12	0.10	0.10	0.08	0.08;];

atable = [ walls ; floors ; ceilings ; others ];

if (material>0)&(material<1),

    alpha = material*ones(size(freq));

else
    alpha = zeros(size(freq));

    for k = 1:length(freq)
        if freq(k) == 0 
            alpha(k) = 0;
        elseif freq(k) <= freqtable(1)
            alpha(k) = atable(material,1);
        elseif freq(k) >= freqtable(6);
            alpha(k) = atable(material,6);
        else
            alpha(k) = interp1(freqtable,atable(material,:),freq(k));
        end
    end
end

