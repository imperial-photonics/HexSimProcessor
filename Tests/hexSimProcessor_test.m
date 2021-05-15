close all
h = hexSimProcessor;
h.N = 512;          % Points to use in FFT
h.pixelsize = 5.85;    % Camera pixel size
h.magnification = 40; % Objective magnification
h.NA = 0.75;         % Numerical aperture at sample
n = 1;         % Refractive index at sample
h.lambda = 0.6;   % Wavelength in um
h.alpha = 0.1;      % Zero order attenuation width
h.beta = 0.99;      % Zero order attenuation
        % eta is the factor by which the illumination grid frequency exceeds the
        % incoherent cutoff, eta=1 for normal SIM, eta=sqrt(3)/2 to maximise
        % resolution without zeros in TF
        % carrier is 2*kmax*eta
h.eta = 0.7;
h.w = 0.3;         % Wiener parameter
h.cleanup = true;
h.debug = true;
h.axial = false;     % if axial/radial polarised illumination is used
h.usemodulation = true;

img = zeros(512,512,7,'single');

for i = 1:7
%     img(:,:,i) = single(imread('/Users/maan/OneDrive - Imperial College London/Prochip/Polimi/63X_075.tif',i));
    img(:,:,i) = single(imread('/Users/maan/Documents/Office Projects/Prochip/HexSimProcessor/Tests/SIMdata_2019-11-05_15-21-42.tif',i));
end

h.calibrate(img);
imgo = h.reconstruct(img);
figure();
imshow(imgo,[])

figure(50);
imshow(log(abs(fftshift(fft2(h.sum_separated_comp(:,:,1))))+1),[])
figure(51);
imshow(log(abs(fftshift(fft2(h.sum_separated_comp(:,:,2))))+1),[])
figure(53);
imshow(log(abs(fftshift(fft2(h.sum_separated_comp(:,:,3))))+1),[])
figure(54);
imshow(log(abs(fftshift(fft2(h.sum_separated_comp(:,:,4))))+1),[])

%%
[lkx,lky] = meshgrid(h.k,h.k);
kr=sqrt(lkx.^2+lky.^2);

hpf = kr>h.eta/2;
otf = h.tf(kr);
hpf(kr<2) = hpf(kr<2)./otf(kr<2);
hpf(kr>2) = 0;
hpf = fftshift(hpf);

p0 = ifft2(fft2(h.sum_separated_comp(:,:,1)).*hpf);
p = ifft2(fft2(img - h.sum_separated_comp(:,:,1)/7).*hpf);
% p = ifft2(fft2(img).*hpf);

figure(61);
imshow(abs(fftshift(fft2(p0 .* p1))),[]);
[ixfz,Kx,Ky] = h.zoomf(p0 .* p(:,:,1),h.N,h.kx(1),h.ky(1),50,h.dk*h.N);
figure(62);
imshow(abs(ixfz),[]);
[ixfz,Kx,Ky] = h.zoomf(p0 .* p(:,:,1),h.N,h.kx(2),h.ky(2),50,h.dk*h.N);
figure(63);
imshow(abs(ixfz),[]);
[ixfz,Kx,Ky] = h.zoomf(p0 .* p(:,:,1),h.N,h.kx(3),h.ky(3),50,h.dk*h.N);
figure(64);
imshow(abs(ixfz),[]);

scaling = 1/sum(p0(:).*conj(p0(:)));
abs(ixfz(256,256))*scaling;
angle(ixfz(256,256));

phase_shift_to_xpeak = exp(-1i*h.kx(1)*(-h.N/2*h.dx:h.dx:h.N/2*h.dx-h.dx)*2*pi*h.NA/h.lambda); % may need to change the sign of kx
phase_shift_to_ypeak = exp(-1i*h.ky(1)*(-h.N/2*h.dx:h.dx:h.N/2*h.dx-h.dx)*2*pi*h.NA/h.lambda).';  % may need to change the sign of ky
pshift = phase_shift_to_ypeak * phase_shift_to_xpeak;
c1 = squeeze(sum(p .* (p0 .* pshift), [1,2]))' * scaling;
figure(70);
clf
plot(abs(c1));
figure(71);
clf
tp = (0:6)*2*pi/7;
a1 = unwrap(angle(c1) - tp) + tp - angle(c1(1));
plot(a1,'b-');
% plot(unwrap(angle(c1)),'b-.');
hold on
plot(tp, 'b--');

phase_shift_to_xpeak = exp(-1i*h.kx(2)*(-h.N/2*h.dx:h.dx:h.N/2*h.dx-h.dx)*2*pi*h.NA/h.lambda); % may need to change the sign of kx
phase_shift_to_ypeak = exp(-1i*h.ky(2)*(-h.N/2*h.dx:h.dx:h.N/2*h.dx-h.dx)*2*pi*h.NA/h.lambda).';  % may need to change the sign of ky
pshift = phase_shift_to_ypeak * phase_shift_to_xpeak;
c2 = squeeze(sum(p .* (p0 .* pshift), [1,2]))' * scaling;
figure(70);
hold on
plot(abs(c2));
figure(71);
hold on
tp = (0:6)*4*pi/7;
a2 = unwrap(angle(c2) - tp) + tp - angle(c2(1));
plot(a2,'g-');
% plot(unwrap(angle(c2)),'g-.');
plot(tp, 'g--');

phase_shift_to_xpeak = exp(-1i*h.kx(3)*(-h.N/2*h.dx:h.dx:h.N/2*h.dx-h.dx)*2*pi*h.NA/h.lambda); % may need to change the sign of kx
phase_shift_to_ypeak = exp(-1i*h.ky(3)*(-h.N/2*h.dx:h.dx:h.N/2*h.dx-h.dx)*2*pi*h.NA/h.lambda).';  % may need to change the sign of ky
pshift = phase_shift_to_ypeak * phase_shift_to_xpeak;
c3 = squeeze(sum(p .* (p0 .* pshift), [1,2]))' * scaling;
figure(70);
hold on
plot(abs(c3));
figure(71);
hold on
tp = (0:6)*6*pi/7;
a3 = unwrap(angle(c3) - tp) + tp - angle(c3(1));
plot(a3,'r-');
% plot(unwrap(angle(c3)),'r-.');
plot(tp, 'r--');

plot(a3-a2, 'b-.')
plot(a3-a1, 'g-.')
plot(a2+a1, 'r-.')

figure(75)
clf
plot(unwrap(angle(c2)));
hold on
plot(unwrap(angle(c3 .* conj(c1))));


figure(76)
clf
plot((a3 - a2 - a1)/pi)





            


