clear all;
close all;
clc;

set(0,'DefaultAxesFontSize',16,'DefaultAxesFontName','Times Cyr'); 
set(0,'DefaultTextFontSize',16,'DefaultTextFontName','Times Cyr');

n = 10 * 10;    % amount of ODEs
p = 3;          % amount of patterns
Omega = 10;

% Integration length
d = 0.01;
T = 100;
Per = T/d;
tspan = 0:d:T;

%% IC for training
% initial conditions for cross
Xi1 = eye(sqrt(n), sqrt(n));
Xi1 = Xi1 + Xi1(1:sqrt(n),end:-1:1);
Xi1 = Xi1 * (-2);
Xi1 = Xi1 + 1;
IC1 = Xi1;
IC1 = pi * (IC1 + 1)./(2);
IC1_0 = reshape(IC1, [1,n]);
Xi1 = reshape(Xi1, [1,n]);

% initial conditions for zero
Xi2 = zeros(sqrt(n), sqrt(n)) - 1;
Xi2([1, 10], 5:6) = 1;
Xi2(5:6, [1, 10]) = 1;
v = [1 1 1 1];
D1 = diag(v, 5);
D1 = [[0; 0; 0; 0; 0; 0; 0; 0; 0], D1; [0 0 0 0 0 0 0 0 0 0]];
D2 = diag(v, -5);
D2 = [[0 0 0 0 0 0 0 0 0]; D2];
D2 = [D2, [0; 0; 0; 0; 0; 0; 0; 0; 0; 0]];
Xi2 = Xi2 + 2*D1 + 2*D2;
Xi2 = -1*(Xi2 + 2*D1(:, end:-1:1) + 2*D2(:, end:-1:1));
IC2 = Xi2;
IC2 = pi * (IC2 + 1)./(2);
IC2_0 = reshape(IC2, [1,n]);
Xi2 = reshape(Xi2, [1,n]);

% initial conditions for plus
Xi3 = zeros(sqrt(n), sqrt(n)) + 1;
Xi3(5, :) = -1;
Xi3(:, 5) = -1;
IC3 = Xi3;
IC3 = pi * (IC3 + 1)./(2);
IC3_0 = reshape(IC3, [1,n]);
Xi3 = reshape(Xi3, [1,n]);

%% Traiing (weights calculation)
Xi = cell(1, p);
Xi{1} = Xi1;
Xi{2} = Xi2;
Xi{3} = Xi3;
S = zeros(n, n);
for i = 1:n
    for j = 1:n
        summa = 0;
        for k = 1:p
            summa  = summa + Xi{k}(i) * Xi{k}(j);
        end
        S(i,j)=1/n*summa;
    end
end
% [ts_t,ys_t] = ode15s(@(t, y) my_sys(t,y,n,Omega,S), tspan,IC1_0);

%% Image recogniion
IC = IC1;
noised_IC = awgn(IC, 1);
im = IC./pi;
noised_im = noised_IC; % ./pi;
IC = reshape(noised_im, [1, n]);
imshow(im);
figure, imshow(noised_im)
[ts_cross,ys_cross] = ode15s(@(t, y) my_sys(t,y,n,Omega,S), tspan,IC);

IC = IC2;
noised_IC = awgn(IC, 2);
im = IC./pi;
noised_im = noised_IC; % ./pi;
IC = reshape(noised_im, [1, n]);
figure, imshow(im);
figure, imshow(noised_im)
[ts_zero,ys_zero] = ode15s(@(t, y) my_sys(t,y,n,Omega,S), tspan,IC);

IC = IC3;
noised_IC = awgn(IC, 1);
im = IC./pi;
noised_im = noised_IC; % ./pi;
IC = reshape(noised_im, [1, n]);
figure, imshow(im);
figure, imshow(noised_im)
[ts_plus,ys_plus] = ode15s(@(t, y) my_sys(t,y,n,Omega,S), tspan,IC);

%% Convertation of the ODE solver's output
% pix_t = zeros(size(ys_t));
pix_cross = zeros(size(ys_cross));
pix_zero = zeros(size(ys_zero));
pix_plus = zeros(size(ys_plus));
for i = 1:n
    for j = 1:n
        pix_cross(:, i) = sin(ys_cross(:, i)).* sin(ys_cross(:, j));
        pix_zero(:, i) = sin(ys_zero(:, i)).* sin(ys_zero(:, j));
        pix_plus(:, i) = sin(ys_plus(:, i)).* sin(ys_plus(:, j));
    end
end

%% Images demonstration
% l = length(tspan); 
% for i = 1:round(l/10):l
%     im = reshape(pix_cross(i, :), [sqrt(n), sqrt(n)]);
%     figure, imshow(im)
% end
% for i = 1:round(l/10):l
%     im = reshape(pix_zero(i, :), [sqrt(n), sqrt(n)]);
%     figure, imshow(im)
% end
% for i = 1:round(l/10):l
%     im = reshape(pix_plus(i, :), [sqrt(n), sqrt(n)]);
%     figure, imshow(im)
% end

%% Probably, the best images (to be verified)
% [pks, locs] = findpeaks(sin(ys_zero(:, 1)));
% for l = 1:length(locs)
%     im = reshape(pix_zero(locs(end), :), [sqrt(n), sqrt(n)]);
%      figure, imshow(im)
% end

%% sin(theta) images
figure('Color', 'white')
 plot(ts_cross, sin(ys_cross(:,1)), 'b', 'linewidth', 1.5);
 hold on;
 plot(ts_cross, sin(ys_cross(:,2)), 'r', 'linewidth', 1.5);
 xlabel('Time (s)', 'interpreter', 'latex');
 ylabel('Recognized $\sin\theta$', 'interpreter', 'latex');
 title('Cross');
 legend('Neuron 1=1', 'Neuron 2=0');
 ylim([-2, 2]);
 xlim([0, 10]);
figure('Color', 'white')
 plot(ts_zero, sin(ys_zero(:,1)), 'b', 'linewidth', 1.5);
 hold on;
 plot(ts_zero, sin(ys_zero(:,4)), 'r', 'linewidth', 1.5);
 xlabel('Time (s)', 'interpreter', 'latex');
 ylabel('Recognized $\sin\theta$', 'interpreter', 'latex');
 title('Zero');
 legend('Neuron 1=0', 'Neuron 4=1');
 ylim([-2, 2]);
 xlim([0, 10]);
figure('Color', 'white')
 plot(ts_plus, sin(ys_plus(:,1)), 'b', 'linewidth', 1.5);
 hold on;
 plot(ts_plus, sin(ys_plus(:,5)), 'r', 'linewidth', 1.5);
 xlabel('Time (s)', 'interpreter', 'latex');
 ylabel('Recognized $\sin\theta$', 'interpreter', 'latex');
 title('Plus');
 legend('Neuron 1=0', 'Neuron 5=1');
 ylim([-2, 2]);
 xlim([0, 10]);
 
%% Phase images
% phi = ys - 10*ts;                          
% phi_t = ys_t - 10*ts_t;                       
% figure('Color', 'white')
%  plot(ts, phi(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts, phi(:,2), 'r', 'linewidth', 1.5);
%  xlabel('Time (s)', 'interpreter', 'latex');
%  ylabel('Recognized $\varphi$', 'interpreter', 'latex');
%  legend('Neuron 1=0', 'Neuron 2=0');
%  ylim([-pi - 0.5, pi + 0.5]);
% figure('Color', 'white')
%  plot(ts_t, phi_t(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_t, phi_t(:,2), 'r', 'linewidth', 1.5);
%  xlabel('Time (s)', 'interpreter', 'latex');
%  ylabel('Trained $\varphi$', 'interpreter', 'latex');
%  legend('Neuron 1=1', 'Neuron 2=0');
%  ylim([-pi - 0.5, pi + 0.5]);
% figure('Color', 'white')
%  plot(ts, phi(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts, phi(:,10), 'r', 'linewidth', 1.5);
%  xlabel('Time (s)', 'interpreter', 'latex');
%  ylabel('Recognized $\varphi$', 'interpreter', 'latex');
%  legend('Neuron 1=1', 'Neuron 2=1');
%  ylim([-pi - 0.5, pi + 0.5]);
% figure('Color', 'white')
%  plot(ts_t, phi_t(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_t, phi_t(:,10), 'r', 'linewidth', 1.5);
%  xlabel('Time (s)', 'interpreter', 'latex');
%  ylabel('Trained $\varphi$', 'interpreter', 'latex');
%  legend('Neuron 1=1', 'Neuron 2=1');
%  ylim([-pi - 0.5, pi + 0.5]);

%% Phase differences images
phi_cross = ys_cross - 10*ts_cross;
phi_zero = ys_zero - 10*ts_zero;
phi_plus = ys_plus - 10*ts_plus;
% phi_t = ys_t - 10*ts_t;
figure('Color', 'white')
 plot(ts_cross, abs(phi_cross(:,1) - phi_cross(:,2)), 'm', 'linewidth', 1.5);
 hold on;
 plot(ts_cross, abs(phi_cross(:,1) - phi_cross(:,10)), 'c', 'linewidth', 1.5);
 xlabel('Time (s)', 'interpreter', 'latex');
 ylabel('Recognized $\Delta\varphi$', 'interpreter', 'latex');
 title('Cross')
 legend('Neuron 1=1, neuron 2=0', 'Neuron 1=1, neuron 10=1');
 ylim([-pi - 0.5, pi + 0.5]);
figure('Color', 'white')
 plot(ts_zero, abs(phi_zero(:,1) - phi_zero(:,4)), 'b', 'linewidth', 1.5);
 hold on;
 plot(ts_cross, abs(phi_zero(:,1) - phi_zero(:,10)), 'c', 'linewidth', 1.5);
 xlabel('Time (s)', 'interpreter', 'latex');
 ylabel('Recognized $\Delta\varphi$', 'interpreter', 'latex');
 title('Zero')
 legend('Neuron 1=0, neuron 4=1', 'Neuron 1=0, neuron 10=0');
 ylim([-pi - 0.5, pi + 0.5]);
figure('Color', 'white')
 plot(ts_plus, abs(phi_plus(:,1) - phi_plus(:,5)), 'b', 'linewidth', 1.5);
 hold on;
 plot(ts_plus, abs(phi_plus(:,1) - phi_plus(:,10)), 'b', 'linewidth', 1.5);
 xlabel('Time (s)', 'interpreter', 'latex');
 ylabel('Recognized $\Delta\varphi$', 'interpreter', 'latex');
 title('Plus')
 legend('Neuron 1=0, neuron 5=1', 'Neuron 1=0, neuron 10=0');
 ylim([-pi - 0.5, pi + 0.5]);

%% Error estimation
% phi_cross = ys_cross - 10*ts_cross;
% phi_zero = ys_zero - 10*ts_zero;
% phi_plus = ys_cross - 10*ts_plus;
% dphi_cross_s = zeros(n,n);
% dphi_zero_s = zeros(n,n);
% dphi_plus_s = zeros(n,n);
% for i = 1:n
%     for j = 1:n
%         dphi_cross_s(i,j) = abs(phi_cross(end, i) - phi_cross(end, j));
%         dphi_zero_s(i,j) = abs(phi_zero(end, i) - phi_zero(end, j));
%         dphi_plus_s(i,j) = abs(phi_plus(end, i) - phi_plus(end, j));
%     end
% end
% dphi_cross = zeros(n,n);
% dphi_zero = zeros(n,n);
% dphi_plus = zeros(n,n);
% for i = 1:n
%     for j = 1:n
%         dphi_cross(i,j) = abs(IC1_0(i) - IC1_0(j));
%         dphi_zero(i,j) = abs(IC2_0(i) - IC2_0(j));
%         dphi_plus(i,j) = abs(IC3_0(i) - IC3_0(j));
%     end
% end
% dx_cross = dphi_cross - dphi_cross_s;
% dx_zero = dphi_zero - dphi_zero_s;
% dx_plus = dphi_plus - dphi_plus_s;
% RMSE_cross = sqrt(sum(dx_cross.^2, 'all') / n^2);
% RMSE_zero = sqrt(sum(dx_zero.^2, 'all') / n^2);
% RMSE_plus = sqrt(sum(dx_plus.^2, 'all') / n^2);
% disp(RMSE_cross);
% disp(RMSE_zero);
% disp(RMSE_plus);
