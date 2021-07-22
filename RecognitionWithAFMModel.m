clear all;
close all;
clc;

set(0,'DefaultAxesFontSize',16,'DefaultAxesFontName','Times Cyr'); 
set(0,'DefaultTextFontSize',16,'DefaultTextFontName','Times Cyr');

%% parameters for system of oscillators
n1 = 100;                   % amount of oscillators

a=0.01;                     % damping
s=2*pi*4.32;                % spin torque
w_e=2*pi*1.75*10^9;         % anisotropy
w_ex=2*pi*27.5*10^12;       % exchange
jth=w_e/2/s;                % electric threshold current density
ji = 0.985*jth;             % electric current density
% ji = 1.25*jth;             % electric current density

% External source
w0 = 2*pi*25*10^9;
Am = jth * 0.05;
in_signal = Am;

%% Parameters for ODEs
n = 2 * n1;                 % amount of ODE's
p = 3;                      % amount of patterns

% Integration length
d = 0.1*10^(-13);
T = 1000*10^(-12);
Per = T/d;
tspan = 0:d:T;

v_ic = zeros(1, n1);
% initial conditions for cross
Xi1 = eye(sqrt(n1), sqrt(n1));
Xi1 = Xi1 + Xi1(1:sqrt(n1),end:-1:1);
Xi1 = Xi1 * (-2);
Xi1 = Xi1 + 1;
IC1 = Xi1;
IC1 = pi * (IC1 + 1)./(2);
IC1_0 = [reshape(IC1, [1,n1]),v_ic];
Xi1 = reshape(Xi1, [1,n1]);

% initial conditions for zero
Xi2 = zeros(sqrt(n1), sqrt(n1)) - 1;
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
IC2_0 = [reshape(IC2, [1,n1]),v_ic];
Xi2 = reshape(Xi2, [1,n1]);

% initial conditions for plus
Xi3 = zeros(sqrt(n1), sqrt(n1)) + 1;
Xi3(5, :) = -1;
Xi3(:, 5) = -1;
IC3 = Xi3;
IC3 = pi * (IC3 + 1)./(2);
IC3_0 = [reshape(IC3, [1,n1]),v_ic];
Xi3 = reshape(Xi3, [1,n1]);

%% Training (weights calculation)
Xi = cell(1, p);
Xi{1} = Xi1;
Xi{2} = Xi2;
Xi{3} = Xi3;
S = zeros(n1, n1);
for i = 1:n1
    for j = 1:n1
        summa = 0;
        for k = 1:p
            summa  = summa + Xi{k}(i) * Xi{k}(j);
        end
        S(i,j)=1/n1*summa * 10^(-4);
    end
end

%% Training with genetic algorithm (TO DO)
% options = optimoptions('ga');
% options = optimoptions(options,'HybridFcn', {  @fmincon [] });
% options = optimoptions(options,'Display', 'off');
% options = optimoptions(options,'PlotFcn', {  @gaplotbestf @gaplotscores });
% nvars = n1^2;
% lb = 10^(-5)*ones(1, n1^2);
% ub = 9*10^(-5)*ones(1,n1^2);
% [S,fval,exitflag,output,population,score] = ...
% ga(@(S)pattern_rec(a,w_ex,s,ji,in_signal,w0,w_e,n,S, tspan, IC1_0),nvars,[],[],[],[],lb,ub,[],[],options);
% disp(S)

%% Image recognition
IC = IC1;
noised_IC = awgn(IC, 1);
im = IC./pi;
noised_im = noised_IC; % ./pi;
IC = [reshape(noised_im, [1, n1]), v_ic];
% imshow(im);
% figure, imshow(noised_im)
[ts_cross,ys_cross] = ode15s(@(t, y) my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,S), tspan,IC);
% [ts_cross_t,ys_cross_t] = ode15s(@(t, y) my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,S), tspan,IC1_0);

IC = IC2;
noised_IC = awgn(IC, 1);
im = IC./pi;
noised_im = noised_IC; % ./pi;
IC = [reshape(noised_im, [1, n1]), v_ic];
% figure, imshow(im);
% figure, imshow(noised_im)
[ts_zero,ys_zero] = ode15s(@(t, y) my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,S), tspan,IC);
% [ts_zero_t,ys_zero_t] = ode15s(@(t, y) my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,S), tspan,IC2_0);

IC = IC3;
noised_IC = awgn(IC, 1);
im = IC./pi;
noised_im = noised_IC; % ./pi;
IC = [reshape(noised_im, [1, n1]), v_ic];
% figure, imshow(im);
% figure, imshow(noised_im)
[ts_plus,ys_plus] = ode15s(@(t, y) my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,S), tspan,IC);
% [ts_plus_t,ys_plus_t] = ode15s(@(t, y) my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,S), tspan,IC3_0);

%% Convertation of the ODE solver's output
pix_cross = zeros(size(ys_cross(:, 1:n1)));
pix_zero = zeros(size(ys_zero(:, 1:n1)));
pix_plus = zeros(size(ys_plus(:, 1:n1)));
for i = 1:n1
    for j = 1:n1
        % pix_t(:, i) = sin(ys_t(:, i)).* sin(ys_t(:, j));
        pix_cross(:, i) = (ys_cross(:, i) - ys_cross(:, j));
        pix_zero(:, i) = sin(ys_zero(:, i)).* sin(ys_zero(:, j));
        pix_plus(:, i) = sin(ys_plus(:, i)).* sin(ys_plus(:, j));
    end
end

%% Images demonstration
% l = length(tspan); 
% for i = 1:round(l/20):l*0.5
%      im = reshape(pix_cross(i, :), [sqrt(n1), sqrt(n1)]);
%      figure, imshow(im)
% end
% for i = 1:round(l/20):l*0.5
%      im = reshape(pix_zero(i, :), [sqrt(n1), sqrt(n1)]);
%      figure, imshow(im)
% end
% for i = 1:round(l/20):l*0.5
%      im = reshape(pix_plus(i, :), [sqrt(n1), sqrt(n1)]);
%      figure, imshow(im)
% end

%% Probably, the best images (to be verified)
% [pks, locs] = findpeaks(sin(ys_cross(:, 1)));
% l = length(locs);
% for i = 1:round(l/10):l
%     im = reshape(pix_cross(locs(end), :), [sqrt(n1), sqrt(n1)]);
%     figure, imshow(im)
% end

%% Phase images

% % figure('Color', 'white')
% %  plot(ts_cross_t*10^12, ys_cross_t(:,1), 'b', 'linewidth', 1.5);
% %  hold on;
% %  plot(ts_cross_t*10^12, ys_cross_t(:,2), 'r', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\varphi$', 'interpreter', 'latex');
% %  title('Cross');
% %  legend('Neuron 1 = 1', 'Neuron 2 = 0');
% % figure('Color', 'white')
% %  plot(ts_zero_t*10^12, ys_zero_t(:,1), 'b', 'linewidth', 1.5);
% %  hold on;
% %  plot(ts_zero_t*10^12, ys_zero_t(:,4), 'r', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\varphi$', 'interpreter', 'latex');
% %  title('Zero');
% %  legend('Neuron 1 = 0', 'Neuron 4 = 1');
% % figure('Color', 'white')
% %  plot(ts_plus_t*10^12, ys_plus_t(:,1), 'b', 'linewidth', 1.5);
% %  hold on;
% %  plot(ts_plus_t*10^12, ys_plus_t(:,5), 'r', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\varphi$', 'interpreter', 'latex');
% %  title('Plus');
% %  legend('Neuron 1 = 0', 'Neuron 5 = 1');
%  
% figure('Color', 'white')
%  plot(ts_cross*10^12, ys_cross(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_cross*10^12, ys_cross(:,2), 'r', 'linewidth', 1.5);
%  xlabel('Time (ps)', 'interpreter', 'latex');
%  ylabel('Recognized $\varphi$', 'interpreter', 'latex');
%  title('Cross');
%  legend('Neuron 1 = 1', 'Neuron 2 = 0');
% figure('Color', 'white')
%  plot(ts_zero*10^12, ys_zero(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_zero*10^12, ys_zero(:,4), 'r', 'linewidth', 1.5);
%  xlabel('Time (ps)', 'interpreter', 'latex');
%  ylabel('Recognized $\varphi$', 'interpreter', 'latex');
%  title('Zero');
%  legend('Neuron 1 = 0', 'Neuron 4 = 1');
% figure('Color', 'white')
%  plot(ts_plus*10^12, ys_plus(:,1), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_plus*10^12, ys_plus(:,5), 'r', 'linewidth', 1.5);
%  xlabel('Time (ps)', 'interpreter', 'latex');
%  ylabel('Recognized $\varphi$', 'interpreter', 'latex');
%  title('Plus');
%  legend('Neuron 1 = 0', 'Neuron 5 = 1');
 
%% Velocity images
% % figure('Color', 'white')
% %  plot(ts_cross_t*10^12, ys_cross_t(:,101), 'b', 'linewidth', 1.5);
% %  hold on;
% %  plot(ts_cross_t*10^12, ys_cross_t(:,102), 'r', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\dot\varphi$', 'interpreter', 'latex');
% %  title('Cross');
% %  legend('Neuron 1 = 1', 'Neuron 2 = 0');
% % figure('Color', 'white')
% %  plot(ts_zero_t*10^12, ys_zero_t(:,101), 'b', 'linewidth', 1.5);
% %  hold on;
% %  plot(ts_zero_t*10^12, ys_zero_t(:,104), 'r', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\dot\varphi$', 'interpreter', 'latex');
% %  title('Zero');
% %  legend('Neuron 1 = 0', 'Neuron 4 = 1');
% % figure('Color', 'white')
% %  plot(ts_plus_t*10^12, ys_plus_t(:,101), 'b', 'linewidth', 1.5);
% %  hold on;
% %  plot(ts_plus_t*10^12, ys_plus_t(:,105), 'r', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\dot\varphi$', 'interpreter', 'latex');
% %  title('Plus');
% %  legend('Neuron 1 = 0', 'Neuron 5 = 1');
 
% figure('Color', 'white')
%  plot(ts_cross*10^12, ys_cross(:,101), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_cross*10^12, ys_cross(:,102), 'r', 'linewidth', 1.5);
%  xlabel('Time (ps)', 'interpreter', 'latex');
%  ylabel('Recognized $\dot\varphi$', 'interpreter', 'latex');
%  title('Cross');
%  legend('Neuron 1 = 1', 'Neuron 2 = 0');
% figure('Color', 'white')
%  plot(ts_zero*10^12, ys_zero(:,101), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_zero*10^12, ys_zero(:,104), 'r', 'linewidth', 1.5);
%  xlabel('Time (ps)', 'interpreter', 'latex');
%  ylabel('Recognized $\dot\varphi$', 'interpreter', 'latex');
%  title('Zero');
%  legend('Neuron 1 = 0', 'Neuron 4 = 1');
% figure('Color', 'white')
%  plot(ts_plus*10^12, ys_plus(:,101), 'b', 'linewidth', 1.5);
%  hold on;
%  plot(ts_plus*10^12, ys_plus(:,105), 'r', 'linewidth', 1.5);
%  xlabel('Time (ps)', 'interpreter', 'latex');
%  ylabel('Recognized $\dot\varphi$', 'interpreter', 'latex');
%  title('Plus');
%  legend('Neuron 1 = 0', 'Neuron 5 = 1');
%  
 %% Phase differences images
% % figure('Color', 'white')
% %  plot(ts_cross_t*10^12, ys_cross_t(:,1) - ys_cross_t(:, 2), 'b', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\Delta\varphi$', 'interpreter', 'latex');
% %  title('Cross');
% %  legend('Neuron 1 = 1, neuron 2 = 0');
% % figure('Color', 'white')
% %  plot(ts_zero_t*10^12, ys_zero_t(:,1) - ys_zero_t(:, 4), 'b', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\Delta\varphi$', 'interpreter', 'latex');
% %  title('Zero');
% %  legend('Neuron 1 = 0, neuron 4 = 1');
% % figure('Color', 'white')
% %  plot(ts_plus_t*10^12, ys_plus_t(:,1) - ys_plus_t(:, 5), 'b', 'linewidth', 1.5);
% %  xlabel('Time (ps)', 'interpreter', 'latex');
% %  ylabel('Trained $\Delta\varphi$', 'interpreter', 'latex');
% %  title('Plus');
% %  legend('Neuron 1 = 0, neuron 5 = 1');
 
figure('Color', 'white')
 plot(ts_cross_t*10^12, ys_cross(:,1) - ys_cross(:, 2), 'b', 'linewidth', 1.5);
 xlabel('Time (ps)', 'interpreter', 'latex');
 ylabel('Recognized $\Delta\varphi$', 'interpreter', 'latex');
 title('Cross');
 % ylim([-pi - 0.5, pi + 0.5]);
 legend('Neuron 1 = 1, neuron 2 = 0');
figure('Color', 'white')
 plot(ts_zero_t*10^12, ys_zero(:,1) - ys_zero(:, 4), 'b', 'linewidth', 1.5);
 xlabel('Time (ps)', 'interpreter', 'latex');
 ylabel('Recognized $\Delta\varphi$', 'interpreter', 'latex');
 title('Zero');
 % ylim([-pi - 0.5, pi + 0.5]);
 legend('Neuron 1 = 0, neuron 4 = 1');
figure('Color', 'white')
 plot(ts_plus_t*10^12, ys_plus(:,1) - ys_plus(:, 5), 'b', 'linewidth', 1.5);
 xlabel('Time (ps)', 'interpreter', 'latex');
 ylabel('Recognized $\Delta\varphi$', 'interpreter', 'latex');
 title('Plus');
 % ylim([-pi - 0.5, pi + 0.5]);
 legend('Neuron 1 = 0, neuron 5 = 1');
 
%% Error estimation
% dphi_cross_s = zeros(n,n);
% dphi_zero_s = zeros(n,n);
% dphi_plus_s = zeros(n,n);
% for i = 1:n1
%     for j = 1:n1
%         dphi_cross_s(i,j) = abs(ys_cross(end, i) - ys_cross(end, j));
%         dphi_zero_s(i,j) = abs(ys_zero(end, i) - ys_zero(end, j));
%         dphi_plus_s(i,j) = abs(ys_plus(end, i) - ys_plus(end, j));
%     end
% end
% dphi_cross = zeros(n,n);
% dphi_zero = zeros(n,n);
% dphi_plus = zeros(n,n);
% for i = 1:n1
%     for j = 1:n1
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