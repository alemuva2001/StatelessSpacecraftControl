%% LAB 3: Euler Equations
clc;
close all;
clear all;
set(0,'DefaultFigureWindowStyle','alwaysontop');

%% Data
%Parameters
Ix = 0.0504; %kg m^2
Iy = 0.0504; %kg m^2 
Iz = 0.0109; %kg m^2

I = diag([Ix, Iy, Iz]); %kg m^2

%Initial Conditions
wx_0 = 0.45; %rad/s
wy_0 = 0.45; %rad/s
wz_0 = 0.45; %rad/s

w_0 = [wx_0, wy_0, wz_0]'; %rad/s

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('Task1_sim', sim_options); %Governing equations

%Analytic solution
lambda = (Iz-Ix)/Iy*wz_0;

N = length(result.tout);
wx = zeros(N,1);
wy = zeros(N,1);
wz = zeros(N,1);
i=1;

for t=0:0.1:100
    wx(i) = wx_0*cos(lambda*t) - wy_0*sin(lambda*t);
    wy(i) = wx_0*sin(lambda*t) + wy_0*cos(lambda*t);
    wz(i) = wz_0;
    i = i + 1;
end

%% Plots
figure();
subplot(2,3,1)
plot(result.tout, result.w(:,1),'LineWidth',2)
hold on
plot(result.tout, wx,'LineWidth',2)
grid on
grid minor
legend('wx_i','wx_a');
subplot(2,3,4)
plot(result.tout, result.w(:,1)-wx,'LineWidth',2)
grid on
grid minor
legend('error wx');

subplot(2,3,2)
plot(result.tout, result.w(:,2),'LineWidth',2)
hold on
plot(result.tout, wy,'LineWidth',2)
grid on
grid minor
legend('wy_i','wy_a');
subplot(2,3,5)
plot(result.tout, result.w(:,2)-wy,'LineWidth',2)
grid on
grid minor
legend('error wy');

subplot(2,3,3)
plot(result.tout, result.w(:,3),'LineWidth',2)
hold on
plot(result.tout, wz,'LineWidth',2)
grid on
grid minor
legend('wz_i','wz_a');
subplot(2,3,6)
plot(result.tout, result.w(:,3)-wz,'LineWidth',2)
grid on
grid minor
legend('error wz');

% %% Validation: 
% %Check if the angular momentum and kinetic energy preserve their value
% N = length(result.tout);
% h_norm = zeros(N,1);
% T = zeros(N,1);
% 
% for i=1:N
%     %Angular momentum conservation
%     h = I*result.w(i,:)';
%     h_norm(i) = norm(h);
% 
%     %Kinetic energy conservation
%     T(i)= 1/2*result.w(i,:)*I*result.w(i,:)';
% end
% 
% figure();
% subplot(2,1,1)
% plot(result.tout, h_norm,'LineWidth',2);
% grid on
% grid minor
% subplot(2,1,2)
% plot(result.tout, T,'LineWidth',2);
% grid on
% grid minor
% 




