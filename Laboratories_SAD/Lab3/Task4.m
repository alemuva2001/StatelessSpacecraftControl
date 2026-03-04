%% LAB 3: Euler Equations IV
clc;
close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');
addpath(genpath('/Users/alejandromunozvazquez/Documents/Universidad/MII/POLIMI/2nd Course/Thesis/Code/StatelessSpacecraftControl/Laboratories_SAD'))

%% Data
%Parameters
Ix = 0.0700; %kg m^2
Iy = 0.0109; %kg m^2 
Iz = 0.0504; %kg m^2
Ir = 0.0050; %kg m^2

I = diag([Ix, Iy, Iz]); %kg m^2

%Initial Conditions
w_0 = [1e-6, 1e-6, 0.02]'; %rad/s
wr_0 = 2*pi; %rad/s

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.01';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('Task4_sim', sim_options);

%% Plots
figure();
subplot(1,3,1)
plot(result.tout, result.w(:,1),'LineWidth',2)
grid on
grid minor
legend('wx_i');

subplot(1,3,2)
plot(result.tout, result.w(:,2),'LineWidth',2)
hold on
grid on
grid minor
legend('wy_i');

subplot(1,3,3)
plot(result.tout, result.w(:,3),'LineWidth',2)
grid on
grid minor
legend('wz_i');

%% Validation: 
%Check if the angular momentum and kinetic energy preserve their value
N = length(result.tout);
h_norm = zeros(N,1);
T = zeros(N,1);

hw = [0; 0; Ir*wr_0];
Tw = 1/2*Ir*wr_0^2;

for i=1:N
    %Angular momentum conservation
    h = I*result.w(i,:)' + hw;
    h_norm(i) = norm(h);

    %Kinetic energy conservation
    T(i)= 1/2*result.w(i,:)*I*result.w(i,:)' + Tw;
end

figure('Name','Conservation');
subplot(2,1,1)
plot(result.tout, h_norm,'LineWidth',2);
grid on
grid minor
legend('Angular momentum')
subplot(2,1,2)
plot(result.tout, T,'LineWidth',2);
grid on
grid minor
legend('Kinetic Energy')





