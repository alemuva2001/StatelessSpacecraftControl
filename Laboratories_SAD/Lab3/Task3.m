%% LAB 3: Euler Equations
clc;
close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');
addpath(genpath('/Users/alejandromunozvazquez/Documents/Universidad/MII/POLIMI/2nd Course/Thesis/Code/StatelessSpacecraftControl/Laboratories_SAD'))

%% Data
%Parameters
Ix = 0.0250; %kg m^2
Iy = 0.0550; %kg m^2 
Iz = 0.0700; %kg m^2

I = diag([Ix, Iy, Iz]); %kg m^2

%Initial Conditions
w_0 = [0.1, 0.1, 2]'; %rad/s

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('Task1_sim', sim_options);

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

for i=1:N
    %Angular momentum conservation
    h = I*result.w(i,:)';
    h_norm(i) = norm(h);

    %Kinetic energy conservation
    T(i)= 1/2*result.w(i,:)*I*result.w(i,:)';
end

figure();
subplot(2,1,1)
plot(result.tout, h_norm,'LineWidth',2);
grid on
grid minor
subplot(2,1,2)
plot(result.tout, T,'LineWidth',2);
grid on
grid minor





