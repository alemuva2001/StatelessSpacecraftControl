%% LAB 3: Euler Equations
clc;
close all;
clear all;
set(0,'DefaultFigureWindowStyle','alwaysontop');

%% Data
%Parameters
Ix = 0.0700; %kg m^2
Iy = 0.055; %kg m^2 
Iz = 0.025; %kg m^2

I = diag([Ix, Iy, Iz]); %kg m^2

%Initial Conditions
w_0 = [0.45, 0.52, 0.52]'; %rad/s

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
plot(result.tout, result.w,'LineWidth',2)
grid on
grid minor
legend('wx','wy','wz');

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






