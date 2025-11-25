%% LAB 4: Attitude Kinematics
clc;
%close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');

%% Data
%Parameters
Ix = 0.0700; %kg m^2
Iy = 0.0550; %kg m^2 
Iz = 0.0250; %kg m^2

I = diag([Ix, Iy, Iz]); %kg m^2

%Initial Conditions
w_0 = [0.3, 0.15, 0.6]'; %rad/s
q_0 = [0 0 0 1]';

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('L4_sim2', sim_options);

%% Computations and plots
N = length(result.tout);

%Angular velocities
figure();
plot(result.tout, result.w,'LineWidth',2)
grid on
grid minor
legend('wx','wy','wz');

%% Euler angles for visualization
roll = zeros(N,1);
pitch = zeros(N,1);
yaw = zeros(N,1);

for i=1:N
    q = [result.q(i,4) result.q(i,1:3)];
    eul = quat2eul(q,"ZYX");
    yaw(i) = eul(1);
    pitch(i) = eul(2);
    roll(i) = eul(3);
end

figure()
subplot(3,1,1)
plot(result.tout,roll,'LineWidth',2)
grid on
grid minor
legend('roll')

subplot(3,1,2)
plot(result.tout,pitch,'LineWidth',2)
grid on
grid minor
legend('pitch')

subplot(3,1,3)
plot(result.tout,yaw,'LineWidth',2)
grid on
grid minor
legend('yaw')


%% Validation of dynamics: 
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

%% Validation Quaternions kinematic
norma = zeros(N,1);

for i=1:length(result.tout)
    norma(i) = norm(result.q(i,:));
end

figure();
subplot(2,1,1)
plot(result.tout, norma,'LineWidth',2)
grid on
grid minor
legend('norm');

%% Animation
%animate_cubo(result.tout,roll,pitch,yaw,2);







