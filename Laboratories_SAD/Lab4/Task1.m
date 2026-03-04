%% LAB 4: Attitude Kinematics
%clc;
close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');

%% Data
%Parameters
Ix = 0.0700; %kg m^2
Iy = 0.0550; %kg m^2 
Iz = 0.0250; %kg m^2

I = diag([Ix, Iy, Iz]); %kg m^2

%Initial Conditions
w_0 = [0.45, 0.52, 0.55]'; %rad/s
A_0 = eye(3);

orthonormalize = 0; %1-desactivated   0-activated

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('L4_sim1', sim_options);

%% Computations and plots
N = length(result.tout);

%Angular velocities
figure();
plot(result.tout, result.w,'LineWidth',2)
grid on
grid minor
legend('wx','wy','wz');

%DCM Properties
err = zeros(N,1);
d = zeros(N,1);

for i=1:length(result.tout)
    err(i) = norm(result.A(:,:,i)'*result.A(:,:,i)-eye(3));
    d(i) = det(result.A(:,:,i));
end

figure();
subplot(2,1,1)
plot(result.tout, err,'LineWidth',2)
grid on
grid minor
legend('orthonormality');

subplot(2,1,2)
plot(result.tout, d,'LineWidth',2)
grid on
grid minor
legend('determinant');

%Euler angles for visualization
roll = zeros(N,1);
pitch = zeros(N,1);
yaw = zeros(N,1);

for i=1:N
    roll(i) = atan2(result.A(3,2,i),result.A(3,3,i));
    pitch(i) = atan2(-result.A(3,1,i),sqrt(result.A(3,2,i)^2+result.A(3,3,i)^2));
    yaw(i) = atan2(result.A(2,1,i),result.A(1,1,i));
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
legend('Angular momentum')
subplot(2,1,2)
plot(result.tout, T,'LineWidth',2);
grid on
grid minor
legend('Kinetic energy')

%% Animation
animate_cubo(result.tout,roll,pitch,yaw,2);
   






