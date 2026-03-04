%% LAB 5: Attitude Kinematics II
clc;
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
C = 2; %rad/s
w_0 = [C, 0.1, 0.1]'; %rad/s
eul_0 = [0, 0, 0]'; %rad

%We want the initial attitude such that h(0) || [1 0 0]'
H = norm(I*w_0);
h_b = I*w_0;
h_i = [H 0 0]';

eul_0(1) = asin(-h_b(2)/H);
eul_0(2) = 0;
eul_0(3) = atan2(h_b(3), h_b(1));

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.01';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('L5_sim1', sim_options);

%% Computations and plots
N = length(result.tout);

%Angular velocities
figure();
plot(result.tout, result.w,'LineWidth',2)
grid on
grid minor
legend('wx','wy','wz');

%Euler angles for visualization
roll = result.euler(:,1);
pitch = result.euler(:,2);
yaw = result.euler(:,3);

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

%DCM
err = zeros(N,1);
d = zeros(N,1);
A = zeros(3,3,N);

for i=1:length(result.tout)
    A(:,:,i) = [cos(yaw(i))*cos(roll(i))-sin(yaw(i))*sin(roll(i))*sin(pitch(i)) cos(yaw(i))*sin(roll(i))+sin(yaw(i))*cos(roll(i))*sin(pitch(i)) -sin(yaw(i))*cos(pitch(i));...
               -sin(roll(i))*cos(pitch(i)) cos(roll(i))*cos(pitch(i)) sin(pitch(i));...
                sin(yaw(i))*cos(roll(i))+cos(yaw(i))*sin(roll(i))*sin(pitch(i)) sin(yaw(i))*sin(roll(i))-cos(yaw(i))*cos(roll(i))*sin(pitch(i)) cos(pitch(i))*cos(yaw(i))];

    err(i) = norm(A(:,:,i)'*A(:,:,i)-eye(3));
    d(i) = det(A(:,:,i));
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

%Pointing error
pointing_error = zeros(N,1);

for i=1:N
    % El ángulo de error es arccos(A_11). Lo pasamos a grados.
    pointing_error(i) = acos(A(1,1,i)) * (180/pi); 
end

figure();
plot(result.tout, pointing_error, 'LineWidth', 2)
grid on; grid minor;
title('Pointing Error');
xlabel('Time [s]');
ylabel('Pointing Error [deg]');

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
animate_cubo(result.tout,roll,pitch,yaw,1);







