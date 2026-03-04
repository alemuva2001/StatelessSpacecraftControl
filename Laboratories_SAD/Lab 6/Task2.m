%% LAB 6: Gravity Gradient perturbation
%clc;
close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');

%% Data
%Parameters
Ix = [0.0080 0.080 0.060 0.060 0.040 0.040]; %kg m^2
Iy = [0.0060 0.040 0.080 0.040 0.060 0.080]; %kg m^2 
Iz = [0.0040 0.060 0.040 0.080 0.080 0.060]; %kg m^2

%Notice that the gravity gradient only stabilizes the system if the inertia
%of the axis that points towards the center of the massive body has the minimal inertia
I = diag([Ix(3), Iy(3), Iz(3)]); %kg m^2

%Earth properties
G = 6.673e-20; %km^3/s^2
M_t = 5.973e24; %kg
Rt = 6378.137; %km
H = 500; %km
R = Rt+H;

n = sqrt(G*M_t/R^3);

%Initial Conditions
%w_0 = [0, 0, n]'; %rad/s
w_0 = [1e-6, n, 1e-6]'; %rad/s
A_0 = eye(3);

orthonormalize = 0; %1-desactivated   0-activated

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = '6000';

%% Model Simulation
result = sim('L6_sim2', sim_options);

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

%Attitude error

w_BL_error = zeros(3,N);
att_error_norm = zeros(N,1);
w_LN = [0 0 n]';

for i = 1:N
    %Attitude error
    A_BL = result.A_BN(:,:,i)*result.A_LN(:,:,i)';
    att_error_norm(i) = norm(result.A_BL(:,:,i)-eye(3));

    %Angular velocity error
    w_BL_error(:,i) = (result.w(i,:)' - result.A_BL(:,:,i)*w_LN)';
end

figure('Name','Pointing Error')
subplot(2,1,1)
plot(result.tout, att_error_norm,'LineWidth',2);
grid on
grid minor
legend('Attitude error')
subplot(2,1,2)
plot(result.tout, w_BL_error, 'LineWidth',2);
grid on
grid minor
legend('\omega_x error', '\omega_y error', '\omega_z error');


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
animate_cubo(result.tout,roll,pitch,yaw,100);
   






