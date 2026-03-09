%% LAB 7: Perturbations
%clc;
close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');

%% Data
%Parameters
Ix = [0.0080 0.080 0.060 0.060 0.040 0.040]; %kg m^2
Iy = [0.0060 0.040 0.080 0.040 0.060 0.080]; %kg m^2 
Iz = [0.0040 0.060 0.040 0.080 0.080 0.060]; %kg m^2

I = diag([Ix(5), Iy(5), Iz(5)]); %kg m^2

%Earth properties
G = 6.673e-20; %km^3/s^2
M_t = 5.973e24; %kg
mu = G*M_t; %km^3*kg/s^2
Rt = 6378.137; %km
H = 420; %km

%Orbital Parameters
H_pe = 500; %km
H_ap = 5000; %km
Rp = Rt+H_pe;
Ra = Rt+H_ap;

e = (Ra-Rp)/(Ra+Rp); %excentricity
a = ((Ra+Rp)/2)*1e3; %[km] major axis

i = deg2rad(28.5); %[rad] inclination

n = sqrt(G*M_t/(a/1e3)^3); %rad/s
mu_o = n^2*a^3; %m^3*kg/s^2

%Sun parameters
T_year = 3600*24*365.25; %Earth period [s]
n_sun = 2*pi/T_year; %Earth rotation velocity
eps_rad = deg2rad(23.45); %Earth ecliptic

%Spacecraft Geometry
T = readtable('satellite_geometry.xlsx');
satellite_data.Normals = [T.Nx, T.Ny, T.Nz]';
satellite_data.Positions = [T.rx, T.ry, T.rz]';
satellite_data.Areas = T.Area;
satellite_data.RhoS = T.rho_s;
satellite_data.RhoD = T.rho_d;

r_cm = [0; 0; 0.02]; % Center of mass

%Radiation
Fe = 1358; %Solar radiation intensity (W/m^2)
c = 299792458; %Light speed (m/s)
P = Fe / c; % Solar Radiation Pressure 

% Deployed inertia matrix (kg m^2)
J_depl = diag([100.9, 25.1, 91.6]) * 1e-2;
I = J_depl;

%Initial Conditions
w_0 = [1e-6, 1e-6, n]'; %rad/s
A_0 = eye(3);

orthonormalize = 0; %1-desactivated   0-activated

N = 1; %Number of orbits to simulate
sim_time = N*2*pi/n;

%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = 'sim_time';

%% Model Simulation
result = sim('L7_sim2', sim_options);

%% Computations and plots
N = length(result.tout);

%Angular velocities
figure('Name','Angular Velocity');
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

figure('Name','DCM properties');
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

for i = 1:N
    t = result.tout(i);
    w_B = result.w(i,:)';
    A_BN = result.A(:,:,i);

    A_LN = [cos(n*t) sin(n*t) 0;...
           -sin(n*t) cos(n*t) 0;...
            0        0        1];
    w_LN = [0 0 n]';
    
    %Attitude error
    A_BL = A_BN*A_LN';
    att_error_norm(i) = norm(A_BL-eye(3));

    %Angular velocity error
    w_BL_error(:,i) = (w_B - A_BL*w_LN)';
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

%External torques
T_ext_representation(result.tout, result.T_tot, result.T_srp)

%% Visualization
%Orbit visualization
orbit_visualization(result.tout,result.r_N, Rt)

%Euler angles for visualization from DCM
euler_visualization(result.tout, result.A)

%% Validation: 
att_validation(result.tout, I, result.w);
orb_validation(result.tout, result.r_N, result.r_Nd, mu_o)

%% Animation
%animate_cubo(result.tout,roll,pitch,yaw,100);
animate_system(result.tout, result.r_N, result.A, result.S_N, Rt, 100);
   




