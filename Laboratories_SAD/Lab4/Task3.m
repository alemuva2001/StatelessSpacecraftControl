%% LAB 4: Attitude Kinematics
clc;
close all;
clear;
set(0,'DefaultFigureWindowStyle','alwaysontop');

% 1. Configuración del intérprete de LaTeX
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% 2. Configuración global de tamaños de fuente para "Papers"
set(groot, 'defaultAxesFontSize', 14);   % Tamaño de los números en los ejes
set(groot, 'defaultTextFontSize', 16);   % Tamaño de las etiquetas (labels)
set(groot, 'defaultLegendFontSize', 14); % Tamaño del texto en las leyendas

f1.Units = 'centimeters';
f1.Position = [2, 2, 8, 6]; % 8 cm de ancho por 6 de alto

%% Data
%Parameters
Ix = 0.0700; %kg m^2
Iy = 0.0550; %kg m^2 
Iz = 0.0250; %kg m^2
I = diag([Ix, Iy, Iz]); %kg m^2
%Initial Conditions
w_0 = [0.3, 0.15, 0.6]'; %rad/s
q_0 = [1 0 0 0]';
%Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = '0.1';
sim_options.StartTime = '0';
sim_options.StopTime = '100';

%% Model Simulation
result = sim('Model', sim_options);

%% Computations and plots
N = length(result.tout);

%Angular velocities
figure();
plot(result.tout, result.w, 'LineWidth', 2)
grid on
grid minor
% Uso de \textbf{} para negrita en LaTeX y un tamaño de título aún mayor
title('\textbf{Angular Velocities}', 'FontSize', 18)
xlabel('Time $t$ (s)')
ylabel('$\omega$ (rad/s)')
legend({'$\omega_x$', '$\omega_y$', '$\omega_z$'}, 'Location', 'best');

exportgraphics(gcf, 'w_AttSim.pdf', 'ContentType', 'vector');

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
plot(result.tout, roll, 'LineWidth', 2)
grid on
grid minor
title('\textbf{Euler Angles}', 'FontSize', 18)
ylabel('Roll $\phi$ (rad)')
legend('$\phi$')

subplot(3,1,2)
plot(result.tout, pitch, 'LineWidth', 2)
grid on
grid minor
ylabel('Pitch $\theta$ (rad)')
legend('$\theta$')

subplot(3,1,3)
plot(result.tout, yaw, 'LineWidth', 2)
grid on
grid minor
xlabel('Time $t$ (s)')
ylabel('Yaw $\psi$ (rad)')
legend('$\psi$')

exportgraphics(gcf, 'Euler_AttSim.pdf', 'ContentType', 'vector');

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
plot(result.tout, h_norm, 'LineWidth', 2);
grid on
grid minor
title('\textbf{Conservation of Angular Momentum}', 'FontSize', 18)
ylabel('$\| \mathbf{h} \|$ ($\mathrm{kg \cdot m^2/s}$)')
legend('$\| \mathbf{h} \|$')

subplot(2,1,2)
plot(result.tout, T, 'LineWidth', 2);
grid on
grid minor
title('\textbf{Conservation of Kinetic Energy}', 'FontSize', 18)
xlabel('Time $t$ (s)')
ylabel('Kinetic Energy $T$ (J)')
legend('$T$')

exportgraphics(gcf, 'Constants_AttSim.pdf', 'ContentType', 'vector');

%% Validation Quaternions kinematic
norma = zeros(N,1);
for i=1:length(result.tout)
    norma(i) = norm(result.q(i,:));
end

figure();
plot(result.tout, norma, 'LineWidth', 2)
grid on
grid minor
title('\textbf{Quaternion Kinematic Validation}', 'FontSize', 18)
xlabel('Time $t$ (s)')
ylabel('$\| \mathbf{q} \|$')
legend('Norm $\| \mathbf{q} \|$');

exportgraphics(gcf, 'qNorm_AttSim.pdf', 'ContentType', 'vector');

%% Quaternions
q = result.q;
tic;
figure();
subplot(4,1,1)
plot(result.tout, result.q(:,1), 'LineWidth', 1.5)
grid on
grid minor
title('\textbf{Quaternion Components}', 'FontSize', 18)
ylabel('$q_0$')

subplot(4,1,2)
plot(result.tout, result.q(:,2), 'LineWidth', 1.5)
grid on
grid minor
ylabel('$q_1$')

subplot(4,1,3)
plot(result.tout, result.q(:,3), 'LineWidth', 1.5)
grid on
grid minor
ylabel('$q_2$')

subplot(4,1,4)
plot(result.tout, result.q(:,4), 'LineWidth', 1.5)
grid on
grid minor
xlabel('Time $t$ (s)')
ylabel('$q_3$')

exportgraphics(gcf, 'q_AttSim.pdf', 'ContentType', 'vector');

fprintf("Quaternions: %.3f s\n",toc)