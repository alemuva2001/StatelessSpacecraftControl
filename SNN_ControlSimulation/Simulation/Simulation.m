%clc;
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

%% %%%%%%%%%%%%%%%%%%%%%%
%%-TAREAS PARA MAÑANA-%%
%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Configurar SNN
USE_SNN = 1;
% 
%% Data
% Sensor Parameters
FOV = deg2rad(60);
W = 1024/4; % Horizontal resolution [px]
H = 1024/4; % Vertical resolution [px]
f = W/(2*tan(FOV/2)); % Focal distance
sigma = 1;
cx = W/2;
cy = H/2;

% Event sensor parameters
k_ev1 = 5;
k_ev2 = 1;
Th_ON = 5;
Th_OFF = -5;

% CameraBody Parameters
Ix = [0.080 0.080 0.060 0.060 0.040 0.040]; %[kg*m^2]
Iy = [0.060 0.040 0.080 0.040 0.060 0.080]; %[kg*m^2]
Iz = [0.040 0.060 0.040 0.080 0.080 0.060]; %[kg*m^2]

I = diag([Ix(3), Iy(3), Iz(3)])/10; %[kg*m^2]

% Initial conditions
w_0 = [0.0, 0.0, 0.0]'; %[rad/s]

%w_0 = -0.05 + (0.05 + 0.05).*rand(3, 1)

Euler = deg2rad([0,0,0]); %-89+(180)*rand(3, 1)); %

roll = Euler(1); 
pitch = Euler(2);
yaw = Euler(3);

R_c = eul2rotm([yaw, pitch, roll]); % Camera attitude in the space
q_0 = eul2quat([yaw, pitch, roll]); % Initial attitude
A_0 = R_c;

P0 = diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]); % Covariance matrix initialization

% Stars Generation
r = 200;
[stars, s_intensity] = generateStars(r, 1000, 5000 );

% stars_ = load("Starfield_Test.mat");
% s_intensity_ = load("S_intensity_Test.mat");
% stars = stars_.stars;
% s_intensity = s_intensity_.s_intensity;

% visualizeCelestialSpace(stars, s_intensity)

figure('Name','Star Field')
gcf.Units = 'centimeters';
gcf.Position = [2, 2, 12, 8];
scatter3(stars(:,1),stars(:,2),stars(:,3),50*s_intensity,'filled')
axis('equal')
hold on; grid on; grid minor;
drawCamera(R_c);
title('\textbf{Celestial Sphere}', 'FontSize', 18);
xlabel('Coordinate $X$', 'FontSize', 16);
ylabel('Coordinate $Y$', 'FontSize', 16);
zlabel('Coordinate $Z$', 'FontSize', 16);
% exportgraphics(gca, 'Celestial_Sphere.pdf', 'ContentType', 'vector');

% Controller Parameters
wn = 0.2; zeta = sqrt(2)/2;
Ix = I(1,1); Iy = I(2,2) ; Iz = I(3,3);
% Ix = 0; Iy = 0; Iz = 0;

Kp_x = Ix * (wn^2);       Kd_x = Ix * 2 * zeta * wn;    Ki_x = 0;%wn/10*Kp_x; 
Kp_y = Iy * (wn^2);       Kd_y = Iy * 2 * zeta * wn;    Ki_y = 0;%wn/10*Kp_y;
Kp_z = Iz * (wn^2);       Kd_z = Iz * 2 * zeta * wn;    Ki_z = 0;%wn/10*Kp_z;

% Simulation options
sim_time = 100;        % Para perturbaciones
sim_time_ = sim_time;  % Para simulacion
dt = 0.1;

%Perturbations
p1 = randi([-8, 8], 3, 1);
p2 = randi([-8, 8], 3, 1);
p3 = randi([-8, 8], 3, 1);
p_val = [zeros(3,2), p1, zeros(3,2), p2, zeros(3,2), p3, zeros(3,2)] * 1e-3

pt1 = randi([5, int32(sim_time/3-5)]);
pt2 = randi([int32(sim_time/3+5), int32(sim_time/3+sim_time/3-5)]);
pt3 = randi([int32(2*sim_time/3+5), int32(sim_time-20)]);
p_time = [0 pt1-0.1 pt1 pt1+0.1 pt2-0.1 pt2 pt2+0.1 pt3-0.1 pt3 pt3+0.1 1000]

% p_val = [0 0  2e-4 0 0 -5e-4 0 0  7e-4 0 0;
%          0 0 -6e-4 0 0  7e-4 0 0  3e-4 0 0;
%          0 0 -3e-4 0 0  6e-4 0 0  6e-4 0 0]*4;
% p_time = [0	8.9	9 9.1 46.9 47 47.1 65.9	66 66.1	1000];
% pt1 = p_time(3);
% pt2 = p_time(6);
% pt3 = p_time(9);

sim_options.SolverType = 'Fixed-step';
sim_options.Solver = 'ode4';
sim_options.FixedStep = string(dt);
sim_options.StartTime = '0';
sim_options.StopTime = 'sim_time_';

%% Model Simulation
tic;
result = sim('Model.slx', sim_options);
fprintf("Simulation complete: %.3f s\n",toc)

%% Results

N = length(stars);
time_max = length(result.tout);
stars_c = result.starsSeen;
u = zeros(N,time_max);
v = zeros(N,time_max);


% Search for stars in the image
tic
for t = 1:time_max
    % Extraer slice una sola vez
    frame = result.image(:,:,t);
    
    % Indexación lógica vectorizada
    [rows, cols] = find(frame > 160);
    
    % Limitar a N detecciones máximas
    n_detected = min(length(rows), N);
    
    u(1:n_detected, t) = cols(1:n_detected);
    v(1:n_detected, t) = rows(1:n_detected);
    
    %fprintf('Instante %d/%d — %d puntos detectados\n', t, time_max, n_detected);
end

fprintf('Stars search: %.4f s\n', toc);
%
% tic
% k = 1;
% for t=1:time_max
%     for i = 1:N
%         if stars_c(i,3) > 0
%             if (result.u(t,i)>0 && result.u(t,i)<=W) && (result.v(t,i)>0 && result.v(t,i)<=H)
%                 u(k,t) = result.u(t,i);
%                 v(k,t) = result.v(t,i);
%                 k = k + 1;
%             end
%         end
%     end
%     k = 1;
% end
% disp(toc)

%% -- Captured Image --
% tic
% figure('Name','Captured Image')
% hold on;
% for t=1:time_max
%     intensity = t/time_max;
%     color = [min([1-intensity,0.8]), min([1-intensity,0.8]), min([1-intensity,0.8])];
%     plot(u(:,t),v(:,t),'LineStyle','none','Marker','.','Color',color)
%     axis([0,W,0,H])
% end
% plot(u(:,1),v(:,1),'LineStyle','none','Marker','*','Color','r')
% plot(u(:,end),v(:,end),'LineStyle','none','Marker','*','Color','g')
% legend('Trajectory', 'Initial', 'End','Location','best')
% set(gca, 'YDir', 'reverse');
% daspect([1 1 1])
% fprintf("Sequence of captured images: %.3f s\n",toc)

% tic
% figure('Name','Image motion over time')
% plot3(result.tout,u,v,'Marker','.','MarkerSize',1,'LineStyle','none')
% axis([0,time_max,0,W,0,H])
% daspect([1 1 1])
% set(gca, 'ZDir', 'reverse');
% set(gca, 'YDir', 'reverse');
% hold on; grid on; grid minor;
% fprintf("Time-sequence: %.3f s\n",toc)

%% -- Prediction comparison --
% u = result.u';
% v = result.v';

u_pred = result.u_meas';
v_pred = result.v_meas';

tic
figure('Name','Prediction Image')
hold on;

% Bucle de ploteo de trayectorias
for t=1:time_max
    intensity = t/time_max;
    color = [min([1-intensity,0.8]), min([1-intensity,0.8]), min([1-intensity,0.8])];
    plot(u(:,t),v(:,t),'LineStyle','none','Marker','.','Color',color)

    color_pred = [1, min([1-intensity,0.8]), min([1-intensity,0.8])];
    plot(u_pred(:,t),v_pred(:,t),'LineStyle','none','Marker','.','Color',color_pred)
end

% Definir los límites de los ejes UNA SOLA VEZ fuera del bucle
axis([0,W,0,H]) 

% Marcadores de inicio y fin
plot(u(:,1),v(:,1),'LineStyle','none','Marker','*','Color','r')
plot(u(:,end),v(:,end),'LineStyle','none','Marker','*','Color','g')

plot(u_pred(:,2),v_pred(:,2),'LineStyle','none','Marker','o','Color','r')
plot(u_pred(:,end),v_pred(:,end),'LineStyle','none','Marker','o','Color','g')

% Estilos de texto compatibles con LaTeX
title('\textbf{Real vs Predicted Trajectory}', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$v$ [px]', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$u$ [px]', 'Interpreter', 'latex', 'FontSize', 16);

% --- CREACIÓN DE DUMMY HANDLES PARA LA LEYENDA ---
% Dibujamos elementos invisibles (en coordenadas NaN, NaN) con el estilo deseado
h_real = plot(NaN, NaN, '-', 'Color', 'k', 'LineWidth', 1.5);
h_pred = plot(NaN, NaN, '-', 'Color', 'r', 'LineWidth', 1.5);
h_init_real = plot(NaN, NaN, '*', 'Color', 'r', 'MarkerSize', 8);
h_end_real  = plot(NaN, NaN, '*', 'Color', 'g', 'MarkerSize', 8);
h_init_pred = plot(NaN, NaN, 'o', 'Color', 'r', 'MarkerSize', 8);
h_end_pred  = plot(NaN, NaN, 'o', 'Color', 'g', 'MarkerSize', 8);

% Llamamos a la leyenda pasándole específicamente estos manejadores
legend([h_real, h_pred, h_init_real, h_end_real, h_init_pred, h_end_pred], ...
    {'Real trajectory', 'Predicted trajectory', ...
     'Real initial position', 'Real final position', 'Predicted initial position', 'Predicted final position'}, ...
    'Location', 'eastoutside', 'Interpreter', 'latex', 'FontSize', 11);

set(gca, 'YDir', 'reverse', 'TickLabelInterpreter', 'latex');
daspect([1 1 1])

%exportgraphics(gca, 'PredictionImage.pdf', 'ContentType', 'vector');

fprintf("Prediction comparison: %.3f s\n",toc)


%% -- Angular velocities --
tic;
figure();
plot(result.tout, result.w,'LineWidth',2)
hold on
plot(result.tout, result.w_corr,'LineWidth',2,'LineStyle','--')
grid on
grid minor

title("\textbf{Angular Velocities}")
ylabel("$\omega [rad/s]$")
xlabel("Time $[s]$")

legend('$w_x$','$w_y$','$w_z$','$w_{pred,x}$', '$w_{pred,y}$', '$w_{pred,z}$');
%exportgraphics(gca, 'AngularVelocities.pdf', 'ContentType', 'vector');
fprintf("Angular Velocities: %.3f s\n",toc)

%% -- Euler Angles --
% Euler angles for visualization from DCM
roll = zeros(time_max,1);
pitch = zeros(time_max,1);
yaw = zeros(time_max,1);
eul = zeros(3,1);

tic
for i=1:time_max
    eul = rotm2eul(result.A_BN(:,:,i));
    roll(i) = eul(3); %atan2(result.A_BN(3,2,i),result.A_BN(3,3,i));
    pitch(i) = eul(2); %atan2(-result.A_BN(3,1,i),sqrt(result.A_BN(3,2,i)^2+result.A_BN(3,3,i)^2));
    yaw(i) = eul(1); %atan2(result.A_BN(2,1,i),result.A_BN(1,1,i));
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

fprintf("Euler angles: %.3f s\n",toc)

%% Quaterniones 

tic;
f_quat = figure();
subplot(4,1,1)
plot(result.tout, result.q(:,1),'LineWidth',1.5)
hold on
plot(result.tout, -result.q_corr(:,1),'LineStyle','--','LineWidth',1.5)
grid on; grid minor;
title("\textbf{Quaternions}")
ylabel("$q_0$")
legend('Real', 'Predicted', 'Location','best')

subplot(4,1,2)
plot(result.tout,result.q(:,2),'LineWidth',1.5)
hold on
plot(result.tout, result.q_corr(:,2),'LineStyle','--','LineWidth',1.5)
grid on; grid minor;
ylabel("$q_1$")

subplot(4,1,3)
plot(result.tout, result.q(:,3),'LineWidth',1.5)
hold on
plot(result.tout, result.q_corr(:,3),'LineStyle','--','LineWidth',1.5)
grid on; grid minor;
ylabel("$q_2$")

subplot(4,1,4)
plot(result.tout, result.q(:,4),'LineWidth',1.5)
hold on
plot(result.tout, result.q_corr(:,4),'LineStyle','--','LineWidth',1.5)
grid on; grid minor;
ylabel("$q_3$")
xlabel("Time $[s]$")

%exportgraphics(f_quat, 'Quaternions.pdf', 'ContentType', 'vector');
fprintf("Quaternions: %.3f s\n",toc)

%% Error y T_control

tic;
f_ErrorTorque = figure();

ax1 = subplot(2, 3, [1, 2, 3]); % Guardamos el handle en ax1
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt2, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt3, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(result.tout, result.error(:,2:4), 'LineWidth', 1.5)
grid on; grid minor;
title('\textbf{Quaternion error}','FontSize',18)
xlabel("Time $[s]$")
legend('$e_{q2}$', '$e_{q3}$', '$e_{q4}$', 'Location', 'best')
hold off

%%--- EJE X ---
ax2 = subplot(2, 3, 4); % Guardamos el handle en ax2
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt2, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt3, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Extraemos la columna 1 (X)
plot(result.tout, result.T_control_PID(:, 1), 'LineWidth', 1.5, 'Color', '#0072BD') % Azul
if USE_SNN
    plot(result.tout, result.T_control_SNN(:, 1), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', '#D95319') % Naranja rojizo
end
grid on; grid minor;
title('\textbf{Torque Eje X (Roll)}');
ylabel("$[N]$")
xlabel("Time $[s]$")
legend('$T_{x}$ PID', '$T_{x}$ SNN', 'Location', 'best')
hold off

%%--- EJE Y ---
ax3 = subplot(2, 3, 5); % Guardamos el handle en ax3
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt2, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt3, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Extraemos la columna 2 (Y)
plot(result.tout, result.T_control_PID(:, 2), 'LineWidth', 1.5, 'Color', '#0072BD')
if USE_SNN 
    plot(result.tout, result.T_control_SNN(:, 2), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', '#D95319')
end
grid on; grid minor;
title('\textbf{Torque Eje Y (Pitch)}');
ylabel("$[N]$")
xlabel("Time $[s]$")
legend('$T_{y}$ PID', '$T_{y}$ SNN', 'Location', 'best')
hold off

%%--- EJE Z ---
ax4 = subplot(2, 3, 6); % Guardamos el handle en ax4
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt2, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(pt3, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Extraemos la columna 3 (Z)
plot(result.tout, result.T_control_PID(:, 3), 'LineWidth', 1.5, 'Color', '#0072BD')
if USE_SNN
    plot(result.tout, result.T_control_SNN(:, 3), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', '#D95319')
end
grid on; grid minor;
title('\textbf{Torque Eje Z (Yaw)}');
ylabel("$[N]$")
xlabel("Time $[s]$")
legend('$T_{z}$ PID', '$T_{z}$ SNN', 'Location', 'best')
hold off

%%--- VINCULAR EJES ---
% Sincroniza el zoom y el desplazamiento solo en el eje del tiempo (x)
linkaxes([ax1, ax2, ax3, ax4], 'x');

exportgraphics(f_ErrorTorque, 'ErrorTorque.pdf', 'ContentType', 'vector');

fprintf("Error y T_control: %.3f s\n", toc)
%% Representacion de eventos
% 1. Encontrar los índices lineales donde hay eventos (valores 'true')
% Al ser lógica, 'find' extrae directamente los 'true' sin necesidad de comparar
result.Events_ON(:,:,1) = zeros(size(W,H));
idx_ON = find(result.Events_ON);
idx_OFF = find(result.Events_OFF);
% 2. Convertir los índices lineales a coordenadas 3D (y, x, t)
[y_ON, x_ON, t_ON] = ind2sub(size(result.Events_ON), idx_ON);
[y_OFF, x_OFF, t_OFF] = ind2sub(size(result.Events_ON), idx_OFF);
% 3. Crear la figura y representarla en 3D
figure('Name', 'Events Image');
plot3(t_ON, x_ON, y_ON, 'b','LineStyle','none','Marker','.');
hold on
plot3(t_OFF, x_OFF, y_OFF, 'r','LineStyle','none','Marker','.');
axis([0,time_max,0,W,0,H])
daspect([1 1 1])
set(gca, 'ZDir', 'reverse');
% set(gca, 'YDir', 'reverse');
set(gca, 'XDir', 'reverse');
hold on; grid on; grid minor;
xlabel('Time [s]')


%%
% dV_acc = squeeze(result.dV_acc(120, 162, :));
% figure();
% subplot(2,1,1)
% stairs(result.tout, dV_acc,'LineWidth',1)
% title('dV Accumulated');
% subplot(2,1,2)
% plot(result.tout, squeeze(result.RealImage(squeeze(u(1,end)),squeeze(v(1,end)),:)),'LineWidth',1)
% title('Image R pixel value');

%% Animation
% pause;
% figure()
% 
% for t = 1:time_max
%     delete(line);
% 
%     plot(u(:,t),v(:,t),'LineStyle','none','Marker','.','Color','b')
%     axis([0,W,0,H])
%     set(gca, 'YDir', 'reverse');
%     daspect([1 1 1])
% 
%     title(sprintf('Tiempo transcurrido: %.2f s', t*dt));
% 
%     F(t)=getframe(gcf);
% end
% 
% F=F(2:time_max-1);
% hold off
% 
% %Video Creation
% video = VideoWriter('test1', 'MPEG-4');
% video.FrameRate = 30;
% open(video)
% writeVideo(video, F());
% close(video);
