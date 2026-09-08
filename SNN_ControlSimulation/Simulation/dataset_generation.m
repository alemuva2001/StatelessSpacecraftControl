% generate_dataset.m
% First approach: fixed inertia, variable initial attitude + variable w_0
clc; close all; clear;
set(0, 'DefaultFigureWindowStyle', 'normal');
tic;
% =========================================================================
% CONFIGURATION
% =========================================================================
initial_episode = 1;       % Episodio por el que empezar a numerar
num_episodes    = 150;       % Cuántos episodios generar en esta sesión
sim_time        = 100;      % Perturbations
sim_time_       = 120;      % Simulation time
dt              = 0.01;     % Simulation timestep
dt_control      = dt*10;    % Controller timestep
dt              = dt;    % Same frequency for speed ans test
data_dir        = '../Data2';

USE_SNN       = 0; 
PD_CONTROLLER = 1;
GUARDA_VEL    = 0; % Flag para guardar velocidad angular y actitud (nuevos datos para entrenamiento separado)

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
fprintf('Saving data to: %s\n', fullfile(pwd, data_dir));

% --- Fixed parameters (do not vary between episodes) ---
FOV = deg2rad(60);
W = 1024/4; H = 1024/4;
f = W / (2*tan(FOV/2));
cx = W/2; cy = H/2;
sigma = 1;
k_ev1 = 50; k_ev2 = 1; Th_ON = 5; Th_OFF = -5;

% Fixed inertia tensor (tuned PID works for this configuration)
I = diag([0.040, 0.060, 0.080])/10;

% Kalman Filter Initialization
sigma_q = 1e-12;    % attitude noise
sigma_w = 1e-6;     % velocity noise
P0 = diag([sigma_q, sigma_q, sigma_q, sigma_q, sigma_w, sigma_w, sigma_w]); % Covariance matrix initialization

%w_0 = [0.0 0.0 0.0]; %(2*rand(3,1) - 1) * w_max;

% Fixed PID gains (tuned for this inertia)
wn = 0.5; zeta = sqrt(2)/2;
Kp_x = I(1,1)*(wn^2); Kd_x = I(1,1)*2*zeta*wn; Ki_x = 0.0;
Kp_y = I(2,2)*(wn^2); Kd_y = I(2,2)*2*zeta*wn; Ki_y = 0.0;
Kp_z = I(3,3)*(wn^2); Kd_z = I(3,3)*2*zeta*wn; Ki_z = 0.0;

% p_val = [0 0  2e-4 0 0 -5e-4 0 0  7e-4 0 0;
%          0 0 -6e-4 0 0  7e-4 0 0  3e-4 0 0;
%          0 0 -3e-4 0 0  6e-4 0 0  6e-4 0 0]*4;
% p_time = [0	8.9	9 9.1 46.9 47 47.1 65.9	66 66.1	1000];

% Simulation options
sim_options.SolverType = 'Fixed-step';
sim_options.Solver     = 'ode4';
sim_options.FixedStep  = string(dt);
sim_options.StartTime  = '0';
sim_options.StopTime   = num2str(sim_time_);

failed_episodes = [];

fprintf('Dataset generation: %d episodes \n', num_episodes);
fprintf('Fixed inertia: [%.3f, %.3f, %.3f] kg·m²\n\n', I(1,1), I(2,2), I(3,3));

for ep = initial_episode : (initial_episode + num_episodes - 1)
    fprintf('[%03d/%03d] ', ep, initial_episode + num_episodes - 1);

    % Initial camera attitude: sampled independently for each episode.
    Euler = [deg2rad(-45+(90)*rand), deg2rad(-45+(90)*rand), deg2rad(-45+(90)*rand)];
    roll = Euler(1); pitch = Euler(2); yaw = Euler(3);
    R_c = eul2rotm([yaw, pitch, roll]);
    q_0 = eul2quat([yaw, pitch, roll]);
    A_0 = R_c;

    % Star field
    [stars, s_intensity] = generateStars(200, 1000, 5000);
    % stars_ = load("Starfield_Test.mat");
    % s_intensity_ = load("S_intensity_Test.mat");
    % stars = stars_.stars;
    % s_intensity = s_intensity_.s_intensity;
    
    % Initial angular velocity
    w_0 = zeros(3,1); % -0.05 + (0.1).*rand(3, 1);

    % Perturbations
    p1 = randi([-8, 8], 3, 1);
    p2 = randi([-8, 8], 3, 1);
    p3 = randi([-8, 8], 3, 1);
    p_val = [zeros(3,2), p1, zeros(3,2), p2, zeros(3,2), p3, zeros(3,2)] * 1e-3;
    
    pt1 = randi([5, int32(sim_time/3-5)]);
    pt2 = randi([int32(sim_time/3+5), int32(sim_time/3+sim_time/3-5)]);
    pt3 = randi([int32(2*sim_time/3+5), int32(sim_time-20)]);
    p_time = [0 pt1-0.1 pt1 pt1+0.1 pt2-0.1 pt2 pt2+0.1 pt3-0.1 pt3 pt3+0.1 1000];

    % --- Simulate ---
    try
        result = sim('Model.slx', sim_options);
        sim_elapsed = toc;
    catch ME
        fprintf('FAILED — %s\n', ME.message);
        failed_episodes(end+1) = ep;
        continue;
    end

    % --- Validate: reject diverged simulations ---
    omega_sim  = result.w;
    torque_pid = result.T_control_D;
    omega_final = omega_sim(end, :);

    if any(isnan(omega_final)) || any(abs(omega_final) > 5.0)
        fprintf('DIVERGED — skipping\n');
        failed_episodes(end+1) = ep;
        continue;
    end

    % --- Pack tensors: [T, 2, H, W] ---
    eventos_4d    = cat(4, result.Events_ON, result.Events_OFF);
    events_matrix = permute(eventos_4d, [4, 1, 2, 3]);

    % Convergence time (useful for weighted sampling in training)
    omega_norm = vecnorm(omega_sim, 2, 2);
    conv_idx   = find(omega_norm < 0.001, 1, 'first');
    convergence_time = conv_idx * dt;
    if isempty(conv_idx); convergence_time = sim_time; end

    % --- Save ---
    meta = struct( ...
        'episode',          ep, ...
        'w0',               w_0', ...
        'convergence_time', convergence_time, ...
        'sim_duration',     sim_elapsed ...
    );

    save(sprintf('%s/ep_%04d_events.mat', data_dir, ep), ...
         'events_matrix', '-v7.3');
    
    if GUARDA_VEL
        save(sprintf('%s/ep_%04d_states.mat', data_dir, ep), ...
             'torque_pid', 'omega_sim', 'meta');
    else
        save(sprintf('%s/ep_%04d_states.mat', data_dir, ep), ...
             'torque_pid', 'meta');
    end

    fprintf('OK | %.1fs | w0=[%+.3f %+.3f %+.3f] | conv: %.1fs\n', ...
        sim_elapsed, w_0(1), w_0(2), w_0(3), convergence_time);

    clear eventos_4d result events_matrix omega_sim torque_pid meta omega_norm;
    Simulink.sdi.clear;
end

% --- Summary ---
n_saved = num_episodes - length(failed_episodes);
fprintf('\n════════════════════════════════════════════\n');
fprintf('Saved: %d/%d episodes\n', n_saved, num_episodes);
if ~isempty(failed_episodes)
    fprintf('Failed: %s\n', num2str(failed_episodes));
    fprintf('Rerun with: EPISODES_TO_RERUN = [%s]\n', num2str(failed_episodes));
end
fprintf('════════════════════════════════════════════\n');
