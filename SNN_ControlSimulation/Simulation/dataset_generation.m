% dataset_generation.m — common event dataset with separate P/D labels
% First approach: fixed inertia, variable initial attitude + variable w_0
clc; close all; clearvars -except dataset_options;
if ~exist('dataset_options','var'), dataset_options=struct(); end
set(0, 'DefaultFigureWindowStyle', 'normal');
tic;
% =========================================================================
% CONFIGURATION
% =========================================================================
initial_episode = 1;         % Episodio por el que empezar a numerar
num_episodes    = 150;       % Cuántos episodios generar en esta sesión
sim_time        = 100;      % Perturbations
sim_time_       = 120;      % Simulation time
dt              = 0.01;     % Simulation timestep
dt_control      = dt*10;    % Controller timestep
dt              = dt*10;    % Same frequency for speed and test
tau_f = 0.1;                % Derivative-output filter
alpha_f = exp(-dt_control/tau_f);

simulation_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(simulation_dir);
addpath(simulation_dir);
data_dir = fullfile(project_root,'Data'); % shared by NeuralD and NeuralP

% Optional overrides for reproducible small checks; normal interactive defaults above.
allowed_options={'initial_episode','num_episodes','sim_time_','data_dir','seed','p_time','p_val','fail_fast'};
assert(all(ismember(fieldnames(dataset_options),allowed_options)), 'Unknown dataset_options field.');
if isfield(dataset_options,'initial_episode'), initial_episode=dataset_options.initial_episode; end
if isfield(dataset_options,'num_episodes'), num_episodes=dataset_options.num_episodes; end
if isfield(dataset_options,'sim_time_'), sim_time_=dataset_options.sim_time_; end
if isfield(dataset_options,'data_dir'), data_dir=dataset_options.data_dir; end
if isfield(dataset_options,'seed'), rng(dataset_options.seed,'twister'); end
assert(isfield(dataset_options,'p_time')==isfield(dataset_options,'p_val'), 'Supply p_time and p_val together.');

% Funcionamiento de la simulación
Ideal         = 1;  % Usa omega y q_err ideales [1] o Usa omega y q_err estimados [0]
USE_SNN       = 0;  % Compila la red al iniciar la simulación [1] o no la compila [0]
PD_CONTROLLER = 1;  % Para ajustar la diferencia de timestep entre simulacion y estimacion [1] (Dejar siempre a 1)
NeuralD       = 0;  % Utiliza como componente derivativa la red [1] o el PD convencional [0]    
NeuralP       = 0;  % Utiliza como componente proporcional la red [1] o el PD convencional [0]

label_torque_tolerance = 1e-10; % N*m; valida T_D=-K_D*omega seleccionada por Ideal

% Criterios de evaluacion
recovery_threshold_deg = 5.0;
recovery_hold_s = 2;
instability_error_threshold_deg = 90;
instability_trend_min_error_deg = 60;
instability_trend_threshold_deg_s = 1;
instability_window_s = 5;
instability_trend_window_s = 10;
% Umbral solo diagnostico para comparar velocidad estimada y real.
reference_rate_tolerance = 0.01; % rad/s; no invalida la etiqueta del controlador
review_dir = fullfile(data_dir,'review');
unstable_dir = fullfile(data_dir,'unstable');

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
if ~exist(review_dir,'dir'), mkdir(review_dir); end
if ~exist(unstable_dir,'dir'), mkdir(unstable_dir); end
fprintf('Saving data to: %s\n', data_dir);

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
% El termino proporcional utiliza q_e,v, aproximadamente theta_e/2.
Kp_x = 2*I(1,1)*(wn^2); Kd_x = I(1,1)*2*zeta*wn; Ki_x = 0.0;
Kp_y = 2*I(2,2)*(wn^2); Kd_y = I(2,2)*2*zeta*wn; Ki_y = 0.0;
Kp_z = 2*I(3,3)*(wn^2); Kd_z = I(3,3)*2*zeta*wn; Ki_z = 0.0;

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
review_episodes = [];

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

    if isfield(dataset_options,'p_time')
        p_time=dataset_options.p_time; p_val=dataset_options.p_val;
        assert(size(p_val,1)==3 && size(p_val,2)==numel(p_time) && all(diff(p_time)>0));
        active=find(any(p_val~=0,1));
        assert(~isempty(active) && active(end)<numel(p_time));
        disturbance_end=p_time(active(end)+1);
    else
        disturbance_end=pt3+0.1;
    end

    % --- Simulate ---
    % No sobrescribir episodios ya generados, tampoco los pendientes de revision.
    assert(~isfile(fullfile(data_dir,sprintf('ep_%04d_states.mat',ep))) && ...
        ~isfile(fullfile(review_dir,sprintf('ep_%04d_states.mat',ep))), ...
        'El episodio %d ya existe. Cambia initial_episode o data_dir.',ep);
    for check_dir = {data_dir,review_dir,unstable_dir}
        assert(~isfile(fullfile(check_dir{1},sprintf('ep_%04d_events.mat',ep))) && ...
               ~isfile(fullfile(check_dir{1},sprintf('ep_%04d_states.mat',ep))), ...
               'Episode %d already exists in %s.',ep,check_dir{1});
    end
    episode_timer = tic;
    try
        result = sim(fullfile(simulation_dir,'Model.slx'), sim_options);
        sim_elapsed = toc(episode_timer);
    catch ME
        fprintf('FAILED — %s\n', getReport(ME,'extended','hyperlinks','off'));
        if isfield(dataset_options,'fail_fast') && dataset_options.fail_fast, rethrow(ME); end
        failed_episodes(end+1) = ep;
        continue;
    end

    % --- Registrar y clasificar; no borrar episodios por falta de convergencia ---
    omega_sim  = result.w;
    torque_pid = result.T_control_D; % legacy field: D only
    kp = [Kp_x Kp_y Kp_z];
    if ismember('q_error_control',result.who)
        q_error_control = result.q_error_control;
    elseif Ideal
        q_error_control = result.errorR;
    else
        q_error_control = result.error;
    end
    if ismember('T_control_P',result.who)
        torque_p = result.T_control_P;
        torque_p_source = 'measured conventional P+gyro branch';
    else
        if Ideal, w_teacher=result.w; else, w_teacher=result.w_corr; end
        torque_p = q_error_control(:,2:4).*kp + cross(w_teacher,(I*w_teacher')',2);
        torque_p_source = 'reconstructed P+gyro, verified against total-D';
    end
    torque_total = result.T_control_PID;
    q_sim = result.q;
    time = result.tout(:);
    event_time = time;
    time_control = result.tout_control(:);
    omega_estimated = result.w_corr;

    % --- Pack tensors: [T, 2, H, W] ---
    eventos_4d    = cat(4, result.Events_ON, result.Events_OFF);
    events_matrix = permute(eventos_4d, [4, 1, 2, 3]);

    % Error angular real respecto a la referencia inicial; q y -q equivalentes.
    state_times_valid = size(q_sim,1)==numel(time) && size(omega_sim,1)==numel(time) && ...
        numel(time)>1 && all(isfinite(time)) && all(diff(time)>0);
    numerical_valid = all(isfinite(q_sim),'all') && all(isfinite(omega_sim),'all') && ...
        all(isfinite(torque_p),'all') && all(isfinite(q_error_control),'all') && ...
        all(isfinite(torque_total),'all') && all(isfinite(torque_pid),'all') && all(isfinite(omega_estimated),'all') && ...
        all(isfinite(events_matrix),'all') && all(vecnorm(q_sim,2,2)>1e-12);
    angular_error_deg = NaN(size(time));
    recovered = false; recovery_time_s = NaN; recovery_start_s = NaN;
    residual_error_deg = NaN; peak_error_deg = NaN;
    terminal_error_deg = NaN; terminal_error_slope_deg_s = NaN;
    instability_detected = false; instability_type = 'not_evaluated';
    last_disturbance_end = disturbance_end;
    if state_times_valid && numerical_valid
        q_unit = q_sim ./ vecnorm(q_sim,2,2);
        cosine = abs(q_unit * (q_0(:)/norm(q_0)));
        angular_error_deg = rad2deg(2*acos(min(1,max(0,cosine))));
        peak_error_deg = max(angular_error_deg);
        residual_error_deg = mean(angular_error_deg(time>=time(end)-5));
        terminal_error_deg = angular_error_deg(end);
        final_window = time >= time(end)-instability_window_s;
        if nnz(final_window) >= 2
            persistent_large_error = mean(angular_error_deg(final_window)) >= instability_error_threshold_deg;
        else
            persistent_large_error = false;
        end
        trend_window = time >= time(end)-instability_trend_window_s;
        if nnz(trend_window) >= 2
            trend_coefficients = polyfit(time(trend_window),angular_error_deg(trend_window),1);
            terminal_error_slope_deg_s = trend_coefficients(1);
        end
        worsening_large_error = terminal_error_deg >= instability_trend_min_error_deg && ...
            terminal_error_slope_deg_s >= instability_trend_threshold_deg_s;
        instability_detected = persistent_large_error || worsening_large_error;
        if persistent_large_error
            instability_type = 'persistent_large_attitude_error';
        elseif worsening_large_error
            instability_type = 'increasing_large_attitude_error';
        else
            instability_type = 'not_detected_by_operational_criterion';
        end
        after = find(time>=last_disturbance_end);
        if ~isempty(after)
            last_above = find(angular_error_deg(after)>recovery_threshold_deg,1,'last');
            candidate = 1;
            if ~isempty(last_above), candidate=last_above+1; end
            if candidate<=numel(after)
                idx = after(candidate);
                recovered = time(end)-time(idx)>=recovery_hold_s-1e-8;
                if recovered
                    recovery_start_s = time(idx);
                    recovery_time_s = time(idx)-last_disturbance_end;
                end
            end
        end
    end
    % No alinear por indice ni interpolar silenciosamente frecuencias diferentes.
    synchronized = state_times_valid && size(events_matrix,4)==numel(event_time) && ...
        size(torque_pid,1)==numel(time_control) && ...
        isequal(size(torque_p),[numel(time_control),3]) && ...
        isequal(size(torque_total),[numel(time_control),3]) && ...
        isequal(size(q_error_control),[numel(time_control),4]) && ...
        size(omega_estimated,1)==numel(time_control) && ...
        numel(time_control)==numel(event_time);
    if synchronized
        synchronized = all(isfinite(time_control)) && all(diff(time_control)>0) && ...
            all(abs(time_control-event_time)<1e-8);
    end
    reference_rate_error = NaN(size(time_control));
    rate_error_under_diagnostic_tolerance = false(size(time_control));
    expected_torque_D = NaN(size(torque_pid));
    label_torque_error = NaN(size(time_control));
    label_torque_error_max = NaN;
    expected_torque_P = NaN(size(torque_p));
    label_torque_p_error = NaN(size(time_control));
    label_torque_decomposition_error = NaN(size(time_control));
    torque_gyro = NaN(size(torque_p));
    if synchronized && numerical_valid
        reference_rate_error = vecnorm(omega_estimated-omega_sim,2,2);
        rate_error_under_diagnostic_tolerance = reference_rate_error<=reference_rate_tolerance;
        if Ideal
            expected_torque_D = -omega_sim .* [Kd_x Kd_y Kd_z];
            omega_control = omega_sim;
        else
            expected_torque_D = -omega_estimated .* [Kd_x Kd_y Kd_z];
            omega_control = omega_estimated;
        end
        label_torque_error = vecnorm(torque_pid-expected_torque_D,2,2);
        torque_gyro = cross(omega_control,(I*omega_control')',2);
        pure_torque_P = q_error_control(:,2:4).*kp;
        % Fixed target contract: P plus gyro, never accept pure P silently.
        expected_torque_P = pure_torque_P + torque_gyro;
        label_torque_decomposition_error = vecnorm(torque_total-torque_pid-torque_p,2,2);
        label_torque_p_error = vecnorm(torque_p-expected_torque_P,2,2);
    end
    % T_D debe concordar con la velocidad seleccionada en el controlador:
    % real si Ideal~=0, estimada si Ideal==0. El error del EKF es diagnostico.
    label_consistent = synchronized && numerical_valid && ...
        all(isfinite(label_torque_error)) && max(label_torque_error)<=label_torque_tolerance && ...
        all(isfinite(label_torque_p_error)) && max(label_torque_p_error)<=label_torque_tolerance && ...
        all(isfinite(label_torque_decomposition_error)) && max(label_torque_decomposition_error)<=1e-9;
    reference_valid = label_consistent;
    completed_horizon = ~isempty(time) && isfinite(time(end)) && time(end)>=sim_time_-1e-8;
    recovered = recovered && completed_horizon;
    if ~recovered
        recovery_time_s = NaN; recovery_start_s = NaN;
    end
    % Include every valid teacher-labelled episode except operationally unstable ones.
    % Recovery remains a separate outcome metric; non-recovered but finite episodes
    % can still teach the derivative branch how the PD teacher responds off-nominal.
    training_ready = label_consistent && numerical_valid && synchronized && completed_horizon && ~instability_detected;
    % Alias antiguo: ahora es tiempo desde el final de la ultima perturbacion.
    convergence_time = recovery_time_s;

    % --- Save ---
    meta = struct( ...
        'episode',          ep, ...
        'w0',               w_0', ...
        'convergence_time', convergence_time, ...
        'sim_duration',     sim_elapsed ...
    );
    meta.recovered = recovered;
    meta.label_torque_p_error_max = max(label_torque_p_error);
    meta.label_torque_decomposition_error_max = max(label_torque_decomposition_error);
    meta.torque_p_source = torque_p_source;
    meta.target_kind = 'p_plus_gyro';
    meta.torque_p_definition = '+Kp .* q_error_control(:,2:4) + omega x (I*omega); excludes D';
    meta.quaternion_error_definition = 'inv(q_selected) Hamilton-product q_0; scalar first; no hemisphere flip';
    meta.recovery_time_s = recovery_time_s;
    meta.recovery_start_s = recovery_start_s;
    meta.last_disturbance_end = last_disturbance_end;
    meta.recovery_threshold_deg = recovery_threshold_deg;
    meta.recovery_hold_s = recovery_hold_s;
    meta.peak_error_deg = peak_error_deg;
    meta.residual_error_deg = residual_error_deg;
    meta.terminal_error_deg = terminal_error_deg;
    meta.terminal_error_slope_deg_s = terminal_error_slope_deg_s;
    meta.instability_detected = instability_detected;
    meta.instability_type = instability_type;
    meta.instability_error_threshold_deg = instability_error_threshold_deg;
    meta.instability_trend_min_error_deg = instability_trend_min_error_deg;
    meta.instability_trend_threshold_deg_s = instability_trend_threshold_deg_s;
    meta.instability_window_s = instability_window_s;
    meta.instability_trend_window_s = instability_trend_window_s;
    meta.numerical_valid = numerical_valid;
    meta.synchronized = synchronized;
    meta.reference_valid = reference_valid;
    meta.label_consistent = label_consistent;
    meta.label_torque_tolerance = label_torque_tolerance;
    meta.label_torque_decomposition_tolerance = 1e-9;
    if all(isfinite(label_torque_error)), label_torque_error_max=max(label_torque_error); end
    meta.label_torque_error_max = label_torque_error_max;
    meta.completed_horizon = completed_horizon;
    meta.estimator_rate_error_rms = sqrt(mean(reference_rate_error.^2));
    sorted_rate_error = sort(reference_rate_error);
    meta.estimator_rate_error_p95 = sorted_rate_error(max(1,ceil(0.95*numel(sorted_rate_error))));
    meta.estimator_rate_error_max = max(reference_rate_error);
    meta.estimator_rate_diagnostic_tolerance = reference_rate_tolerance;
    meta.estimator_rate_fraction_below_tolerance = mean(rate_error_under_diagnostic_tolerance);
    meta.training_ready = training_ready;
    meta.parameters = struct('dt',dt,'dt_control',dt_control,'q_0',q_0,'I',I, ...
        'Kp',[Kp_x Kp_y Kp_z],'Kd',[Kd_x Kd_y Kd_z], ...
        'stars',stars,'s_intensity',s_intensity,'p_time',p_time,'p_val',p_val, ...
        'USE_SNN',USE_SNN,'NeuralD',NeuralD,'NeuralP',NeuralP,'PD_CONTROLLER',PD_CONTROLLER, ...
        'Ideal',Ideal);
    % El cargador toma los pares del directorio raiz Data; aislar fallos claros.
    episode_dir = data_dir;
    if ~training_ready
        if instability_detected
            episode_dir = unstable_dir;
        else
            episode_dir = review_dir;
        end
        review_episodes(end+1) = ep;
    end

    save(sprintf('%s/ep_%04d_events.mat', episode_dir, ep), ...
         'events_matrix','event_time', '-v7.3');
    
    % Necesarios para auditar las etiquetas, independientemente de GUARDA_VEL.
    save(sprintf('%s/ep_%04d_states.mat', episode_dir, ep), ...
        'torque_p','kp','q_error_control','expected_torque_P','label_torque_p_error', ...
        'torque_total','torque_gyro','label_torque_decomposition_error','torque_pid','omega_sim','omega_estimated','q_sim','time','time_control', ...
        'expected_torque_D','label_torque_error','angular_error_deg', ...
        'reference_rate_error','rate_error_under_diagnostic_tolerance','meta');

    fprintf('GUARDADO | %.1fs | recuperado=%d | residual=%.3f deg | recuperacion=%.1fs | T_D consistente=%d | listo=%d\n', ...
        sim_elapsed,recovered,residual_error_deg,recovery_time_s,label_consistent,training_ready);

    clear eventos_4d result events_matrix omega_sim torque_pid meta omega_norm;
    Simulink.sdi.clear;
end

% --- Summary ---
n_saved = num_episodes - length(failed_episodes);
fprintf('\n════════════════════════════════════════════\n');
fprintf('Saved: %d/%d episodes\n', n_saved, num_episodes);
fprintf('Entrenables: %d | Revision/inestables: %d (%s, %s)\n', ...
    n_saved-numel(review_episodes),numel(review_episodes),review_dir,unstable_dir);
if ~isempty(failed_episodes)
    fprintf('Failed: %s\n', num2str(failed_episodes));
    fprintf('Rerun with: EPISODES_TO_RERUN = [%s]\n', num2str(failed_episodes));
end
fprintf('════════════════════════════════════════════\n');
