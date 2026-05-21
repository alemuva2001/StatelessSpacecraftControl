% =========================================================================
% Script de Análisis Visual: Torque Original vs. Torque Normalizado
% =========================================================================
%clear; clc; close all;

% 1. Configuración de Rutas (Ajusta la carpeta si es necesario)
data_dir = '../Data'; 
filename = 'ep_0001_states.mat'; % Cambia el número para ver otros episodios
filepath = fullfile(data_dir, filename);

% 2. Cargar los Datos
if ~isfile(filepath)
    error('No se encontró el archivo. Comprueba la ruta y el nombre.');
end
data = load(filepath);
torque_raw = data.torque_pid; % Matriz de tamaño [T, 3]

% 3. Calcular Estadísticas (Simulando compute_dataset_stats.py)
% Nota: En Python usas la estadística de TODO el dataset. 
% Aquí usamos la del episodio actual para ver el contraste directo.
mu = mean(torque_raw, 1);
sigma = std(torque_raw, 0, 1);

% Evitar divisiones por cero si un eje no se mueve en absoluto
sigma(sigma == 0) = 1e-8;

% 4. Normalizar los Datos (Z-score)
torque_norm = (torque_raw - mu) ./ sigma;

% 5. Configurar el vector de tiempo (Asumiendo dt = 0.1s)
dt = 0.1;
T_total = size(torque_raw, 1);
t = (0:T_total-1) * dt;

% =========================================================================
% 6. Generar la Visualización
% =========================================================================
fig = figure('Name', 'Análisis de Magnitud SNN: Reposo vs Impactos', ...
             'Position', [100, 100, 1400, 800], 'Color', 'w');

ejes = {'Eje X (Roll)', 'Eje Y (Pitch)', 'Eje Z (Yaw)'};
ax_raw = zeros(1,3);
ax_norm = zeros(1,3);

for i = 1:3
    % --- Columna Izquierda: Torque Físico Original (PID) ---
    ax_raw(i) = subplot(3, 2, i*2 - 1);
    plot(t, torque_raw(:, i), 'b', 'LineWidth', 1.2);
    title(['Original (PID) - ' ejes{i}], 'FontWeight', 'bold');
    ylabel('Torque (Nm)');
    grid on;
    % Añadir línea base de 0
    yline(0, 'k--', 'Alpha', 0.3);
    
    % --- Columna Derecha: Torque Normalizado (Lo que ve la SNN) ---
    ax_norm(i) = subplot(3, 2, i*2);
    plot(t, torque_norm(:, i), 'r', 'LineWidth', 1.2);
    title(['Normalizado (Target SNN) - ' ejes{i}], 'FontWeight', 'bold');
    ylabel('Magnitud (Z-score)');
    grid on;
    % Añadir línea base de 0
    yline(0, 'k--', 'Alpha', 0.3);
end

xlabel(ax_raw(3), 'Tiempo de Simulación (s)', 'FontWeight', 'bold');
xlabel(ax_norm(3), 'Tiempo de Simulación (s)', 'FontWeight', 'bold');

% Sincronizar el Zoom en el eje X para todas las gráficas
linkaxes([ax_raw, ax_norm], 'x');

disp('Gráficas generadas. Haz zoom en los picos para comparar magnitudes.');
disp('Estadísticas calculadas para este episodio:');
fprintf('  Sigma X: %e\n', sigma(1));
fprintf('  Sigma Y: %e\n', sigma(2));
fprintf('  Sigma Z: %e\n', sigma(3));