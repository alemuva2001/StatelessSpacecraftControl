% Script para visualizar la telemetría de la Red Neuronal Pulsante (SNN)
% Verifica que el objeto 'result' exista en el Workspace

% 1. Extraer el tiempo y los datos
% Simulink guarda los datos típicamente como objetos 'timeseries' dentro del Dataset
if isa(result.SNN_activity, 'timeseries')
    tiempo = result.SNN_activity.Time;
    datos = result.SNN_activity.Data;
elseif isstruct(result.SNN_activity) && isfield(result.SNN_activity, 'signals')
    % Compatibilidad con formato antiguo de guardado de Simulink
    tiempo = result.SNN_activity.time;
    datos = result.SNN_activity.signals.values;
else
    % Si se guardó como un array directo
    tiempo = result.tout; 
    datos = result.SNN_activity;
end

% 2. Asegurar la orientación correcta de la matriz (N pasos de tiempo x 4 señales)
% Si Simulink expulsó los datos transpuestos [4 x N], los enderezamos
if size(datos, 2) > size(datos, 1) && size(datos, 1) == 4
    datos = datos';
end

% 3. Separar las metricas para mayor claridad
fr_layer1 = datos(:, 1);
fr_layer2 = datos(:, 2);
fr_layer3 = datos(:, 3);
gru_act   = datos(:, 4);

% 4. Generar la visualización
fig = figure('Name', 'Telemetría del Controlador SNN', 'NumberTitle', 'off', ...
             'Color', 'w', 'Position', [100, 100, 1000, 800]);

% --- Gráfica Superior: Tasas de disparo de las capas SNN ---
subplot(3, 1, 1);
plot(tiempo, fr_layer1, 'b', 'LineWidth', 1.5); hold on;
plot(tiempo, fr_layer2, 'r', 'LineWidth', 1.5);
title('Actividad Estocastica de las Capas Pulsantes (Firing Rate)');
ylabel('Proporcion de Disparo');
legend('Capa Convolucional 1 (Spikes)', 'Capa Convolucional 2 (Spikes)', 'Location', 'best');
grid on;
set(gca, 'FontSize', 11);

% --- Grafica Central: Tasa de disparo de la tercera capa SNN ---
subplot(3, 1, 2);
plot(tiempo, fr_layer3, 'm', 'LineWidth', 1.5);
title('Actividad de la Capa Pulsante 3');
ylabel('Proporcion de Disparo');
grid on;
set(gca, 'FontSize', 11);

% --- Grafica Inferior: Actividad de la Red Recurrente (GRU) ---
subplot(3, 1, 3);
plot(tiempo, gru_act, 'k', 'LineWidth', 1.5);
title('Integracion Temporal (Actividad GRU)');
xlabel('Tiempo de Simulacion (s)', 'FontWeight', 'bold');
ylabel('Activacion Absoluta Media');
grid on;
set(gca, 'FontSize', 11);

% Sincronizar el zoom en el eje X para todas las gráficas
linkaxes(findall(fig, 'type', 'axes'), 'x');
