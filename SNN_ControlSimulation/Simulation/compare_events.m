function compare_events()
%% compare_events — Comparador de matrices de eventos
% Muestra lado a lado: tu algoritmo (izq) vs V2E (der).
% Ejecutar con 'result' en el workspace.

result = evalin('base', 'result');

ev_own = cat(4,result.Events_ON,result.Events_OFF);

% Extraer datos
if isa(result, 'Simulink.SimulationOutput')
    %ev_own = result.get('Events_Matrix');
    ev_v2e = result.get('Events_Matrix_Val');
else
    %ev_own = result.Events_Matrix;
    ev_v2e = result.Events_Matrix_Val;
end

fprintf('=== DIAGNOSTICO DE EVENTOS ===\n');
if isempty(ev_own)
    fprintf('PROPIO (Events_Matrix) esta vacio!\n');
else
    if isa(ev_own, 'timeseries')
        fprintf('PROPIO (Events_Matrix) es timeseries, Data size: %s, max: %f, sum: %f\n', ...
            num2str(size(ev_own.Data)), max(ev_own.Data(:)), sum(ev_own.Data(:)));
    elseif isa(ev_own, 'Simulink.SimulationData.Signal')
        fprintf('PROPIO (Events_Matrix) es Signal, Data size: %s, max: %f, sum: %f\n', ...
            num2str(size(ev_own.Values.Data)), max(ev_own.Values.Data(:)), sum(ev_own.Values.Data(:)));
    else
        fprintf('PROPIO (Events_Matrix) es %s, size: %s, max: %f, sum: %f\n', ...
            class(ev_own), num2str(size(ev_own)), max(ev_own(:)), sum(ev_own(:)));
    end
end
if isempty(ev_v2e)
    fprintf('V2E (Events_Matrix_Val) esta vacio!\n');
else
    if isa(ev_v2e, 'timeseries')
        fprintf('V2E (Events_Matrix_Val) es timeseries, Data size: %s, max: %f, sum: %f\n', ...
            num2str(size(ev_v2e.Data)), max(ev_v2e.Data(:)), sum(ev_v2e.Data(:)));
    elseif isa(ev_v2e, 'Simulink.SimulationData.Signal')
        fprintf('V2E (Events_Matrix_Val) es Signal, Data size: %s, max: %f, sum: %f\n', ...
            num2str(size(ev_v2e.Values.Data)), max(ev_v2e.Values.Data(:)), sum(ev_v2e.Values.Data(:)));
    else
        fprintf('V2E (Events_Matrix_Val) es %s, size: %s, max: %f, sum: %f\n', ...
            class(ev_v2e), num2str(size(ev_v2e)), max(ev_v2e(:)), sum(ev_v2e(:)));
    end
end
fprintf('==============================\n');


[data_own, t] = extract_data(ev_own);
[data_v2e, ~] = extract_data(ev_v2e);
data_own = fix_dims(data_own);
data_v2e = fix_dims(data_v2e);

[H, W, ~, N_own] = size(data_own);
N_v2e = size(data_v2e, 4);
N = min(N_own, N_v2e);
if isempty(t)
    try
        % Intentar usar el vector de tiempo de la simulación
        if isa(result, 'Simulink.SimulationOutput')
            t_sim = result.tout;
        else
            t_sim = result.tout;
        end
        if length(t_sim) >= N
            t = t_sim(1:N);
        else
            t = [t_sim; zeros(N - length(t_sim), 1)];
        end
    catch
        % Si no está disponible, intentar obtener 'dt' del workspace
        try
            dt_val = evalin('base', 'dt');
        catch
            dt_val = 0.1; % fallback por defecto
        end
        t = (0:N-1)' * dt_val;
    end
end

data_own = logical(data_own(:,:,:,1:N));
data_v2e_bin = data_v2e(:,:,:,1:N) > 0;

fprintf('=========================\n');
fprintf('Resolucion: %dx%d | Frames: %d\n', H, W, N);
fprintf('V2E total ON: %d | V2E total OFF: %d\n', ...
    sum(data_v2e_bin(:,:,1,:), 'all'), sum(data_v2e_bin(:,:,2,:), 'all'));

%% Metricas por frame
stats = struct('acc', zeros(N,1), 'f1_on', zeros(N,1), 'f1_off', zeros(N,1));
for k = 1:N
    own = [data_own(:,:,1,k), data_own(:,:,2,k)];
    v2e = [data_v2e_bin(:,:,1,k), data_v2e_bin(:,:,2,k)];
    stats.acc(k) = sum(own(:) == v2e(:)) / numel(own);

    % F1 ON
    tp = sum(data_own(:,:,1,k) & data_v2e_bin(:,:,1,k), 'all');
    fp = sum(data_own(:,:,1,k) & ~data_v2e_bin(:,:,1,k), 'all');
    fn = sum(~data_own(:,:,1,k) & data_v2e_bin(:,:,1,k), 'all');
    stats.f1_on(k) = 2*tp / max(2*tp+fp+fn, 1);

    % F1 OFF
    tp = sum(data_own(:,:,2,k) & data_v2e_bin(:,:,2,k), 'all');
    fp = sum(data_own(:,:,2,k) & ~data_v2e_bin(:,:,2,k), 'all');
    fn = sum(~data_own(:,:,2,k) & data_v2e_bin(:,:,2,k), 'all');
    stats.f1_off(k) = 2*tp / max(2*tp+fp+fn, 1);
end

%% Figura 1 — Metricas temporales
figure('Name', 'Metrics', 'NumberTitle', 'off', ...
       'Position', [100 500 900 350], 'Color', 'w');

subplot(1,2,1);
plot(t(1:N), sum(reshape(data_own(:,:,1,:), [], N))', 'g-', 'LineWidth', 1); hold on;
plot(t(1:N), sum(reshape(data_v2e_bin(:,:,1,:), [], N))', 'g--', 'LineWidth', 1);
plot(t(1:N), sum(reshape(data_own(:,:,2,:), [], N))', 'r-', 'LineWidth', 1);
plot(t(1:N), sum(reshape(data_v2e_bin(:,:,2,:), [], N))', 'r--', 'LineWidth', 1);
legend('ON propio', 'ON V2E', 'OFF propio', 'OFF V2E', 'Location', 'best');
xlabel('Tiempo (s)'); ylabel('Pixeles activos'); title('Eventos por frame'); grid on;

subplot(1,2,2);
plot(t(1:N), stats.f1_on, 'g-', 'LineWidth', 1); hold on;
plot(t(1:N), stats.f1_off, 'r-', 'LineWidth', 1);
plot(t(1:N), stats.acc, 'b-', 'LineWidth', 1);
legend('F1 ON', 'F1 OFF', 'Accuracy', 'Location', 'best');
xlabel('Tiempo (s)'); ylabel('Score'); title('Metricas'); ylim([0 1.05]); grid on;

%% Figura 2 — Comparación Lado a Lado (Video Simultáneo)
fig2 = figure('Name', 'DVS Event Playback', 'NumberTitle', 'off', ...
       'Position', [100 50 1000 550], 'Color', 'w');

% Crear subplots
ax_own = subplot(1, 2, 1);
h_own = imshow(zeros(H, W, 3), 'Parent', ax_own); 
title_own = title(ax_own, 'Model  t=0.00s', 'FontSize', 12, 'Interpreter', 'none');
xlabel(ax_own, 'Green: ON | Red: OFF', 'FontSize', 10);

ax_v2e = subplot(1, 2, 2);
h_v2e = imshow(zeros(H, W, 3), 'Parent', ax_v2e); 
title_v2e = title(ax_v2e, 'V2E  t=0.00s', 'FontSize', 12, 'Interpreter', 'none');
xlabel(ax_v2e, 'Green: ON | Red: OFF', 'FontSize', 10);

sgtitle('DVS Model - V2E events comparison', 'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');

% Determinar retardo aproximado de reproducción
if N > 1
    mean_dt = mean(diff(t));
    if isempty(mean_dt) || isnan(mean_dt) || mean_dt <= 0
        mean_dt = 0.033; % default 30 fps if no time vector
    end
else
    mean_dt = 0.033;
end

% Ajustar velocidad de reproducción
pause_time = max(0.001, mean_dt);

fprintf('Reproduciendo animación de eventos... Cierra la ventana para detener.\n');

while ishandle(fig2)
    t_start = tic; % Iniciar cronómetro para reproducción en tiempo real
    for k = 1:N
        if ~ishandle(fig2), break; end
        
        % Imagen propia (ON = Verde, OFF = Rojo)
        img_own = zeros(H, W, 3);
        img_own(:,:,2) = double(data_own(:,:,1,k));   % Verde = ON
        img_own(:,:,1) = double(data_own(:,:,2,k));   % Rojo  = OFF
        
        % Imagen V2E
        img_v2e = zeros(H, W, 3);
        img_v2e(:,:,2) = double(data_v2e_bin(:,:,1,k)); % Verde = ON
        img_v2e(:,:,1) = double(data_v2e_bin(:,:,2,k)); % Rojo  = OFF
        
        % Actualizar CData
        set(h_own, 'CData', img_own);
        set(h_v2e, 'CData', img_v2e);
        
        % Actualizar títulos
        set(title_own, 'String', sprintf('PROPIO  t=%.3f s', t(k)));
        set(title_v2e, 'String', sprintf('V2E  t=%.3f s', t(k)));
        
        drawnow limitrate;
        
        % Sincronización en tiempo real (1s simulado = 1s real)
        if k < N
            % Tiempo simulado que debe haber transcurrido desde el inicio
            target_sim_elapsed = t(k+1) - t(1);
            % Tiempo real transcurrido desde que empezó la animación
            real_elapsed = toc(t_start);
            
            % Cuánto debemos esperar para sincronizarnos
            wait_time = target_sim_elapsed - real_elapsed;
            
            if wait_time > 0
                pause(wait_time);
            end
        end
    end
    
    % Pequeña pausa antes de volver a empezar
    if ishandle(fig2)
        pause(1.0);
    end
end

%% Estadisticas
fprintf('\n--- ESTADISTICAS ---\n');
fprintf('Accuracy media: %.1f%%\n', mean(stats.acc)*100);
fprintf('F1 ON medio:    %.4f\n', mean(stats.f1_on));
fprintf('F1 OFF medio:   %.4f\n', mean(stats.f1_off));

end

%% Funciones auxiliares
function [data, t] = extract_data(ev)
    if isa(ev, 'timeseries')
        data = ev.Data; t = ev.Time;
    elseif isa(ev, 'Simulink.SimulationData.Signal')
        data = ev.Values.Data; t = ev.Values.Time;
    else
        data = squeeze(ev); t = [];
    end
end

function data = fix_dims(data)
    if ndims(data) == 4 && size(data, 4) == 2
        data = permute(data, [1 2 4 3]);
    end
end
