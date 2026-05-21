%% Animación 2D de Eventos en TIEMPO REAL (dt = 0.1)
% Asume que idx_ON, y_ON, x_ON, etc. ya están calculados

% 1. Configuración de Tiempo Físico
dt = 0.1; % Paso de simulación en segundos
T_max_idx = max([max(t_ON), max(t_OFF)]); % Número total de fotogramas

% 2. Configurar figura 2D (Fondo negro)
fig = figure('Name', 'Animación DVS - Tiempo Real', 'Color', 'k', 'Position', [100, 100, 600, 600]);
ax = axes('Parent', fig, 'Color', 'k');
hold(ax, 'on');

% Limitar ejes a la resolución de la cámara (W y H deben estar definidos)
axis(ax, [1, W, 1, H]);
set(ax, 'YDir', 'reverse'); 
daspect(ax, [1 1 1]);       
set(ax, 'XColor', 'w', 'YColor', 'w'); 

% 3. Crear objetos de ploteo vacíos
plot_ON  = plot(ax, NaN, NaN, 'g.', 'MarkerSize', 8);
plot_OFF = plot(ax, NaN, NaN, 'r.', 'MarkerSize', 8);

disp('Reproduciendo animación 2D a 10 Hz (Tiempo Real)...');

%%4. Bucle Sincronizado
for i = 1 : T_max_idx
    % Iniciamos el cronómetro de renderizado
    tic; 
    
    % Filtrar los eventos exactos de este instante 'i'
    % (Como t_ON son índices enteros de ind2sub, usamos '==')
    mask_ON  = (t_ON == i);
    mask_OFF = (t_OFF == i);
    
    % Actualizar posiciones
    set(plot_ON, 'XData', x_ON(mask_ON), 'YData', y_ON(mask_ON));
    set(plot_OFF, 'XData', x_OFF(mask_OFF), 'YData', y_OFF(mask_OFF));
    
    % Calcular y mostrar el tiempo físico real
    tiempo_fisico = i * dt;
    title(ax, sprintf('Cámara DVS | Tiempo Físico: %.1f s', tiempo_fisico), 'Color', 'w', 'FontSize', 14);
    
    % Refrescar pantalla
    drawnow;
    
    % 5. Sincronización estricta del tiempo
    tiempo_dibujado = toc; % Cuánto ha tardado MATLAB en ejecutar este frame
    tiempo_espera = dt - tiempo_dibujado;
    
    if tiempo_espera > 0
        % Si sobró tiempo, pausamos hasta cumplir el dt exacto
        pause(tiempo_espera);
    end
    % Si tiempo_espera es negativo, significa que el Mac tardó más de 0.1s. 
    % En ese caso, el bucle continúa inmediatamente para intentar recuperar el ritmo.
end

disp('Animación finalizada.');