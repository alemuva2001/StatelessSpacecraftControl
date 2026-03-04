function animate_cubo(tiempo, roll_grados, pitch_grados, yaw_grados, velocidad)
% ANIMATE_CUBO - Anima un cubo 3D según los datos de actitud proporcionados.
%
% Sintaxis:
%   animate_cubo(tiempo, roll, pitch, yaw, velocidad)
%
% Descripción:
%   Crea una figura y anima la orientación de un cubo 3D y su sistema de
%   referencia asociado. La orientación se define por los ángulos de Euler
%   (roll, pitch, yaw) en cada instante de tiempo.
%
% Argumentos de Entrada:
%   tiempo       - Vector (Nx1) con las marcas de tiempo de la simulación.
%   roll_grados  - Vector (Nx1) de ángulos de alabeo (Roll) en radianes.
%   pitch_grados - Vector (Nx1) de ángulos de cabeceo (Pitch) en radianes.
%   yaw_grados   - Vector (Nx1) de ángulos de guiñada (Yaw) en radianes.
%   velocidad    - Multiplicador de la velocidad de reproducción.
%                  1.0 = Tiempo real.
%                  2.0 = Doble de velocidad.
%                  0.5 = Mitad de velocidad.
%
% Ejemplo de Uso (copia esto en la ventana de comandos):
%   t = linspace(0, 10, 200); % 10 segundos, 200 puntos
%   roll = 30 * sin(t);
%   pitch = 60 * cos(t/2);
%   yaw = linspace(0, 360, 200);
%   animate_cubo(t, roll, pitch, yaw, 2); % Reproducir al doble de velocidad

% =========================================================================
% 1. VALIDACIÓN DE ENTRADAS
% =========================================================================
if nargin ~= 5
    error('Se requieren 5 argumentos: tiempo, roll, pitch, yaw, velocidad.');
end

num_puntos = length(tiempo);
if num_puntos < 2
    warning('Se necesitan al menos 2 puntos de datos para la animación. No se hará nada.');
    return;
end

if ~(length(roll_grados) == num_puntos && length(pitch_grados) == num_puntos && length(yaw_grados) == num_puntos)
    error('Los vectores de tiempo, roll, pitch y yaw deben tener la misma longitud.');
end

if ~isnumeric(velocidad) || velocidad <= 0
    error('La velocidad de simulación debe ser un número positivo.');
end

% Cierra figuras previas con el mismo nombre para evitar confusiones
fig_existente = findobj('type', 'figure', 'Name', 'Visualizador de Actitud 3D');
if ~isempty(fig_existente)
    close(fig_existente);
end

roll_grados = rad2deg(roll_grados);
pitch_grados = rad2deg(pitch_grados);
yaw_grados = rad2deg(yaw_grados);

% =========================================================================
% 2. DEFINICIÓN DE LA GEOMETRÍA (Cubo y Ejes de Referencia)
% =========================================================================
lado = 1; s = lado / 2;
vertices_cubo_orig = [-s -s -s; s -s -s; s s -s; -s s -s; -s -s s; s -s s; s s s; -s s s];
caras_cubo = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];

long_eje = 1.0;
eje_x_orig = [0 0 0; long_eje 0 0];
eje_y_orig = [0 0 0; 0 long_eje 0];
eje_z_orig = [0 0 0; 0 0 long_eje];

% =========================================================================
% 3. CONFIGURACIÓN DE LA FIGURA
% =========================================================================
fig = figure('Name', 'Visualizador de Actitud 3D', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
ax = axes('Parent', fig);
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');

lim = lado * 1.5;
axis(ax, [-lim lim -lim lim -lim lim]);
view(ax, 30, 20);

xlabel(ax, 'Eje X (Inercial)'); ylabel(ax, 'Eje Y (Inercial)'); zlabel(ax, 'Eje Z (Inercial)');

%DIBUJO DE LOS EJES FIJOS Y TRANSPARENTES %%%
long_eje_fijo = lim * 0.8; % Longitud para los ejes fijos
alpha_fijo = 0.3; % Nivel de transparencia (0=invisible, 1=opaco)

% Dibuja los ejes fijos con un color, estilo de línea y transparencia específicos
plot3(ax, [0 long_eje_fijo], [0 0], [0 0], '--', 'Color', [1 0 0 alpha_fijo], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot3(ax, [0 0], [0 long_eje_fijo], [0 0], '--', 'Color', [0 1 0 alpha_fijo], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot3(ax, [0 0], [0 0], [0 long_eje_fijo], '--', 'Color', [0 0 1 alpha_fijo], 'LineWidth', 1.5, 'HandleVisibility', 'off');
%HandleVisibility='off' evita que estos ejes aparezcan en la leyenda

% Dibuja el estado inicial y guarda los "handles" para actualizarlos
cubo_patch = patch(ax, 'Vertices', vertices_cubo_orig, 'Faces', caras_cubo, ...
                 'FaceColor', [0.3 0.6 1.0], 'FaceAlpha', 0.6, 'EdgeColor', 'k');
h_eje_x = plot3(ax, eje_x_orig(:,1), eje_x_orig(:,2), eje_x_orig(:,3), 'r-', 'LineWidth', 3);
h_eje_y = plot3(ax, eje_y_orig(:,1), eje_y_orig(:,2), eje_y_orig(:,3), 'g-', 'LineWidth', 3);
h_eje_z = plot3(ax, eje_z_orig(:,1), eje_z_orig(:,2), eje_z_orig(:,3), 'b-', 'LineWidth', 3);
h_label_x = text(ax, 0,0,0, ' X_{body}', 'FontSize', 12, 'Color', 'r');
h_label_y = text(ax, 0,0,0, ' Y_{body}', 'FontSize', 12, 'Color', 'g');
h_label_z = text(ax, 0,0,0, ' Z_{body}', 'FontSize', 12, 'Color', 'b');
legend(ax, [h_eje_x, h_eje_y, h_eje_z], {'Eje X (Roll)', 'Eje Y (Pitch)', 'Eje Z (Yaw)'});
title_handle = title(ax, 'Preparando animación...');

% =========================================================================
% 4. BUCLE DE ANIMACIÓN
% =========================================================================
fprintf('Iniciando animación...\n');

tic; % Inicia un cronómetro para la sincronización

for i = 1:num_puntos
    % Comprueba si la figura ha sido cerrada por el usuario
    if ~isvalid(fig)
        fprintf('Animación detenida por el usuario.\n');
        return;
    end
    
    % --- Obtener la actitud y calcular la matriz de rotación ---
    yaw   = deg2rad(yaw_grados(i));
    pitch = deg2rad(pitch_grados(i));
    roll  = deg2rad(roll_grados(i));

    Rz = [cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1];
    Ry = [cos(pitch) 0 sin(pitch); 0 1 0; -sin(pitch) 0 cos(pitch)];
    Rx = [1 0 0; 0 cos(roll) -sin(roll); 0 sin(roll) cos(roll)];
    R = Rz * Ry * Rx; % Matriz de rotación final (convención ZYX)

    % --- Aplicar la rotación ---
    vertices_rotados = (R * vertices_cubo_orig')';
    eje_x_rotado = (R * eje_x_orig')';
    eje_y_rotado = (R * eje_y_orig')';
    eje_z_rotado = (R * eje_z_orig')';

    % --- Actualizar los datos de los objetos gráficos ---
    set(cubo_patch, 'Vertices', vertices_rotados);
    set(h_eje_x, 'XData', eje_x_rotado(:,1), 'YData', eje_x_rotado(:,2), 'ZData', eje_x_rotado(:,3));
    set(h_eje_y, 'XData', eje_y_rotado(:,1), 'YData', eje_y_rotado(:,2), 'ZData', eje_y_rotado(:,3));
    set(h_eje_z, 'XData', eje_z_rotado(:,1), 'YData', eje_z_rotado(:,2), 'ZData', eje_z_rotado(:,3));
    set(h_label_x, 'Position', eje_x_rotado(2,:));
    set(h_label_y, 'Position', eje_y_rotado(2,:));
    set(h_label_z, 'Position', eje_z_rotado(2,:));
    set(title_handle, 'String', sprintf('Tiempo: %.2fs | Roll: %.1f° | Pitch: %.1f° | Yaw: %.1f°', ...
        tiempo(i), yaw_grados(i), pitch_grados(i), roll_grados(i)));
    
    % --- Pausa para controlar la velocidad ---
    if i < num_puntos
        % Calcula la duración real del siguiente paso
        delta_t_real = tiempo(i+1) - tiempo(i);
        % Ajusta la pausa según la velocidad de reproducción
        tiempo_pausa = delta_t_real / velocidad;
        
        % Pausa inteligente: descuenta el tiempo que se tardó en renderizar
        % para una sincronización más precisa.
        pause(max(0, tiempo_pausa - toc));
        tic; % Reinicia el cronómetro para el siguiente fotograma
    end
    
    drawnow; % Fuerza la actualización de la figura
end

fprintf('Animación finalizada.\n');
end