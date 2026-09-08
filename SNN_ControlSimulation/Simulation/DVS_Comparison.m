% =========================================================================
% SCRIPT COMPLETO DE VISUALIZACIÓN Y EXPORTACIÓN A .MP4 (DVS vs IMAGEN REAL)
% =========================================================================

video_filename = 'Simulacion_DVS.mp4'; % Nombre del archivo de salida
fps = 30;                              % Fotogramas por segundo del vídeo final

% 1. Extracción de tiempos y datos desde la estructura 'result'
t_events = result.tout;
ev_on_data = result.Events_ON;     % Matriz de eventos ON
ev_off_data = result.Events_OFF;   % Matriz de eventos OFF
img_data = result.RealImage;       % Matriz de la imagen real B/W

num_frames = length(t_events);

% --- FUNCIONES AUXILIARES PARA INDEXACIÓN SEGURA ---
% Simulink a veces exporta matrices 3D [H,W,N] y otras veces 4D [H,W,1,N].
if ndims(ev_on_data) == 4
    get_frame_ev = @(mat, idx) double(mat(:,:,1,idx));
else
    get_frame_ev = @(mat, idx) double(mat(:,:,idx));
end

if ndims(img_data) == 4
    get_frame_img = @(mat, idx) mat(:,:,1,idx);
else
    get_frame_img = @(mat, idx) mat(:,:,idx);
end

% Inicialización del primer fotograma RGB de eventos [R, G, B]
R_init = get_frame_ev(ev_off_data, 1);
G_init = get_frame_ev(ev_on_data, 1);
B_init = zeros(size(R_init));
first_ev_frame = cat(3, R_init, G_init, B_init);

% 2. Configuración de la figura
f = figure('Name', 'DVS Event Tracker', 'Position', [100, 100, 1000, 450], 'Color', 'w');

% Eje izquierdo: Eventos (R=OFF, G=ON)
ax1 = subplot(1, 2, 1);
h_ev = imshow(first_ev_frame, 'Parent', ax1); 
title_ev = title(ax1, 'Eventos (R=OFF, G=ON)', 'FontSize', 12);

% Eje derecho: Imagen Real (B/W)
ax2 = subplot(1, 2, 2);
h_img = imshow(get_frame_img(img_data, 1), 'Parent', ax2);
title_img = title(ax2, 'Imagen Real (B/W)', 'FontSize', 12);

% 3. INICIALIZACIÓN DEL GRABADOR DE VÍDEO
v = VideoWriter(video_filename, 'MPEG-4'); 
v.FrameRate = fps;                         
v.Quality = 95;                            
open(v);

% =========================================================================
% 4. BUCLE DE REPRODUCCIÓN Y GRABACIÓN
% =========================================================================
disp('Grabando vídeo... Por favor, NO minimices ni tapes la figura.');

% Forzamos a que la ventana de la figura se ponga en primer plano
figure(f); 

for i = 1:num_frames
    t_current = t_events(i);
    
    % --- Construcción de la imagen RGB de Eventos ---
    R_current = get_frame_ev(ev_off_data, i);
    G_current = get_frame_ev(ev_on_data, i); 
    
    % Combinamos los canales: Rojo para OFF, Verde para ON, Azul a cero
    current_ev_frame = cat(3, R_current, G_current, zeros(size(R_current)));
    
    % Actualización gráfica de eventos modificando CData (mucho más rápido)
    set(h_ev, 'CData', current_ev_frame);
    title_ev.String = sprintf('Eventos - Tiempo: %.3f s', t_current);
    
    % --- Actualización de la imagen Real ---
    set(h_img, 'CData', get_frame_img(img_data, i));
    title_img.String = sprintf('Imagen Real - Tiempo: %.3f s', t_current);
    
    % 1. Solución Previsualización: Obliga al sistema operativo a refrescar la pantalla
    drawnow;
    pause(0.001); 
    
    % 2. Captura del fotograma de la figura
    frame = getframe(f); 
    
    % 3. Solución Distorsión Diagonal: Forzar dimensiones pares para MPEG-4
    [alto, ancho, ~] = size(frame.cdata);
    if rem(alto, 2) ~= 0
        alto = alto - 1;
    end
    if rem(ancho, 2) ~= 0
        ancho = ancho - 1;
    end
    
    % Recortamos el frame a las dimensiones pares antes de guardarlo
    frame.cdata = frame.cdata(1:alto, 1:ancho, :);
    
    % Escribir fotograma en el archivo de vídeo
    writeVideo(v, frame);
end

% --- CI