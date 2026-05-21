function explorador_pixeles(data1, data2)
    % data1, data2: matrices de 101x101x701 (asumimos que miden lo mismo)
    [rows, cols, steps] = size(data1);
    t = 1:steps;
    
    % Sumamos la matriz a lo largo de la dimensión 3 (el tiempo) usando valor absoluto
    matriz_actividad = sum(abs(data1), 3); 
    
    fig = figure('Name', 'Explorador de Actividad', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 450]);
    
    % Panel 1: Mapa de calor de la ACTIVIDAD TOTAL
    ax1 = subplot(1, 2, 1);
    img = imagesc(matriz_actividad); 
    colorbar;
    title('Actividad Total (Data1). Haz clic en un píxel');
    xlabel('Columna'); ylabel('Fila');
    
    % Panel 2: Evolución temporal (Líneas)
    ax2 = subplot(1, 2, 2);
    
    % --- CORRECCIÓN 1: Guardar handles separados ---
    line_plot1 = plot(t, squeeze(data1(51, 51, :)), 'LineWidth', 1.5, 'DisplayName', 'Data 1');
    hold on;
    line_plot2 = plot(t, squeeze(data2(51, 51, :)), 'LineWidth', 1.5, 'DisplayName', 'Data 2');
    grid on;
    legend('show'); % Añadida leyenda
    title('Evolución Temporal: Fila 51, Col 51');
    xlabel('Tiempo'); ylabel('Valor');
    hold off;
    
    % --- CORRECCIÓN 2: Pasar todas las variables necesarias al callback ---
    set(img, 'ButtonDownFcn', @(src, event) actualizar_grafica(src, event, data1, data2, ax2, line_plot1, line_plot2));
end

% --- CORRECCIÓN 3: Recibir las nuevas variables ---
function actualizar_grafica(src, ~, data1, data2, ax2, line_plot1, line_plot2)
    % Obtener coordenadas exactas del clic
    coords = get(gca, 'CurrentPoint');
    c = round(coords(1,1));
    r = round(coords(1,2));
    
    % Validar que el clic no se salga de los límites de la matriz
    [max_r, max_c, ~] = size(data1);
    if r > 0 && r <= max_r && c > 0 && c <= max_c
        % --- CORRECCIÓN 4: Actualizar ambas gráficas ---
        nuevo_perfil1 = squeeze(data1(r, c, :));
        nuevo_perfil2 = squeeze(data2(r, c, :));
        
        % Actualizar la gráfica de la derecha
        set(line_plot1, 'YData', nuevo_perfil1);
        set(line_plot2, 'YData', nuevo_perfil2);
        
        title(ax2, sprintf('Evolución del Píxel: Fila %d, Col %d', r, c));
    end
end