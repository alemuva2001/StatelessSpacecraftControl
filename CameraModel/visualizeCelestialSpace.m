function visualizeCelestialSpace(stars, star_intensity)
% visualizeCelestialSpace Plotea el modelo espacial 3D.
%   Toma como entrada las posiciones de estrellas y su intensidad.
%   Calcula internamente el tamaño y el color para la representación visual.

    % Extraer coordenadas
    star_x = stars(:, 1);
    star_y = stars(:, 2);
    star_z = stars(:, 3);
    
    num_stars = length(star_x);

    % --- LÓGICA DE VISUALIZACIÓN (Mapeo de datos a píxeles) ---
    % 1. Mapear la intensidad al TAMAÑO de representación visual
    star_sizes = 2 + 48 * star_intensity; 
    
    % 2. Mapear la intensidad al COLOR RGB (escala de grises)
    star_colors = repmat(star_intensity, 1, 3); 

    % --- Configurar la Figura ---
    figure('Name', sprintf('Modelo Celeste (%d Estrellas)', num_stars), ...
           'Color', 'k', 'Position', [100, 100, 800, 800]);
    hold on; axis equal;
    view(3);
    
    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', ...
             'GridColor', 'w', 'GridAlpha', 0.3);
    grid on;
    xlabel('X (Equinoccio Vernal)'); ylabel('Y'); zlabel('Z (Polo Norte Celeste)');
    title('Modelo Espacial Celeste 3D', 'Color', 'w', 'FontSize', 14);

    % --- Radios y Cuerpos Celestes de Referencia ---
    r_earth = 1;        
    r_celestial = 5;    

    % Dibujar la Tierra
    [x, y, z] = sphere(50);
    surf(x * r_earth, y * r_earth, z * r_earth, ...
        'FaceColor', [0.1 0.4 0.8], 'EdgeColor', 'none', 'FaceAlpha', 1.0);

    % Dibujar la Esfera Celeste interior
    surf(x * r_celestial, y * r_celestial, z * r_celestial, ...
        'FaceColor', 'none', 'EdgeColor', [0.2 0.2 0.2], 'FaceAlpha', 0.1);

    % Dibujar Ecuador y Eclíptica
    theta = linspace(0, 2*pi, 100);
    eq_x = r_celestial * cos(theta);
    eq_y = r_celestial * sin(theta);
    eq_z = zeros(size(theta));
    plot3(eq_x, eq_y, eq_z, 'b', 'LineWidth', 1.5, 'DisplayName', 'Ecuador Celeste');
    
    tilt_rad = deg2rad(23.5);
    ecliptic_y = r_celestial * sin(theta) * cos(tilt_rad);
    ecliptic_z = r_celestial * sin(theta) * sin(tilt_rad);
    plot3(eq_x, ecliptic_y, ecliptic_z, 'r', 'LineWidth', 1.5, 'DisplayName', 'Eclíptica');

    % --- Plotear las Estrellas ---
    % Aplicar los cálculos de tamaño y color que hicimos al principio de la función
    scatter3(star_x, star_y, star_z, star_sizes, star_colors, 'filled', ...
        'MarkerEdgeColor', 'none', 'DisplayName', 'Estrellas');

    % --- Marcas de Referencia ---
    plot3([0 0], [0 0], [-r_celestial*1.1 r_celestial*1.1], 'w--', ...
          'LineWidth', 1, 'DisplayName', 'Eje de Rotación');
    text(0, 0, r_celestial*1.15, 'PNC', 'Color', 'w', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    text(0, 0, -r_celestial*1.15, 'PSC', 'Color', 'w', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    plot3(r_celestial, 0, 0, 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g', ...
          'DisplayName', 'Equinoccio Vernal');

    camlight right;
    lighting gouraud;
    legend('TextColor', 'w', 'Color', 'none', 'Location', 'northeastoutside');
    hold off;
end