function [stars, star_intensity, star_sizes, star_colors] = generateStars(num_stars, min_dist, max_dist)
    % Generar Ascensión Recta aleatoria (0 a 2*pi)
    RA = 2 * pi * rand(num_stars, 1);
    
    % Generar Declinación aleatoria (usando asin para evitar agrupamiento en los polos)
    sin_Dec = 2 * rand(num_stars, 1) - 1;
    Dec = asin(sin_Dec);
    
    % Generar distancias aleatorias en el rango [min_dist, max_dist]
    star_r = 1000; % min_dist + (max_dist - min_dist) * rand(num_stars, 1);
    
    % Convertir Esféricas (RA, Dec, radio aleatorio) a Cartesianas (x, y, z)
    star_x = star_r .* cos(Dec) .* cos(RA);
    star_y = star_r .* cos(Dec) .* sin(RA);
    star_z = star_r .* sin(Dec);
    
    % Agrupar coordenadas espaciales en una matriz [num_stars x 3]
    stars = [star_x, star_y, star_z];
    
    % 1. Asignar intensidad aleatoria entre 0.2 y 1.0
    star_intensity = 0.2 + 0.8 * rand(num_stars, 1);
end