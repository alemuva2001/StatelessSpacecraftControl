function drawCelestialSpace(num_stars)
% drawCelestialSpace Generates a 3D celestial sphere model with stars.
%   drawCelestialSpace(num_stars) plots the model with the specified 
%   number of stars.

    % Default to 100 stars if no parameter is provided
    if nargin < 1
        num_stars = 100; 
    end

    % --- 1. Setup the Figure (Space Theme) ---
    figure('Name', sprintf('Celestial Space Model (%d Stars)', num_stars), ...
           'Color', 'k', 'Position', [100, 100, 800, 800]);
    hold on; axis equal;
    view(3); % Set 3D view

    % Set axes colors to white so they show up against the black background
    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', ...
             'GridColor', 'w', 'GridAlpha', 0.3);
    grid on;
    xlabel('X (Vernal Equinox)'); ylabel('Y'); zlabel('Z (North Celestial Pole)');
    title(sprintf('3D Celestial Space Model with %d Stars', num_stars), ...
          'Color', 'w', 'FontSize', 14);

    % --- 2. Define Radii ---
    r_earth = 1;        % Radius of the Earth
    r_celestial = 5;    % Radius of the Celestial Sphere

    % --- 3. Draw the Earth ---
    [x, y, z] = sphere(50);
    surf(x * r_earth, y * r_earth, z * r_earth, ...
        'FaceColor', [0.1 0.4 0.8], 'EdgeColor', 'none', 'FaceAlpha', 1.0);

    % --- 4. Draw the Celestial Sphere (Faint Wireframe) ---
    surf(x * r_celestial, y * r_celestial, z * r_celestial, ...
        'FaceColor', 'none', 'EdgeColor', [0.2 0.2 0.2], 'FaceAlpha', 0.1);

    % --- 5. Draw the Celestial Equator & Ecliptic ---
    theta = linspace(0, 2*pi, 100);
    eq_x = r_celestial * cos(theta);
    eq_y = r_celestial * sin(theta);
    eq_z = zeros(size(theta));

    plot3(eq_x, eq_y, eq_z, 'b', 'LineWidth', 1.5, 'DisplayName', 'Celestial Equator');

    % Ecliptic (Tilted by 23.5 degrees)
    tilt_rad = deg2rad(23.5);
    ecliptic_y = r_celestial * sin(theta) * cos(tilt_rad);
    ecliptic_z = r_celestial * sin(theta) * sin(tilt_rad);
    plot3(eq_x, ecliptic_y, ecliptic_z, 'r', 'LineWidth', 1.5, 'DisplayName', 'Ecliptic');

    % --- 6. Generate and Plot Stars ---
    % Generate random Right Ascension (0 to 2*pi)
    RA = 2 * pi * rand(num_stars, 1);

    % Generate random Declination (using asin to avoid pole clumping)
    sin_Dec = 2 * rand(num_stars, 1) - 1;
    Dec = asin(sin_Dec);

    % Convert Spherical (RA, Dec) to Cartesian (x, y, z)
    star_x = r_celestial * cos(Dec) .* cos(RA);
    star_y = r_celestial * cos(Dec) .* sin(RA);
    star_z = r_celestial * sin(Dec);

    % Generate random sizes for the stars (magnitude)
    star_sizes = 2 + 15 * rand(num_stars, 1);

    % Plot the stars
    scatter3(star_x, star_y, star_z, star_sizes, 'w', 'filled', ...
        'MarkerEdgeColor', 'none', 'DisplayName', 'Stars');

    % Add a few visually distinct "bright" stars if we have enough stars
    if num_stars >= 15
        scatter3(star_x(1:15), star_y(1:15), star_z(1:15), 35, ...
                 [1 0.9 0.7], 'filled', 'HandleVisibility', 'off');
    end

    % --- 7. Mark the Poles and Vernal Equinox ---
    plot3([0 0], [0 0], [-r_celestial*1.1 r_celestial*1.1], 'w--', ...
          'LineWidth', 1, 'DisplayName', 'Rotation Axis');
    text(0, 0, r_celestial*1.15, 'NCP', 'Color', 'w', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    text(0, 0, -r_celestial*1.15, 'SCP', 'Color', 'w', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    plot3(r_celestial, 0, 0, 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g', ...
          'DisplayName', 'Vernal Equinox');

    % --- 8. Final Touches ---
    camlight right;
    lighting gouraud;
    legend('TextColor', 'w', 'Color', 'none', 'Location', 'northeastoutside');
    hold off;
end