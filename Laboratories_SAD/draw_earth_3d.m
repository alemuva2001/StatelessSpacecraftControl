function draw_earth_3d(R_earth_m)
    % DRAW_EARTH_3D Dibuja la Tierra y los ejes inerciales.
    % Input:
    %   R_earth_m - Radio de la Tierra en METROS.

    % 1. Dibujar la esfera de la Tierra
    [X, Y, Z] = sphere(50);
    surf(X * R_earth_m, Y * R_earth_m, Z * R_earth_m, ...
        'EdgeColor', 'none', 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.8);
    hold on;
    
    % 2. Dibujar el Sistema de Referencia Inercial (Frame N)
    % Las flechas sobresaldrán un 50% más allá del radio de la Tierra
    axis_len = R_earth_m * 1.5; 
    
    % Eje X Inercial (Rojo - Apunta al Punto Aries)
    quiver3(0, 0, 0, axis_len, 0, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    text(axis_len * 1.1, 0, 0, 'X_N', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 12);
    
    % Eje Y Inercial (Verde)
    quiver3(0, 0, 0, 0, axis_len, 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    text(0, axis_len * 1.1, 0, 'Y_N', 'Color', 'g', 'FontWeight', 'bold', 'FontSize', 12);
    
    % Eje Z Inercial (Azul - Eje de rotación terrestre)
    quiver3(0, 0, 0, 0, 0, axis_len, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    text(0, 0, axis_len * 1.1, 'Z_N', 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 12);
    
    % 3. Ajustes de visualización
    axis equal; % OBLIGATORIO para que la Tierra sea redonda y no un óvalo
end