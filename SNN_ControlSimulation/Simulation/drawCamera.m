function drawCamera(R_c)
  
    % 1. Dibujar la cámara en el origen (0, 0, 0)
    % Cambiado a negro ('ks') para diferenciarlo del eje X rojo
    plot3(0, 0, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'DisplayName', 'Cámara');
    hold on;
    
    % 2. Definir los vectores base locales (Identidad)
    local_X = [1; 0; 0]; % Eje lateral (Derecha)
    local_Y = [0; 1; 0]; % Eje vertical (Abajo)
    local_Z = [0; 0; 1]; % Eje óptico (Frente)
    
    % 3. Rotar los vectores al marco del mundo
    % Matemáticamente, esto equivale a extraer las columnas de R_c
    world_X = R_c * local_X;
    world_Y = R_c * local_Y;
    world_Z = R_c * local_Z;
    
    % 4. Escalar para hacerlos visibles
    length = 250;
    v_X = world_X * length;
    v_Y = world_Y * length;
    v_Z = world_Z * length;
    
    % 5. Dibujar los vectores con quiver3 usando la convención RGB
    
    % Eje X (Rojo)
    quiver3(0, 0, 0, v_X(1), v_X(2), v_X(3), 0, ...
        'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName', 'Eje X_c (Derecha)');
        
    % Eje Y (Verde)
    quiver3(0, 0, 0, v_Y(1), v_Y(2), v_Y(3), 0, ...
        'Color', 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName', 'Eje Y_c (Abajo)');
        
    % Eje Z (Azul) - Tu antiguo vector de dirección
    quiver3(0, 0, 0, v_Z(1), v_Z(2), v_Z(3), 0, ...
        'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName', 'Eje Z_c (Frente/Óptico)');
   
end