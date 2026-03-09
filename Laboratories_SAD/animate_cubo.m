function animate_system(tout, r_N_sat, A_BN, S_N_vec, Rt, speed_multiplier)
    % ANIMATE_SYSTEM: Mission Control (Orbit + Attitude) con Control de Cámara Libre
    
    if nargin < 6 || isempty(speed_multiplier), speed_multiplier = 100; end
    
    %% 1. PREPARACIÓN DE DATOS
    r_N_mat = squeeze(r_N_sat);
    if size(r_N_mat, 1) == 3 && size(r_N_mat, 2) ~= 3, r_N_mat = r_N_mat'; end
    S_N_mat = squeeze(S_N_vec);
    if size(S_N_mat, 1) == 3 && size(S_N_mat, 2) ~= 3, S_N_mat = S_N_mat'; end

    R_earth_vis = Rt * 1000; 
    N = length(tout);
    T_total = tout(end);
    str_total = sprintf('%02dd %02dh %02dm %02ds', floor(T_total/86400), ...
                floor(mod(T_total,86400)/3600), floor(mod(T_total,3600)/60), floor(mod(T_total,60)));

    % Geometría del Cubo (del código proporcionado)
    lado = 1; s = lado / 2;
    vertices_cubo_orig = [-s -s -s; s -s -s; s s -s; -s s -s; -s -s s; s -s s; s s s; -s s s];
    caras_cubo = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];

    %% 2. CONFIGURACIÓN DE FIGURA E INTERFAZ
    fig = figure('Name', 'Mission Control Center', 'WindowState', 'maximized', 'Color', 'k');
    % UserData: [Modo_Camara, Flag_Update_Camara]
    set(fig, 'UserData', [3, 1]); 

    % Panel de Botones
    p_ctrl = uipanel('Parent', fig, 'Position', [0 0 1 0.08], 'BackgroundColor', [0.1 0.1 0.1], 'BorderType', 'none');
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'SYSTEM VIEW', 'Units', 'normalized', 'Position', [0.1 0.2 0.15 0.6], 'Callback', @(s,e) set(fig, 'UserData', [3, 1]));
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'EARTH VIEW', 'Units', 'normalized', 'Position', [0.3 0.2 0.15 0.6], 'Callback', @(s,e) set(fig, 'UserData', [1, 1]));
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'SAT VIEW', 'Units', 'normalized', 'Position', [0.5 0.2 0.15 0.6], 'Callback', @(s,e) set(fig, 'UserData', [2, 1]));
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'FREE ROTATION', 'Units', 'normalized', 'Position', [0.7 0.2 0.15 0.6], 'Callback', @(s,e) set(fig, 'UserData', [0, 0]));

    % --- SUBPLOT 1: ÓRBITA ---
    ax_orb = subplot(1, 2, 1, 'Parent', fig, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    hold(ax_orb, 'on'); axis(ax_orb, 'equal'); grid(ax_orb, 'on'); view(ax_orb, 3);
    h_txt_orb = text(ax_orb, 0.05, 0.95, '', 'Units', 'normalized', 'Color', 'c', 'FontName', 'Courier', 'FontSize', 10, 'FontWeight', 'bold');

    % --- SUBPLOT 2: ACTITUD (CUBO) ---
    ax_att = subplot(1, 2, 2, 'Parent', fig, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    hold(ax_att, 'on'); axis(ax_att, 'equal'); grid(ax_att, 'on');
    lim_att = lado * 1.5; axis(ax_att, [-lim_att lim_att -lim_att lim_att -lim_att lim_att]);
    view(ax_att, 30, 20);
    h_txt_att = text(ax_att, 0.05, 0.95, '', 'Units', 'normalized', 'Color', 'y', 'FontName', 'Courier', 'FontSize', 10, 'FontWeight', 'bold');

    % Habilitar rotación manual para toda la figura
    rotate3d(fig, 'on');

    %% 3. INICIALIZACIÓN DE OBJETOS
    % Elementos Órbita
    [X_e, Y_e, Z_e] = sphere(40);
    surf(ax_orb, X_e*R_earth_vis, Y_e*R_earth_vis, Z_e*R_earth_vis, 'EdgeColor', 'none', 'FaceColor', [0.1 0.4 0.9]);
    h_sat_pos = plot3(ax_orb, 0, 0, 0, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 5);
    plot3(ax_orb, r_N_mat(:,1), r_N_mat(:,2), r_N_mat(:,3), 'c:', 'LineWidth', 0.5);
    h_orb_x = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'r', 'LineWidth', 1.5);
    h_orb_y = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'g', 'LineWidth', 1.5);
    h_orb_z = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'b', 'LineWidth', 1.5);
    h_orb_sun = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'y', 'LineWidth', 2);

    % Elementos Actitud
    cubo_patch = patch(ax_att, 'Vertices', vertices_cubo_orig, 'Faces', caras_cubo, 'FaceColor', [0.3 0.6 1.0], 'FaceAlpha', 0.6, 'EdgeColor', 'k');
    h_att_x = plot3(ax_att, 0, 0, 0, 'r-', 'LineWidth', 3);
    h_att_y = plot3(ax_att, 0, 0, 0, 'g-', 'LineWidth', 3);
    h_att_z = plot3(ax_att, 0, 0, 0, 'b-', 'LineWidth', 3);
    h_att_sun = quiver3(ax_att, 0,0,0, 0,0,0, 0, 'y', 'LineWidth', 3, 'MaxHeadSize', 0.5);

    h_hud = annotation('textbox', [0.1, 0.9, 0.8, 0.08], 'Color', 'y', 'EdgeColor', 'none', 'FontName', 'Courier', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    %% 4. BUCLE DE ANIMACIÓN
    t_sim_prev = tout(1); skip = max(1, floor(N/400)); tic;
    
    for k = 1:skip:N
        if ~isvalid(fig), break; end
        t = tout(k);
        pos_N = r_N_mat(k, :);
        s_N = S_N_mat(k, :)' / norm(S_N_mat(k, :));
        
        A = A_BN(:,:,k); A_NB = A'; 
        pitch = -asin(A(1,3)); roll = atan2(A(2,3), A(3,3)); yaw = atan2(A(1,2), A(1,1));

        % --- ACTUALIZAR VISTA ORBITAL ---
        set(h_sat_pos, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3));
        set(h_orb_x, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', A_NB(1,1)*R_earth_vis, 'VData', A_NB(2,1)*R_earth_vis, 'WData', A_NB(3,1)*R_earth_vis);
        set(h_orb_y, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', A_NB(1,2)*R_earth_vis, 'VData', A_NB(2,2)*R_earth_vis, 'WData', A_NB(3,2)*R_earth_vis);
        set(h_orb_z, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', A_NB(1,3)*R_earth_vis, 'VData', A_NB(2,3)*R_earth_vis, 'WData', A_NB(3,3)*R_earth_vis);
        set(h_orb_sun, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', s_N(1)*R_earth_vis*2.5, 'VData', s_N(2)*R_earth_vis*2.5, 'WData', s_N(3)*R_earth_vis*2.5);
        set(h_txt_orb, 'String', sprintf('POSITION (km):\nX: %.1f\nY: %.1f\nZ: %.1f', pos_N(1)/1000, pos_N(2)/1000, pos_N(3)/1000));

        % --- ACTUALIZAR VISTA DE ACTITUD ---
        s_B = A * s_N; 
        vertices_rot = (A_NB * vertices_cubo_orig')';
        set(cubo_patch, 'Vertices', vertices_rot);
        set(h_att_x, 'XData', [0 A_NB(1,1)], 'YData', [0 A_NB(2,1)], 'ZData', [0 A_NB(3,1)]);
        set(h_att_y, 'XData', [0 A_NB(1,2)], 'YData', [0 A_NB(2,2)], 'ZData', [0 A_NB(3,2)]);
        set(h_att_z, 'XData', [0 A_NB(1,3)], 'YData', [0 A_NB(2,3)], 'ZData', [0 A_NB(3,3)]);
        set(h_att_sun, 'UData', s_B(1)*lado, 'VData', s_B(2)*lado, 'WData', s_B(3)*lado);
        set(h_txt_att, 'String', sprintf('EULER ANGLES (deg):\nRoll:  %.1f\nPitch: %.1f\nYaw:   %.1f', rad2deg(roll), rad2deg(pitch), rad2deg(yaw)));

        % --- CONTROL DE CÁMARA INTELIGENTE ---
        gui_state = get(fig, 'UserData');
        if gui_state(2) == 1 % Solo actualiza si se pulsó un botón
            switch gui_state(1)
                case 1 % Earth View
                    camtarget(ax_orb, [0 0 0]); campos(ax_orb, [R_earth_vis*5, R_earth_vis*5, R_earth_vis*2]);
                case 2 % Sat View
                    camtarget(ax_orb, pos_N); campos(ax_orb, pos_N + [R_earth_vis, R_earth_vis, R_earth_vis*0.5]);
                case 3 % System View
                    camtarget(ax_orb, [0 0 0]); campos(ax_orb, [pos_N(1)*2, pos_N(2)*2, R_earth_vis*10]);
            end
            set(fig, 'UserData', [gui_state(1), 0]); % Reset flag para permitir movimiento manual
        end

        set(h_hud, 'String', sprintf('TIME: %02dd %02dh %02dm %02ds / %s (Speed: %gx)', floor(t/86400), floor(mod(t,86400)/3600), floor(mod(t,3600)/60), floor(mod(t,60)), str_total, speed_multiplier));
        drawnow; 
        pause(max(0, (t - t_sim_prev)/speed_multiplier - toc)); tic;
        t_sim_prev = t;
    end
end