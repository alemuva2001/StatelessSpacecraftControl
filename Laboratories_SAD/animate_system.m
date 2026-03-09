function animate_system(tout, r_N_sat, A_BN, S_N_vec, Rt, speed_multiplier)
    % ANIMATE_SYSTEM: Versión "Full Control" con Telemetría, Slider y Precisión Temporal
    
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

    lado = 1; s = lado / 2;
    vertices_cubo_orig = [-s -s -s; s -s -s; s s -s; -s s -s; -s -s s; s -s s; s s s; -s s s];
    caras_cubo = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];

    %% 2. FIGURA E INTERFAZ
    fig = figure('Name', 'Mission Control Center: Precision Analyzer', 'WindowState', 'maximized', 'Color', 'k');
    % UserData: [K_actual, IsRunning, StopFlag, T0_Sim]
    set(fig, 'UserData', [1, 1, 0, tout(1)]); 

    p_ctrl = uipanel('Parent', fig, 'Position', [0 0.05 1 0.06], 'BackgroundColor', [0.1 0.1 0.1], 'BorderType', 'none');
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'PLAY / PAUSE', 'Units', 'normalized', 'Position', [0.05 0.1 0.15 0.8], 'Callback', @(s,e) toggle_play(fig));
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'RESET', 'Units', 'normalized', 'Position', [0.22 0.1 0.15 0.8], 'Callback', @(s,e) reset_sim(fig));
    uicontrol('Parent', p_ctrl, 'Style', 'pushbutton', 'String', 'STOP (Analyzer Mode)', 'Units', 'normalized', 'Position', [0.75 0.1 0.2 0.8], 'BackgroundColor', [0.6 0.2 0.2], 'ForegroundColor', 'w', 'Callback', @(s,e) stop_loop(fig));

    p_time = uipanel('Parent', fig, 'Position', [0 0 1 0.05], 'BackgroundColor', [0.05 0.05 0.05], 'BorderType', 'none');
    h_slider = uicontrol('Parent', p_time, 'Style', 'slider', 'Units', 'normalized', 'Position', [0.05 0.2 0.9 0.6], 'Min', 1, 'Max', N, 'Value', 1, 'Callback', @(s,e) set_index(fig, round(get(s, 'Value'))));

    % Subplots
    ax_orb = subplot(1, 2, 1, 'Parent', fig, 'Position', [0.05 0.15 0.4 0.75], 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    hold(ax_orb, 'on'); axis(ax_orb, 'equal'); grid(ax_orb, 'on'); view(ax_orb, 3);
    h_txt_orb = text(ax_orb, 0.05, 0.9, '', 'Units', 'normalized', 'Color', 'c', 'FontName', 'Courier', 'FontSize', 10, 'FontWeight', 'bold');

    ax_att = subplot(1, 2, 2, 'Parent', fig, 'Position', [0.55 0.15 0.4 0.75], 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    hold(ax_att, 'on'); axis(ax_att, 'equal'); grid(ax_att, 'on'); view(ax_att, 30, 20);
    h_txt_att = text(ax_att, 0.05, 0.9, '', 'Units', 'normalized', 'Color', 'y', 'FontName', 'Courier', 'FontSize', 10, 'FontWeight', 'bold');
    
    lim_att = lado * 1.5; axis(ax_att, [-lim_att lim_att -lim_att lim_att -lim_att lim_att]);
    color_N = [0.5 0.5 0.5];
    plot3(ax_att, [0 lim_att*0.8], [0 0], [0 0], '--', 'Color', color_N); % Ejes Inerciales N
    plot3(ax_att, [0 0], [0 lim_att*0.8], [0 0], '--', 'Color', color_N);
    plot3(ax_att, [0 0], [0 0], [0 lim_att*0.8], '--', 'Color', color_N);

    %% 3. OBJETOS GRÁFICOS
    [X_e, Y_e, Z_e] = sphere(40);
    surf(ax_orb, X_e*R_earth_vis, Y_e*R_earth_vis, Z_e*R_earth_vis, 'EdgeColor', 'none', 'FaceColor', [0.1 0.4 0.9]);
    h_sat_pos = plot3(ax_orb, 0, 0, 0, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 5);
    plot3(ax_orb, r_N_mat(:,1), r_N_mat(:,2), r_N_mat(:,3), 'w:', 'LineWidth', 0.5);
    
    % Ejes Body en Órbita
    h_orb_x = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'r', 'LineWidth', 1.5);
    h_orb_y = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'g', 'LineWidth', 1.5);
    h_orb_z = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'b', 'LineWidth', 1.5);
    h_orb_sun = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'y', 'LineWidth', 2);
    h_orb_nad = quiver3(ax_orb, 0,0,0, 0,0,0, 0, 'm', 'LineWidth', 2);

    % Cubo y Ejes Body en Actitud
    cubo_patch = patch(ax_att, 'Vertices', vertices_cubo_orig, 'Faces', caras_cubo, 'FaceColor', [0.3 0.6 1.0], 'FaceAlpha', 0.5);
    h_att_x = plot3(ax_att, 0, 0, 0, 'r-', 'LineWidth', 3);
    h_att_y = plot3(ax_att, 0, 0, 0, 'g-', 'LineWidth', 3);
    h_att_z = plot3(ax_att, 0, 0, 0, 'b-', 'LineWidth', 3);
    h_att_sun = quiver3(ax_att, 0,0,0, 0,0,0, 0, 'y', 'LineWidth', 2.5);
    h_att_nad = quiver3(ax_att, 0,0,0, 0,0,0, 0, 'm', 'LineWidth', 2.5);

    h_hud = annotation('textbox', [0.1, 0.9, 0.8, 0.08], 'Color', 'y', 'EdgeColor', 'none', 'FontName', 'Courier', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    rotate3d(fig, 'on');

    %% 4. BUCLE DE REPRODUCCIÓN (Reloj Maestro)
    t_master = tic;
    gui = get(fig, 'UserData');

    while isvalid(fig)
        gui = get(fig, 'UserData');
        if gui(3) == 1, break; end 
        
        k = gui(1); is_running = gui(2); t0_sim = gui(4);
        
        if is_running
            t_real_elapsed = toc(t_master);
            t_target_sim = t0_sim + t_real_elapsed * speed_multiplier;
            [~, k] = min(abs(tout - t_target_sim));
            if k >= N, k = N; gui(2) = 0; end
            gui(1) = k; set(fig, 'UserData', gui);
        else
            t_master = tic;
            gui(4) = tout(k);
            set(fig, 'UserData', gui);
        end

        % Sincronización de Datos
        t = tout(k); pos_N = r_N_mat(k, :);
        s_N = S_N_mat(k, :)' / (norm(S_N_mat(k, :)) + eps);
        n_N = -pos_N' / (norm(pos_N) + eps);
        A = A_BN(:,:,k); A_NB = A'; 

        % Cálculo de Euler para Telemetría
        pitch = rad2deg(-asin(A(1,3))); roll = rad2deg(atan2(A(2,3), A(3,3))); yaw = rad2deg(atan2(A(1,2), A(1,1)));

        % Actualizar Vista Orbital (Distancias en km)
        L_orb = R_earth_vis * 0.8;
        set(h_sat_pos, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3));
        set(h_orb_x, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', A_NB(1,1)*L_orb, 'VData', A_NB(2,1)*L_orb, 'WData', A_NB(3,1)*L_orb);
        set(h_orb_y, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', A_NB(1,2)*L_orb, 'VData', A_NB(2,2)*L_orb, 'WData', A_NB(3,2)*L_orb);
        set(h_orb_z, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', A_NB(1,3)*L_orb, 'VData', A_NB(2,3)*L_orb, 'WData', A_NB(3,3)*L_orb);
        set(h_orb_sun, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', s_N(1)*L_orb*2.5, 'VData', s_N(2)*L_orb*2.5, 'WData', s_N(3)*L_orb*2.5);
        set(h_orb_nad, 'XData', pos_N(1), 'YData', pos_N(2), 'ZData', pos_N(3), 'UData', n_N(1)*L_orb*2.5, 'VData', n_N(2)*L_orb*2.5, 'WData', n_N(3)*L_orb*2.5);
        set(h_txt_orb, 'String', sprintf('POSITION (km):\nX: %.1f\nY: %.1f\nZ: %.1f', pos_N(1)/1000, pos_N(2)/1000, pos_N(3)/1000));

        % Actualizar Vista de Actitud (Ángulos en deg)
        set(cubo_patch, 'Vertices', (A_NB * vertices_cubo_orig')');
        set(h_att_x, 'XData', [0 A_NB(1,1)], 'YData', [0 A_NB(2,1)], 'ZData', [0 A_NB(3,1)]);
        set(h_att_y, 'XData', [0 A_NB(1,2)], 'YData', [0 A_NB(2,2)], 'ZData', [0 A_NB(3,2)]);
        set(h_att_z, 'XData', [0 A_NB(1,3)], 'YData', [0 A_NB(2,3)], 'ZData', [0 A_NB(3,3)]);
        set(h_att_sun, 'UData', s_N(1)*lado, 'VData', s_N(2)*lado, 'WData', s_N(3)*lado);
        set(h_att_nad, 'UData', n_N(1)*lado, 'VData', n_N(2)*lado, 'WData', n_N(3)*lado);
        set(h_txt_att, 'String', sprintf('EULER (deg):\nRoll:  %.1f\nPitch: %.1f\nYaw:   %.1f', roll, pitch, yaw));

        % HUD y Slider
        set(h_hud, 'String', sprintf('TIME: %s / %s (Speed: %gx) [%s]', ...
            datestr(t/86400, 'HH:MM:SS'), str_total, speed_multiplier, if_else(is_running,'PLAYING','PAUSED')));
        set(h_slider, 'Value', k);

        drawnow limitrate;
    end
end

%% FUNCIONES INTERNAS
function toggle_play(fig), d = get(fig, 'UserData'); d(2) = ~d(2); set(fig, 'UserData', d); end
function reset_sim(fig), d = get(fig, 'UserData'); d(1) = 1; d(2) = 1; d(4) = 0; set(fig, 'UserData', d); end
function set_index(fig, val), d = get(fig, 'UserData'); d(1) = val; d(4) = 0; set(fig, 'UserData', d); end
function stop_loop(fig)
    d = get(fig, 'UserData'); d(3) = 1; set(fig, 'UserData', d);
    h = findall(fig, 'Type', 'annotation'); if ~isempty(h), set(h, 'String', 'ANALYZER MODE ACTIVE', 'Color', 'r'); end
end
function out = if_else(c,t,f), if c, out=t; else, out=f; end; end