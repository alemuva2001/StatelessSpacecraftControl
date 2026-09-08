function [hFig]=DibujaTorque(hFig, result, pt1, PD_CONTROLLER, USE_SNN)
tic;
% 1. Comprobar si la figura existe y es válida, si no, crearla
    if isempty(hFig) || ~isvalid(hFig)
        hFig = figure('Name', 'Resultados en Tiempo Real', 'NumberTitle', 'off');
    end
    
    % 2. Limpiar la figura actual (pero mantener la ventana abierta)
    clf(hFig);

ax1 = subplot(2, 3, [1, 2, 3]); % Guardamos el handle en ax1
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
if PD_CONTROLLER 
    plot(result.tout, result.error(:,2:4), 'LineWidth', 1.5)
else
    plot(result.tout, result.errorR(:,2:4), 'LineWidth', 1.5)
end
grid on; grid minor;
title('\textbf{Quaternion error}','FontSize',18)
xlabel("Time $[s]$")
legend('$e_{q2}$', '$e_{q3}$', '$e_{q4}$', 'Location', 'best')
hold off

%%--- EJE X ---
ax2 = subplot(2, 3, 4); % Guardamos el handle en ax2
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Extraemos la columna 1 (X)
if PD_CONTROLLER
    plot(result.tout, result.T_control_PID(:, 1), 'LineWidth', 1.5, 'Color', '#0072BD') % Azul
end
if USE_SNN
    plot(result.tout, result.T_control_SNN(:, 1), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', '#D95319') % Naranja rojizo
end
grid on; grid minor;
title('\textbf{Torque Eje X (Roll)}');
ylabel("$[N]$")
xlabel("Time $[s]$")
legend('$T_{x}$ PID', '$T_{x}$ SNN', 'Location', 'best')
hold off

%%--- EJE Y ---
ax3 = subplot(2, 3, 5); % Guardamos el handle en ax3
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Extraemos la columna 2 (Y)
if PD_CONTROLLER
    plot(result.tout, result.T_control_PID(:, 2), 'LineWidth', 1.5, 'Color', '#0072BD')
end
if USE_SNN 
    plot(result.tout, result.T_control_SNN(:, 2), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', '#D95319')
end
grid on; grid minor;
title('\textbf{Torque Eje Y (Pitch)}');
ylabel("$[N]$")
xlabel("Time $[s]$")
legend('$T_{y}$ PID', '$T_{y}$ SNN', 'Location', 'best')
hold off

%%--- EJE Z ---
ax4 = subplot(2, 3, 6); % Guardamos el handle en ax4
hold on
xline(pt1, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Extraemos la columna 3 (Z)
if PD_CONTROLLER
    plot(result.tout, result.T_control_PID(:, 3), 'LineWidth', 1.5, 'Color', '#0072BD')
end
if USE_SNN
    plot(result.tout, result.T_control_SNN(:, 3), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', '#D95319')
end
grid on; grid minor;
title('\textbf{Torque Eje Z (Yaw)}');
ylabel("$[N]$")
xlabel("Time $[s]$")
legend('$T_{z}$ PID', '$T_{z}$ SNN', 'Location', 'best')
hold off

%%--- VINCULAR EJES ---
% Sincroniza el zoom y el desplazamiento solo en el eje del tiempo (x)
linkaxes([ax1, ax2, ax3, ax4], 'x');

%exportgraphics(f_ErrorTorque, 'ErrorTorque.pdf', 'ContentType', 'vector');

fprintf("Error y T_control: %.3f s\n", toc)

end