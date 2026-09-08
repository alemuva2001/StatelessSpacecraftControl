%% evaluate_simulation.m
% Calcula métricas de error entre la SNN y el PID (componente D).
% Ejecutar directamente después de la simulación, con 'result' en el workspace.

%% Extraer datos
pid = result.T_control_D;
snn = result.T_control_SNN;

T = min(size(pid, 1), size(snn, 1));
pid = pid(1:T, :);
snn = snn(1:T, :);

dt = 0.1;
t = (0:T-1)' * dt;
ejes = {'X (Roll)', 'Y (Pitch)', 'Z (Yaw)'};

err = snn - pid;

%% Métricas por eje
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf(' EVALUACIÓN SNN vs PID (Componente D)\n');
fprintf('%s\n\n', repmat('=', 1, 70));
fprintf('Duración: %.1f s (%d muestras)\n\n', T*dt, T);

fprintf('%-14s %12s %12s %12s %12s %8s\n', ...
    'Eje', 'MSE', 'RMSE', 'MAE', 'MaxErr', 'R²');
fprintf('%s\n', repmat('-', 1, 70));

total_mse = 0;
for i = 1:3
    e = err(:, i);
    mse_i   = mean(e.^2);
    rmse_i  = sqrt(mse_i);
    mae_i   = mean(abs(e));
    maxe_i  = max(abs(e));
    
    ss_res = sum(e.^2);
    ss_tot = sum((pid(:,i) - mean(pid(:,i))).^2);
    if ss_tot > 1e-20
        r2 = 1 - ss_res / ss_tot;
    else
        r2 = NaN;
    end
    
    fprintf('%-14s %12.2e %12.2e %12.2e %12.2e %8.4f\n', ...
        ejes{i}, mse_i, rmse_i, mae_i, maxe_i, r2);
    total_mse = total_mse + mse_i;
end

fprintf('\n%-14s %12.2e %12.2e\n', 'TOTAL', total_mse, sqrt(total_mse/3));

%% Correlación
fprintf('\n--- Correlación de Pearson ---\n');
for i = 1:3
    umbral = max(abs(pid(:,i))) * 0.05;
    mask = abs(pid(:,i)) > umbral;
    if sum(mask) > 10
        r = corrcoef(pid(mask, i), snn(mask, i));
        r = r(1, 2);
        if r > 0.7
            estado = 'BIEN';
        elseif r > 0.3
            estado = 'BAJO';
        else
            estado = 'MALO';
        end
        fprintf('  %s: r = %+.4f  [%s]\n', ejes{i}, r, estado);
    else
        fprintf('  %s: Señal insuficiente\n', ejes{i});
    end
end

%% Gráfica
figure('Name', 'Evaluación SNN vs PID', 'Position', [100 100 1200 700]);

for i = 1:3
    % Comparación temporal
    subplot(3, 2, 2*i - 1);
    plot(t, pid(:,i), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, snn(:,i), 'r--', 'LineWidth', 1.2);
    title(sprintf('%s — Comparación', ejes{i}));
    xlabel('Time [s]'); ylabel('Torque [Nm]');
    legend('PID', 'SNN'); grid on;
    
    % Error instantáneo
    subplot(3, 2, 2*i);
    plot(t, err(:,i), 'k-', 'LineWidth', 0.8);
    yline(0, '--', 'Color', [0.5 0.5 0.5]);
    title(sprintf('%s — Error (SNN - PID)', ejes{i}));
    xlabel('Time [s]'); ylabel('Error [Nm]');
    grid on;
end

sgtitle('Evaluación SNN vs PID (D-term)', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n%s\n', repmat('=', 1, 70));
