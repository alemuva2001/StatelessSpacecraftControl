function [torque, telemetry] = snn_controller_P(event_matrix, USE_SNN)
    % Python calls declared as explicit to avoid C compilation
    coder.extrinsic('py.importlib.import_module');
    coder.extrinsic('py.importlib.reload');
    coder.extrinsic('py.proportional_inference.initialize');
    coder.extrinsic('py.proportional_inference.step');
    coder.extrinsic('cell2mat');
    coder.extrinsic('cell');
    coder.extrinsic('double');

    if nargin < 2
        USE_SNN = 1;
    end

    % Output initialization
    torque = zeros(3, 1);
    telemetry = zeros(4, 1);
    if USE_SNN == 0
        return;
    end

    % Initialize only the first time
    persistent initialized
    if isempty(initialized)
        mod = py.importlib.import_module('proportional_inference');
        py.importlib.reload(mod);
        py.proportional_inference.initialize();
        initialized = true;
    end

    % Pasamos event_matrix directamente a Python (usa memoryview y evita el fallo de libmwbuffer con NumPy 2.x)
    py_out = py.proportional_inference.step(event_matrix);

    % =====================================================================
    % CONVERSIÓN NATIVA: Simulink compilará esto sin errores.
    % Convertimos la lista de Python a Cell, y el Cell a array de MATLAB.
    % =====================================================================
    out_array = double(cell2mat(cell(py_out)));

    % Repartimos los primeros 3 valores al torque proporcional
    torque(1) = out_array(1);
    torque(2) = out_array(2);
    torque(3) = out_array(3);
end



