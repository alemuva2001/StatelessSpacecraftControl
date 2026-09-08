"""
evaluate_simulation.py
Calcula métricas de error entre la salida de la SNN y el PID (componente D)
a partir de los datos guardados en el workspace de MATLAB.

Uso desde MATLAB (después de ejecutar la simulación):
    save('sim_results.mat', 'result');
    
Uso desde terminal:
    python evaluate_simulation.py sim_results.mat
    
O directamente desde MATLAB:
    py.evaluate_simulation.evaluate('sim_results.mat')
"""

import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def evaluate(mat_path: str = 'sim_results.mat'):
    """Carga result.T_control_D y result.T_control_SNN y calcula métricas."""
    
    print(f"\n{'='*60}")
    print(f" EVALUACIÓN SNN vs PID (Componente D)")
    print(f"{'='*60}\n")
    
    # Cargar datos
    data = scipy.io.loadmat(mat_path, squeeze_me=True)
    
    # Intentar extraer las señales del struct 'result'
    if 'result' in data:
        result = data['result']
        # scipy.io carga structs como numpy void arrays
        pid = result['T_control_D'].item()
        snn = result['T_control_SNN'].item()
    elif 'T_control_D' in data and 'T_control_SNN' in data:
        pid = data['T_control_D']
        snn = data['T_control_SNN']
    else:
        print(f"Variables disponibles: {[k for k in data.keys() if not k.startswith('_')]}")
        print("Error: No se encontraron T_control_D ni T_control_SNN.")
        return
    
    # Asegurar que sean (T, 3)
    if pid.ndim == 1:
        pid = pid.reshape(-1, 1)
    if snn.ndim == 1:
        snn = snn.reshape(-1, 1)
    
    T = min(pid.shape[0], snn.shape[0])
    pid = pid[:T].astype(np.float64)
    snn = snn[:T].astype(np.float64)
    n_axes = pid.shape[1]
    
    dt = 0.1  # Timestep de la simulación
    t = np.arange(T) * dt
    
    axis_names = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)'][:n_axes]
    
    # =====================================================================
    # MÉTRICAS POR EJE
    # =====================================================================
    print(f"Duración: {T*dt:.1f}s ({T} muestras)\n")
    
    errors = snn - pid
    
    header = f"{'Eje':<12} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'MaxErr':>12} {'R²':>8}"
    print(header)
    print('-' * len(header))
    
    total_mse = 0.0
    for i, name in enumerate(axis_names):
        err_i = errors[:, i]
        mse  = np.mean(err_i ** 2)
        rmse = np.sqrt(mse)
        mae  = np.mean(np.abs(err_i))
        max_err = np.max(np.abs(err_i))
        
        # R² (coeficiente de determinación)
        ss_res = np.sum(err_i ** 2)
        ss_tot = np.sum((pid[:, i] - np.mean(pid[:, i])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else float('nan')
        
        print(f"{name:<12} {mse:>12.2e} {rmse:>12.2e} {mae:>12.2e} {max_err:>12.2e} {r2:>8.4f}")
        total_mse += mse
    
    print(f"\n{'TOTAL':<12} {total_mse:>12.2e} {np.sqrt(total_mse/n_axes):>12.2e}")
    
    # =====================================================================
    # MÉTRICAS GLOBALES
    # =====================================================================
    print(f"\n--- Análisis de Correlación ---")
    for i, name in enumerate(axis_names):
        # Solo calcular correlación donde hay señal significativa
        mask = np.abs(pid[:, i]) > np.max(np.abs(pid[:, i])) * 0.05
        if np.sum(mask) > 10:
            corr = np.corrcoef(pid[mask, i], snn[mask, i])[0, 1]
            print(f"  {name}: Pearson r = {corr:+.4f}  ({'✅ Bien' if corr > 0.7 else '⚠️ Bajo' if corr > 0.3 else '❌ Malo'})")
        else:
            print(f"  {name}: Señal insuficiente para correlación")
    
    # =====================================================================
    # GRÁFICA COMPARATIVA
    # =====================================================================
    fig, axes = plt.subplots(n_axes, 2, figsize=(14, 3.5 * n_axes))
    if n_axes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Evaluación SNN vs PID (D-term)', fontsize=14, fontweight='bold')
    
    for i, name in enumerate(axis_names):
        # Columna izquierda: Comparación temporal
        axes[i, 0].plot(t, pid[:, i], 'b-', linewidth=1.5, label='PID', alpha=0.8)
        axes[i, 0].plot(t, snn[:, i], 'r--', linewidth=1.2, label='SNN', alpha=0.8)
        axes[i, 0].set_title(f'{name} — Comparación Temporal')
        axes[i, 0].set_xlabel('Time [s]')
        axes[i, 0].set_ylabel('Torque [Nm]')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Columna derecha: Error instantáneo
        axes[i, 1].plot(t, errors[:, i], 'k-', linewidth=0.8, alpha=0.7)
        axes[i, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[i, 1].set_title(f'{name} — Error (SNN - PID)')
        axes[i, 1].set_xlabel('Time [s]')
        axes[i, 1].set_ylabel('Error [Nm]')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = 'simulation_evaluation.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nGráfica guardada en: {out_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'sim_results.mat'
    evaluate(path)
