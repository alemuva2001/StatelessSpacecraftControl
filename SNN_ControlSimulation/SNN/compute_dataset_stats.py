"""
compute_dataset_stats.py
Calcula y guarda las estadísticas de normalización del dataset.
Ejecutar UNA VEZ antes de entrenar, o cada vez que se regenere el dataset.

Genera: ../Data/dataset_stats.npz
"""

import os
import glob
import numpy as np
import scipy.io

DATA_DIR = '../Data'

def compute_stats():
    all_t = []

    state_files = sorted(glob.glob(os.path.join(DATA_DIR, 'ep_*_states.mat')))
    if not state_files:
        raise FileNotFoundError(f"No se encontraron archivos en '{DATA_DIR}'")

    print(f"Analizando {len(state_files)} episodios...")

    for path in state_files:
        d = scipy.io.loadmat(path)
        all_t.append(d['torque_pid'].astype(np.float32))

    t = np.concatenate(all_t, axis=0)   # (N_total, 3)

    # 1. Calculamos la media por eje para centrar perfectamente en 0
    torque_mean = t.mean(axis=0)
    t_centered = t - torque_mean
    
    # 2. Calculamos el pico máximo global de todos los episodios (MaxAbsScaler)
    torque_peak = np.max(np.abs(t_centered), axis=0)  # shape (3,)
    stats = {
        "torque_mean" : torque_mean,
        "torque_std"  : t.std(axis=0),
        "torque_peak" : torque_peak.astype(np.float32),
        }
    # Protección contra división por cero en std
    stats['torque_std'] = np.where(stats['torque_std'] < 1e-8, 1.0, stats['torque_std'])
    
    out_path = os.path.join(DATA_DIR, 'dataset_stats.npz')
    np.savez(out_path, **stats)

    print(f"\nEstadísticas calculadas y guardadas en: {out_path}")
    print(f"  torque_mean = {stats['torque_mean']}")
    print(f"  torque_std  = {stats['torque_std']}")
    print(f"  torque_peak = {stats['torque_peak']}  <-- ¡Valor de escalado máximo!")
    print(f"\nEjecutar supervised_train.py para entrenar con normalización.")

if __name__ == '__main__':
    compute_stats()