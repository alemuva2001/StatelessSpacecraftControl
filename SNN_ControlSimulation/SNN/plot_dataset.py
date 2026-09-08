import argparse
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import numpy as np

class DatasetViewer:
    def __init__(self, data_dir='../Data'):
        self.data_dir = data_dir
        
        # Encontrar todos los episodios disponibles
        self.episodes = []
        for f in os.listdir(data_dir):
            if f.startswith('ep_') and f.endswith('_states.mat'):
                ep_num = int(f.split('_')[1])
                self.episodes.append(ep_num)
        self.episodes.sort()
        
        if not self.episodes:
            print(f"No se encontraron episodios en {data_dir}")
            return
            
        self.current_idx = 0
        
        # Configurar la figura
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.25) # Espacio para el slider
        
        # Crear los elementos interactivos
        axfreq = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(
            ax=axfreq,
            label='Episodio',
            valmin=self.episodes[0],
            valmax=self.episodes[-1],
            valinit=self.episodes[0],
            valstep=1
        )
        self.slider.on_changed(self.update)
        
        axprev = plt.axes([0.1, 0.1, 0.05, 0.03])
        self.bprev = Button(axprev, '<')
        self.bprev.on_clicked(self.prev)
        
        axnext = plt.axes([0.85, 0.1, 0.05, 0.03])
        self.bnext = Button(axnext, '>')
        self.bnext.on_clicked(self.next)

        self.lines = [None, None, None]
        self.plot_current()
        plt.show()

    def load_episode(self, ep_num):
        filepath = os.path.join(self.data_dir, f'ep_{ep_num:04d}_states.mat')
        if not os.path.exists(filepath):
            return None
        data = scipy.io.loadmat(filepath)
        if 'torque_pid' not in data:
            return None
        return data['torque_pid']

    def plot_current(self):
        ep_num = self.episodes[self.current_idx]
        torque = self.load_episode(ep_num)
        
        if torque is None:
            self.fig.suptitle(f'Episodio {ep_num} no encontrado o inválido', fontsize=14, color='red')
            for ax in self.axes:
                ax.clear()
            self.fig.canvas.draw_idle()
            return

        t = np.arange(torque.shape[0]) * 0.1
        titles = ['Torque Eje X (Roll)', 'Torque Eje Y (Pitch)', 'Torque Eje Z (Yaw)']
        
        self.fig.suptitle(f'Torque PID - Dataset (Episodio {ep_num})', fontsize=14, fontweight='bold')
        
        for i in range(3):
            if self.lines[i] is None:
                self.lines[i], = self.axes[i].plot(t, torque[:, i], color='#1f77b4', linewidth=1.5)
                self.axes[i].set_title(titles[i])
                self.axes[i].set_xlabel('Time [s]')
                self.axes[i].set_ylabel('Torque [Nm]')
                self.axes[i].grid(True, linestyle='--', alpha=0.7)
            else:
                self.lines[i].set_ydata(torque[:, i])
                self.lines[i].set_xdata(t)
            
            # Ajustar los límites del eje dinámicamente
            self.axes[i].relim()
            self.axes[i].autoscale_view()
            
        self.fig.canvas.draw_idle()

    def update(self, val):
        ep_num = int(self.slider.val)
        if ep_num in self.episodes:
            self.current_idx = self.episodes.index(ep_num)
        else:
            # Encuentra el más cercano
            self.current_idx = np.argmin(np.abs(np.array(self.episodes) - ep_num))
            self.slider.set_val(self.episodes[self.current_idx])
            return # el set_val volverá a llamar a update
        self.plot_current()

    def next(self, event):
        if self.current_idx < len(self.episodes) - 1:
            self.current_idx += 1
            self.slider.set_val(self.episodes[self.current_idx])

    def prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.slider.set_val(self.episodes[self.current_idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visor interactivo del dataset.')
    parser.add_argument('-d', '--dir', type=str, default='../Data', help='Directorio de datos (por defecto: ../Data)')
    args = parser.parse_args()
    
    viewer = DatasetViewer(args.dir)

