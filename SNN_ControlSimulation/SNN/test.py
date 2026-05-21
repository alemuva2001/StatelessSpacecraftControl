import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 30          # time steps
dt = 1.0        # time interval
Q_pos = 0.2     # process noise (position)
Q_vel = 0.05    # process noise (velocity)
R_meas = 2.0    # measurement noise variance
N = 500         # number of particles

# State initialization
x_true = np.zeros((T, 4))  # [px, py, vx, vy]
z_meas = np.zeros((T,2))
x_true[0] = [0, 0, 1, 0.5] # initial position & velocity

# Particle initialization
particles = np.random.normal(0, 2.0, (N, 4))
weights = np.ones(N)/N

# Helper: state transition function
def f(x, dt) :
    F = np.array ([[1, 0, dt, 0], 
                   [0, 1, 0,  dt],
                   [0, 0, 1,  0],
                   [0, 0, 0,  1]])
    return (F @ x.T).T

# Main filter loop
for k in range (1, T):
    # True motion
    eta_k = np.random.multivariate_normal(
        np.zeros(4),
        np.diag([Q_pos, Q_pos, Q_vel, Q_vel])
        )
    x_true [k] = f(x_true[k-1], dt) + eta_k

    # Measurement
    v_k = np.random.multivariate_normal(
        np.zeros(2),
        R_meas * np.eye(2)
        )
    z_meas[k] = x_true[k, :2] + v_k

    # Prediction
    noise = np.random.multivariate_normal(
        np.zeros (4),
        np.diag([Q_pos, Q_pos, Q_vel, Q_vel]),
        N
        )
    particles = f(particles, dt) + noise

    # Weighting
    diff = z_meas [k] - particles [:, :2]
    likelihood = np.exp(-0.5 * np.sum(diff**2, axis=1)/R_meas)
    weights *= likelihood
    weights += 1.e-300
    weights /= np.sum(weights)

    # Compute effective sample size
    Neff = 1./np.sum(weights**2)

    # Resampling systematic
    if Neff < N/2:
        positions = (np.arange(N) + np.random.rand())/N
        cumulative_sum = np.cumsum(weights)
        indexes = np.searchsorted(cumulative_sum, positions)
        particles = particles[indexes]
        weights.fill (1.0/N)

# -------------------------
# Visualization
# -------------------------
plt. figure(figsize = (6 ,6) )
plt. scatter(particles[: ,0], particles[: ,1], s=5, color='blue', alpha=0.3, label='Particles')
plt.plot(x_true[: ,0], x_true[: ,1], 'k-', linewidth=2, label='True trajectory')
plt.scatter(z_meas[: ,0], z_meas[: ,1], c='r', marker='x', label='Measurements')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('2 D Particle Filter Tracking Example')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()