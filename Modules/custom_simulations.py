from tracemalloc import start
import numpy as np
from tqdm import tqdm, trange


def simulate_particles_in_external_force_Not_HOOMD(
    ext_force,
    particles,
    n_steps,
    delta_fraction=0,
    dt=0.01,
    gamma=100,
    kT=1,
    force_args=(),
    start_position=0,
    type=np.float32,      # your output dtype
    start_positions=None
):
    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed)
    std = np.sqrt(1/dt)
    if start_positions is None:
        start_positions = np.ones(particles) * start_position
        positions = np.zeros((n_steps, particles), dtype=type)
        positions[0, :] = start_positions
    else:
        positions = np.zeros((n_steps, particles), dtype=type)
        positions[0, :] = start_positions
    for i in range(n_steps-1):
        next_positions = positions[i, :] + (dt/gamma)*(ext_force(positions[i, :], *force_args)) + np.sqrt(2*kT/gamma)*dt*rng.normal(0, std, size=particles)
        if delta_fraction != 0:
            reset_threshold = rng.uniform(0, 1, size=particles)
            particles_to_reset = reset_threshold < delta_fraction
            next_positions[particles_to_reset] = start_position # reset to start position
        positions[i+1, :] = next_positions
    return positions

