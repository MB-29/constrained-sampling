import numpy as np
import matplotlib.pyplot as plt

from constraint import L2NormConstraint
from systems.particle import Particle
from samplers import Langevin, ProjectedLangevin, SplitLangevin, PrimalDescentDualAscent

# distribution
sigma_y = .1
particle = Particle(sigma_y=sigma_y)
d = particle.d
grad_potential = particle.grad_potential

# constraint
target_squared_norm = 9.
constraint = L2NormConstraint(target_squared_norm)

# samplers
tau, eta = 1e-2, 1e-2
n_samples = 1_000
n_steps = 2_000
sigma0 = 1.
burn_in = n_steps//10
rho_values = np.zeros(n_steps)
rho_values[burn_in:] = np.logspace(-1, 1., n_steps-burn_in)

langevin = Langevin(d, grad_potential, tau,
                    constraint=constraint, sigma0=sigma0)
projected = ProjectedLangevin(d, grad_potential, constraint, tau, burn_in, sigma0=sigma0)
descent_ascent = sampler = PrimalDescentDualAscent(d, grad_potential, constraint, tau, eta, burn_in=burn_in)
augmented = SplitLangevin(d, grad_potential, constraint, tau, rho_values, eta, burn_in, sigma0=sigma0)
sampler_choice = {
    'langevin': langevin,
    'projected': projected,
    'descent_ascent': descent_ascent,
    'augmented': augmented
    }

# run and plot
for sampler_name, sampler in sampler_choice.items():

    primal_iterates, projected_iterates, dual_iterates = sampler.sample(
        n_steps=n_steps, n_samples=n_samples)
    
    primal_samples = primal_iterates[:, -1, :]
    mode_samples = primal_samples[:, 0]
    squared_norm_samples = np.sum(primal_samples**2, axis=1)

    plt.subplot(1, 2, 1)
    plt.hist(mode_samples, bins=50, alpha=.7, label=sampler_name)
    plt.xlim((-4., 4.))
    plt.title('mode')
    plt.subplot(1, 2, 2)
    plt.hist(squared_norm_samples, bins=50, alpha=.7, label=sampler_name)
    plt.axvline(target_squared_norm, ls='--', color='black')
    plt.xlim((0., 20.))
    plt.title('energy')
plt.legend()
plt.show()