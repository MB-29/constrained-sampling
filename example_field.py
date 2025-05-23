import numpy as np
import matplotlib.pyplot as plt

from constraint import L2NormConstraint
from systems.field import Field
from samplers import Langevin, ProjectedLangevin, SplitLangevin, PrimalDescentDualAscent, VarianceSplit

# distribution
nx, ny = 100, 100
nk, nq = 5, 5
field = Field(nx, ny, nk, nq)
d = field.d
grad_potential = field.grad_potential

# constraint
target_squared_norm = 15.
constraint = L2NormConstraint(target_squared_norm)

# samplers
tau, eta = 1e-3, 1e-3
n_samples = 1_000
n_steps = 10_000
sigma0 = 5.
burn_in = n_steps//10
rho_values = np.zeros(n_steps)
rho_values[burn_in:] = np.arange(burn_in, n_steps)//100

langevin = Langevin(d, grad_potential, tau,
                    constraint=constraint, sigma0=sigma0)
projected = ProjectedLangevin(d, grad_potential, constraint, tau, burn_in, sigma0=sigma0)
descent_ascent = sampler = PrimalDescentDualAscent(d, field.grad_potential, constraint, tau, eta, burn_in=burn_in)
augmented = SplitLangevin(d, grad_potential, constraint, tau, rho_values, eta, burn_in, sigma0=sigma0)
variance_split = VarianceSplit(d, grad_potential, constraint, tau, eta, burn_in, sigma0=sigma0)

sampler_choice = {
    # 'langevin': langevin,
    # 'projected': projected,
    # 'descent_ascent': descent_ascent,
    # 'augmented': augmented,
    'variance-split': variance_split
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