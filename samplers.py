import numpy as np
from tqdm import tqdm

class Langevin:

    def __init__(self, d, grad_potential, tau, constraint=None, sigma0=None):
        self.tau = tau
        self.d = d
        self.m = constraint.m
        self.grad_potential = grad_potential
        self.sigma0 = 1 if sigma0 is None else sigma0

        self.project = constraint.project if constraint is not None else lambda x: x

    def __repr__(self):
        return "langevin"
    def step(self, x, z, lambd, step_index, n_samples):
            gradient = self.grad_potential(x, step_index)

            noise = np.sqrt(2*self.tau)*np.random.randn(n_samples, self.d)
            delta_x = -self.tau * gradient + noise
            proposal = x + delta_x

            x_ = proposal
            x = x_
            z = x_.copy()
            lambd = np.zeros(self.m)

            return x, z, lambd
    def sample(self, n_steps, n_samples=1):
        primal_iterates = np.zeros((n_samples, n_steps, self.d))
        projected_iterates = np.zeros((n_samples, n_steps, self.d))
        dual_iterates = np.zeros((n_samples, n_steps, self.m))

        x = self.sigma0*np.random.randn(n_samples, self.d)
        lambd = np.zeros((n_samples, self.m))
        z = self.project(x)

        for step_index in tqdm(range(n_steps)):
            primal_iterates[:, step_index, :] = x.copy()
            projected_iterates[:, step_index, :] = z.copy()
            dual_iterates[:, step_index, :] = lambd.copy()

            x, z, lambd = self.step(x, z, lambd, step_index, n_samples)

        return primal_iterates, projected_iterates, dual_iterates
    
class ProjectedLangevin(Langevin):
    def __init__(self, d, grad_potential, constraint, tau, burn_in, sigma0=None):
        super().__init__(d, grad_potential, tau, constraint=constraint, sigma0=sigma0)
        self.project = constraint.project
        self.burn_in = burn_in
        
    def __repr__(self):
        return "projected"

    def step(self, x, z, lambd, step_index, n_samples):
        gradient = self.grad_potential(x, step_index)
        noise = np.sqrt(2*self.tau)*np.random.randn(n_samples, self.d)
        delta_x = - self.tau * gradient + noise
        proposal = x + delta_x
        x_ = proposal
        x = x_
        z = self.project(x_)
        lambd = np.zeros((n_samples, self.m))

        if step_index > self.burn_in:
            x = z.copy()
            # print(f'project')
        return x, z, lambd


class PrimalDescentDualAscent(Langevin):

    def __init__(
            self,
            d,
            grad_potential,
            constraint,
            tau,
            eta,
            burn_in,
            rho=0.,
            sigma0=None
            ):
        super().__init__(d, grad_potential, tau, constraint=constraint, sigma0=sigma0)
        self.equality_constraint = constraint.equality_constraint
        self.grad_equality_constraint = constraint.grad_equality_constraint
        self.eta = eta
        self.rho = rho
        self.burn_in = burn_in

    def __repr__(self):
        return "primal-dual"

    def step(self, x, z, lambd, step_index, n_samples):

        constraint_grad = self.grad_equality_constraint(x)
        constraint = self.equality_constraint(x)
        potential_grad = self.grad_potential(x) + self.rho*constraint*constraint_grad
        gradient = potential_grad + lambd * constraint_grad
        noise = np.sqrt(2*self.tau)*np.random.randn(n_samples, self.d)

        x += -self.tau * gradient + noise
        lambd += self.eta*constraint

        return x, z, lambd


class SplitLangevin(Langevin):

    def __init__(
            self,
            d,
            grad_potential,
            constraint,
            tau,
            rho_values,
            eta,
            burn_in,
            sigma0=None
    ):
        super().__init__(d, grad_potential, tau, constraint, sigma0=sigma0)
        self.m = self.d
        self.project = constraint.project
        self.rho_values = rho_values
        self.eta = eta
        self.burn_in = burn_in
        self.sigma = 1.

    def __repr__(self):
        return "split-langevin"
    def step(self, x, z, lambd, step_index, n_samples):
        self.rho = self.rho_values[step_index]
        grad_potential = self.grad_potential(x) 
        squared_distance = np.sum((x - z + lambd)**2, axis=1).reshape((-1, 1))
        grad_augmented_potential = grad_potential + self.rho*squared_distance*(x - z + lambd)
        noise = self.sigma*np.sqrt(2*self.tau)*np.random.randn(n_samples, self.d)

        x += -self.tau * grad_augmented_potential + noise
        z = self.project(x + lambd)
        lambd += self.eta*(x - z)
        return x, z, lambd

    def lagrangian(self, x, z, lambd, rho):
        potential = self.potential(x) 
        constraint = (x - z) @ lambd
        augmentation = rho * np.linalg.sum(constraint**2, axis=1)
        return potential + constraint + augmentation
