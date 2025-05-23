import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

class Langevin:

    def __init__(self, d, grad_potential, tau, get_step=None, sigma0=None):
        self.tau = tau
        self.d = d
        self.m = 1
        self.grad_potential = grad_potential
        self.sigma0 = 1 if sigma0 is None else sigma0

        self.get_step = get_step if get_step is not None else lambda t: tau
        self.project = lambda x: x


    def sample(self, key, n_steps, n_samples):
        x = self.sigma0*jax.random.normal(key, shape=(n_samples, self.d))
        # x = self.sigma0*jnp.ones((n_samples, self.d))
        z = self.project(x)
        lambd = jnp.zeros((n_samples, self.m))
        key, subkey = jax.random.split(key)  # Split the JAX PRNG key.
        step_values = jnp.arange(n_steps - 1, -1, -1)

        first_iterate = (x, z, lambd)
        first_carry = (first_iterate, 0, subkey)

        final_iterate, iterate_values = jax.lax.scan(
            self.step, first_carry, step_values)

        return iterate_values
    
    def step(self, carry, step_index):

        iterate, step_index, key = carry
        x, z, lambd = iterate
        key, subkey = jax.random.split(key)


        tau = self.get_step(step_index)
        gradient = self.grad_potential(x, step_index)
        noise = jax.random.normal(key, x.shape)

        x_ = x - tau*gradient + jnp.sqrt(2*tau)*noise

        z_ = x_.copy()
        lambd_ = lambd

        iterate_ = (x_, z_, lambd_)
        step_index_ = step_index + 1
        carry_ = (iterate_, step_index_, subkey)
        return carry_, iterate_
    
    def __repr__(self):
        return "langevin"
    
    
class ProjectedLangevin(Langevin):
    def __init__(self, d, grad_potential, constraint, tau, burn_in, get_step=None, sigma0=None):
        super().__init__(d, grad_potential, tau, get_step=get_step, sigma0=sigma0)
        self.project = constraint.project
        self.burn_in = burn_in
        
    def __repr__(self):
        return "projected"
    
    def step(self, carry, step_index):

        iterate, step_index, key = carry
        x, z, lambd = iterate
        key, subkey = jax.random.split(key)

        tau = self.get_step(step_index)
        gradient = self.grad_potential(x, step_index)
        noise = jax.random.normal(key, x.shape)

        x_ = x - tau*gradient + jnp.sqrt(2*tau)*noise
        z_ = self.project(x_)
        x_ = z_.copy()
        lambd_ = lambd

        iterate_ = (x_, z_, lambd_)
        step_index_ = step_index + 1
        carry_ = (iterate_, step_index_, subkey)
        return carry_, iterate_



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
        super().__init__(d, grad_potential, tau, sigma0=sigma0)
        self.equality_constraint = constraint.equality_constraint
        self.grad_equality_constraint = constraint.grad_equality_constraint
        self.eta = eta
        self.rho = rho
        self.burn_in = burn_in
        self.project = lambda x: x

    def __repr__(self):
        return "descent-ascent"

    def step(self, carry, step_index):

        iterate, step_index, key = carry
        x, z, lambd = iterate
        key, subkey = jax.random.split(key)

        tau = self.get_step(step_index)
        potential_gradient = self.grad_potential(x, step_index)
        noise = jax.random.normal(key, x.shape)
        constraint_grad = self.grad_equality_constraint(x)
        constraint = self.equality_constraint(x)
        gradient = potential_gradient + lambd * constraint_grad

        x_ = x - tau*gradient + jnp.sqrt(2*tau)*noise
        lambd_ = lambd + self.eta*constraint
        z_ = x_.copy()

        iterate_ = (x_, z_, lambd_)
        step_index_ = step_index + 1
        carry_ = (iterate_, step_index_, subkey)
        return carry_, iterate_

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
        super().__init__(d, grad_potential, tau, sigma0=sigma0)
        self.m = self.d
        self.project = constraint.project
        self.rho_values = rho_values
        self.eta = eta
        self.burn_in = burn_in
        self.sigma = 1.

    def __repr__(self):
        return "split-langevin"
    
    # def sample(self, key, n_steps, n_samples, rho_values):
    #     self.rho_values = rho_values
    #     super().sample(key, n_steps, n_samples)

    def step(self, carry, step_index):

        iterate, step_index, key = carry
        x, z, lambd = iterate
        key, subkey = jax.random.split(key)

        rho = self.rho_values[step_index]
        tau = self.get_step(step_index)
        potential_gradient = self.grad_potential(x, step_index)
        noise = jax.random.normal(key, x.shape)
        augmentation = rho*(x - z + lambd)
        gradient = potential_gradient + augmentation

        x_ = x - tau*gradient + jnp.sqrt(2*tau)*noise
        z_ = self.project(x_ + lambd)
        lambd_ = lambd + self.eta*(x_ - z_)

        iterate_ = (x_, z_, lambd_)
        step_index_ = step_index + 1
        carry_ = (iterate_, step_index_, subkey)
        return carry_, iterate_
