import numpy as np

epsilon = .7
b, c = -epsilon, -(1 + epsilon)
shift = -1.

def mode_potential(mode):
    return .5*((1/4)*(mode-shift)**4 + (b/3)*(mode-shift)**3 + (c/2)*(mode-shift)**2 )
def mode_potential_gradient(mode):
    return .5*((mode-shift)**3 + b*(mode-shift)**2 + c*(mode-shift) - .0)

class Particle:

    def __init__(self, sigma_y):
        self.d = 2
        self.sigma_y = sigma_y

    def grad_potential(self, position, step_index=0):
        x, y = position[:, :1], position[:, 1:]
        gradient_x = mode_potential_gradient(x)
        gradient_y = y/self.sigma_y**2
        gradient = np.concatenate((gradient_x, gradient_y), axis=1)
        return gradient
    
    def potential(self, position, step_index=0):
        x, y = position[:, :1], position[:, 1:]
        potential_x = mode_potential(x)
        potential_y = y**2/self.sigma_y**2
        potential = potential_x + potential_y
        return potential
