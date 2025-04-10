import numpy as np

def power_function(distance):
    return .1/(1+distance)**2

epsilon = .7
b, c = -epsilon, -(1 + epsilon)
shift = -1.
def mode_potential(mode):
    return .5*((1/4)*(mode-shift)**4 + (b/3)*(mode-shift)**3 + (c/2)*(mode-shift)**2 )
def mode_potential_gradient(mode):
    return .5*((mode-shift)**3 + b*(mode-shift)**2 + c*(mode-shift) - .0)
    
class Field:

    def __init__(self, nx, ny, nk, nq):
        self.nx = nx
        self.ny = ny
        self.nk = nk
        self.nq = nq

        self.d = 2*nk*nq
        self.build_parameters()
    
    def build_parameters(self):
        self.inv_variances = np.zeros(self.d)
        for i in range(self.d//2):
            q_i = i//self.nq
            k_i = i % self.nk
            wave = np.array([k_i, q_i])
            power = power_function(np.linalg.norm(wave))
            self.inv_variances[2*i] = 1/power
            self.inv_variances[2*i+1] = 1/power
        self.variances = 1/self.inv_variances
        self.inv_cov = np.diag(self.inv_variances)
        self.cov = np.diag(self.variances)
        self.mean = np.zeros(self.d)
    
    def grad_potential(self, x, step_index=0):
        mean, fluctuations = x[:, :2], x[:, 2:]
        gradient_mode = mode_potential_gradient(mean).reshape((-1, 2))
        gradient_fluctuations = fluctuations@self.inv_cov[2:, 2:].T
        gradient = np.concatenate((gradient_mode, gradient_fluctuations), axis=1)
        return gradient
    
    def potential(self, x, step_index=0):
        mean, fluctuations = x[:, :2], x[:, 2:]
        potential_mode = mode_potential(mean).sum(axis=1)
        potential_fluctuations = np.sum(fluctuations**2@self.inv_cov[2:, 2:].T, axis=1)
        potential = potential_mode + potential_fluctuations
        return potential

    def build_wavevector_matrix(self):
        self.wave_vectors = np.zeros((self.d//2, 2))
        for i in range(self.d//2):
            q_i = i//self.nq
            k_i = i % self.nk
            vector = np.array([k_i, q_i])
            self.wave_vectors[i] = vector
        return self.wave_vectors
