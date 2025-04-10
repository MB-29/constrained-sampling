import numpy as np

from geometry import get_linear_projection, get_sphere_projection


def get_squared_norm_residual(target_squared_norm):

    def squared_norm_residual(x):
        squared_norm = np.sum(x**2, axis=1).reshape((-1, 1))
        residual = squared_norm - target_squared_norm
        return residual
    return squared_norm_residual

class Constraint:

    def __init__(self, project, m, equality_constraint=None, grad_equality_constraint=None):
        self.project = project if project is not None else lambda x :x
        self.m = m
        self.equality_constraint = equality_constraint
        self.grad_equality_constraint = grad_equality_constraint
class LinearConstraint(Constraint):

    def __init__(self, A, b, equality_constraint=None, grad_equality_constraint=None):
        m, d = A.shape

        project = get_linear_projection(A, b)
        super().__init__(project, m, equality_constraint, grad_equality_constraint)

class L2NormConstraint(Constraint):

    def __init__(self, target_squared_norm):

        equality_constraint = get_squared_norm_residual(target_squared_norm)
        grad_equality_constraint = lambda x : x
        radius = np.sqrt(target_squared_norm)
        project = get_sphere_projection(radius)
        m = 1

        super().__init__(project, m, equality_constraint, grad_equality_constraint)




