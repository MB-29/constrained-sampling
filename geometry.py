import jax.numpy as jnp

def get_sphere_projection(radius):
    def sphere_projection(x):
        norm = jnp.linalg.norm(x, axis=1).reshape((-1, 1))
        projected = (radius/norm)*x
        return projected
    return sphere_projection


def ellipse_projection(x, a, b):
    angle = jnp.arctan2(x[:, 1], x[:, 0])
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    c = a*b / jnp.sqrt(b**2*cos**2 + a**2*sin**2)
    projected = jnp.stack([c*cos, c*sin], axis=1)
    return projected

def get_linear_projection(A, b):
    A_p = jnp.linalg.pinv(A)
    def project(x):
        x_batch = x.reshape((*x.shape, 1))
        b_batch = b.reshape((1, *b.shape, 1))
        residual = b_batch - A@x_batch
        projected = x_batch + A_p@residual
        return projected.reshape(x.shape)
    return project

