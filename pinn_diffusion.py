"""JAX PINN for 1D diffusion u_t = D u_xx on x in [0,1], t in [0,1].
IC u(0,x)=sin(pi x); BC u(t,0)=u(t,1)=0.  Exact: u = exp(-D pi^2 t) sin(pi x).
Compares (a) residual-only loss (original notebook), (b) full PINN loss, against the exact solution."""
import jax, jax.numpy as jnp, optax, numpy as np
from jax import grad, vmap, jit, random

D = 0.1

def init_params(layers, key):
    params = []
    for k, (i, o) in zip(random.split(key, len(layers) - 1), zip(layers[:-1], layers[1:])):
        params.append({'w': random.normal(k, (i, o)) * jnp.sqrt(1 / i), 'b': jnp.zeros(o)})
    return params

def forward(params, tx):
    a = tx
    for l in params[:-1]:
        a = jnp.tanh(a @ l['w'] + l['b'])
    return (a @ params[-1]['w'] + params[-1]['b'])[0]

def exact(t, x):
    return jnp.exp(-D * jnp.pi ** 2 * t) * jnp.sin(jnp.pi * x)

def residual(params, t, x):
    u = lambda t, x: forward(params, jnp.array([t, x]))
    return grad(u, 0)(t, x) - D * grad(grad(u, 1), 1)(t, x)

def losses(params, col, ic, bc):
    r = jnp.mean(vmap(lambda p: residual(params, p[0], p[1]) ** 2)(col))
    l_ic = jnp.mean(vmap(lambda p: (forward(params, p) - jnp.sin(jnp.pi * p[1])) ** 2)(ic))
    l_bc = jnp.mean(vmap(lambda p: forward(params, p) ** 2)(bc))
    return r, l_ic, l_bc

def total(params, col, ic, bc, use_ic_bc):
    r, a, b = losses(params, col, ic, bc)
    return r + (10. * a + 10. * b if use_ic_bc else 0.)

def run(use_ic_bc, seed=0, iters=10000):
    k = random.split(random.PRNGKey(seed), 4)
    col = random.uniform(k[0], (2000, 2))
    ic = jnp.stack([jnp.zeros(200), random.uniform(k[1], (200,))], 1)
    bc = jnp.stack([random.uniform(k[2], (200,)), jnp.repeat(jnp.array([0., 1.]), 100)], 1)
    params = init_params([2, 64, 64, 64, 1], k[3])
    opt = optax.adam(optax.cosine_decay_schedule(2e-3, iters)); st = opt.init(params)

    @jit
    def step(p, s):
        l, g = jax.value_and_grad(total)(p, col, ic, bc, use_ic_bc)
        u, s = opt.update(g, s); return optax.apply_updates(p, u), s, l
    for _ in range(iters):
        params, st, _ = step(params, st)
    T, X = jnp.meshgrid(jnp.linspace(0, 1, 101), jnp.linspace(0, 1, 101))
    g = jnp.stack([T.ravel(), X.ravel()], 1)
    pred = vmap(lambda p: forward(params, p))(g); ex = exact(g[:, 0], g[:, 1])
    rel = float(jnp.linalg.norm(pred - ex) / jnp.linalg.norm(ex))
    r, a, b = [float(v) for v in losses(params, col, ic, bc)]
    return rel, float(jnp.max(jnp.abs(pred - ex))), r, a, b, params

if __name__ == '__main__':
    for name, flag in [('residual-only (original)', False), ('residual + IC + BC', True)]:
        rels = []
        for s in range(3):
            rel, mx, r, a, b, _ = run(flag, s)
            rels.append(rel); print(f'{name} seed{s}: relL2={rel:.4f} maxabs={mx:.4f} res={r:.2e} ic={a:.2e} bc={b:.2e}', flush=True)
        print(f'>> {name}: relL2 mean={np.mean(rels):.4f} std={np.std(rels):.4f}')
