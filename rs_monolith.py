
import jax
import jax.numpy as jnp
import argparse
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ---------- Safe helpers ----------
def safe_exp(x, low=-700.0, high=700.0):
    return jnp.exp(jnp.clip(x, low, high))

def safe_div(n, d, eps=1e-300):
    return n / (d + eps)

# ---------- Parameters ----------
@dataclass(frozen=True)
class Params:
    # Geometry / model parameters
    k: float = 1.0          # AdS curvature scale
    L: float = 1.0          # (not used explicitly, kept for extensibility)
    rc: float = 12.0        # dimensionless "radius" (Ymax = pi * rc)
    vUV: float = 0.1        # UV boundary value of scalar
    vIR: float = 0.01       # IR target value of scalar

    # Grid / numerics
    Ny: int = 3001
    stretch: float = 0.35
    alpha: float = 4.0
    rtol: float = 1e-12
    atol: float = 1e-14
    max_substeps: int = 800
    clip: float = 1e12

    # Gravity coupling
    kappa5_sq: float = 1.0  # 5D gravitational coupling

# ---------- Grid ----------
def make_stretched_grid(p: Params):
    """
    Stretched grid y in [0, Ymax], with Ymax = pi * rc.
    """
    Ymax = jnp.pi * jnp.array(p.rc, dtype=jnp.float64)
    s = jnp.array(p.stretch, dtype=jnp.float64)
    alpha = jnp.array(p.alpha, dtype=jnp.float64)
    xi = jnp.linspace(0.0, 1.0, p.Ny)
    f = ((1.0 - s) * xi + s * (xi ** alpha)) / ((1.0 - s) + s)
    y = Ymax * f
    return y, Ymax

# ---------- Superpotential & RHS ----------
def W0(p: Params):
    # W0 chosen so that A' = k in the pure RS limit
    return jnp.array(3.0 * p.k / p.kappa5_sq, dtype=jnp.float64)

def rhs_system_core(p: Params, U, c2, W0_const):
    """
    First-order system from superpotential:
      phi' = dW/dphi = 2 c2 phi
      A'   = (kappa5^2 / 3) W(phi) = (kappa5^2 / 3) (W0 + c2 phi^2)
    """
    phi, A = U
    dphi = 2.0 * c2 * phi
    dA   = (p.kappa5_sq / 3.0) * (W0_const + c2 * (phi**2))
    return jnp.array([dphi, dA])

# ---------- Integrator (RK45, JAX-pure) ----------
def integrate_first_order_with_fun(fun, p: Params, y, U0, Ymax, c2):
    Ny = y.shape[0]
    rtol = jnp.array(p.rtol, dtype=jnp.float64)
    atol = jnp.array(p.atol, dtype=jnp.float64)
    max_substeps = p.max_substeps
    clip_val = jnp.array(p.clip, dtype=jnp.float64)

    def one_interval(U, i):
        yi = y[i]
        yi1 = y[i+1]
        h_total = yi1 - yi
        state0 = (U, h_total, h_total, jnp.array(0))

        def cond_fn(state):
            Uc, h_try, h_rem, sub = state
            return jnp.logical_and(h_rem > 1e-18, sub < max_substeps)

        def body_fn(state):
            Uc, h_try, h_rem, sub = state
            h_use = jnp.minimum(h_try, h_rem)
            y_local = yi + (h_total - h_rem)

            k1 = fun(p, y_local, Uc, Ymax, c2)
            k2 = fun(p, y_local + 0.25*h_use,
                     Uc + h_use*(0.25*k1), Ymax, c2)
            k3 = fun(p, y_local + 3.0/8.0*h_use,
                     Uc + h_use*(3.0/32.0*k1 + 9.0/32.0*k2), Ymax, c2)
            k4 = fun(p, y_local + 12.0/13.0*h_use,
                     Uc + h_use*(1932.0/2197.0*k1 - 7200.0/2197.0*k2 + 7296.0/2197.0*k3),
                     Ymax, c2)
            k5 = fun(p, y_local + h_use,
                     Uc + h_use*(439.0/216.0*k1 - 8.0*k2 + 3680.0/513.0*k3 - 845.0/4104.0*k4),
                     Ymax, c2)
            k6 = fun(p, y_local + 0.5*h_use,
                     Uc + h_use*(-8.0/27.0*k1 + 2.0*k2 - 3544.0/2565.0*k3
                                 + 1859.0/4104.0*k4 - 11.0/40.0*k5),
                     Ymax, c2)

            U5 = Uc + h_use*(16.0/135.0*k1 + 6656.0/12825.0*k3
                             + 28561.0/56430.0*k4 - 9.0/50.0*k5 + 2.0/55.0*k6)
            U4 = Uc + h_use*(25.0/216.0*k1 + 1408.0/2565.0*k3
                             + 2197.0/4104.0*k4 - 1.0/5.0*k5)

            # Componentwise error control
            scale = atol + rtol * jnp.maximum(jnp.abs(Uc), jnp.abs(U5))
            err_comp = jnp.abs(U5 - U4) / scale
            err = jnp.max(err_comp)

            accept = err <= 1.0

            Uc_new = jnp.where(accept, U5, Uc)
            Uc_new = jnp.clip(Uc_new, -clip_val, clip_val)
            h_rem_new = jnp.where(accept, h_rem - h_use, h_rem)
            # Simple step adaptation
            factor = jnp.where(err > 0.0, 0.9 * err**(-0.2), 2.0)
            h_try_new = jnp.where(accept, h_use * jnp.clip(factor, 0.2, 5.0), h_use * 0.5)

            return (Uc_new, h_try_new, h_rem_new, sub+1)

        U_final, _, _, sub_final = jax.lax.while_loop(cond_fn, body_fn, state0)
        return U_final, (U_final, sub_final)

    carry, ys = jax.lax.scan(one_interval, U0, jnp.arange(Ny-1))
    states, sub_counts = ys
    Y = jnp.vstack([U0, states])
    return Y, sub_counts

integrate_first_order_with_fun = jax.jit(
    integrate_first_order_with_fun,
    static_argnums=(0, 1),  # fun, p
)

# ---------- Finite differences ----------
def finite_diff_first(y, f):
    dy = jnp.diff(y)
    df_center = (f[2:] - f[:-2]) / (y[2:] - y[:-2])
    df0 = (f[1] - f[0]) / dy[0]
    dfN = (f[-1] - f[-2]) / dy[-1]
    df = jnp.concatenate([df0[None], df_center, dfN[None]])
    return df

def finite_diff_second(y, f):
    d2f_center = ( (f[2:] - f[1:-1]) / (y[2:] - y[1:-1])
                 - (f[1:-1] - f[:-2]) / (y[1:-1] - y[:-2]) ) / (0.5*(y[2:] - y[:-2]))
    d2f0 = (f[2] - 2.0*f[1] + f[0]) / ((y[1]-y[0])**2)
    d2fN = (f[-1] - 2.0*f[-2] + f[-3]) / ((y[-1]-y[-2])**2)
    d2f = jnp.concatenate([d2f0[None], d2f_center, d2fN[None]])
    return d2f

# ---------- Superpotential, constraints ----------
def superpotential_W(p: Params, phi, c2):
    # W(φ) = W0 + c2 φ^2
    return W0(p) + c2 * (phi**2)

def einstein_constraint_superpotential(phi, A, y, p: Params, c2):
    """
    Superpotential consistency constraints:
      C1 = φ' - dW/dφ
      C2 = A' - (κ^2/3) W(φ)
    Jika C1 ≈ 0 dan C2 ≈ 0, solusi konsisten dengan sistem superpotential.
    """
    phi_p = finite_diff_first(y, phi)
    A_p   = finite_diff_first(y, A)

    dW_dphi = 2.0 * c2 * phi
    W_val   = superpotential_W(p, phi, c2)

    C1 = phi_p - dW_dphi
    C2 = A_p   - (p.kappa5_sq / 3.0) * W_val

    return C1, C2, phi_p, A_p, W_val

def ricci_scalar_from_A(A, y):
    """
    Ricci scalar for metric ds^2 = e^{-2A} η + dy^2:
      R = -20 (A')^2 - 8 A''
    """
    A_p  = finite_diff_first(y, A)
    A_pp = finite_diff_second(y, A)
    R = -20.0*(A_p**2) - 8.0*A_pp
    return R

def nec_audit(phi_p):
    """
    Null energy condition along the extra dimension:
      ρ + p_y = φ'^2 >= 0
    """
    nec_val = phi_p**2
    min_nec = jnp.min(nec_val)
    return {
        "min_nec": min_nec,
        "all_nonnegative": (min_nec >= 0.0),
    }

# ---------- Hierarchy & audits ----------
def hierarchy_from_A(A: jnp.ndarray):
    logV = -4.0 * A
    V_eff = safe_exp(logV)
    R_eff = jnp.power(V_eff, 0.25)
    A_eff = jnp.power(R_eff, 3.0)
    dA = jnp.diff(A)
    eps_local = safe_exp(-4.0 * dA)

    def _mean_nonempty(x):
        return jnp.mean(x)

    eps_mean = jax.lax.cond(
        jnp.size(eps_local) > 0,
        lambda x: _mean_nonempty(x),
        lambda x: jnp.nan,
        eps_local,
    )
    return {
        "logV": logV,
        "V_eff": V_eff,
        "R_eff": R_eff,
        "A_eff": A_eff,
        "eps_local": eps_local,
        "eps_mean": eps_mean,
    }

def audit_volume_ratio_pointwise(V_eff: jnp.ndarray, A: jnp.ndarray, tol: float = 1e-6):
    def _body(args):
        V_eff_, A_ = args
        obs = V_eff_[1:] / safe_div(V_eff_[:-1], 1.0)
        dA  = jnp.diff(A_)
        exp_ratio = safe_exp(-4.0 * dA)
        err_vec = jnp.abs(obs - exp_ratio)
        max_err = jnp.max(err_vec)
        mean_err = jnp.mean(err_vec)
        return max_err, mean_err

    max_err, mean_err = jax.lax.cond(
        jnp.logical_or(jnp.size(V_eff) < 2, jnp.size(A) < 2),
        lambda _: (jnp.inf, jnp.inf),
        _body,
        (V_eff, A),
    )
    passed = max_err < tol
    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "pass": passed,
    }

def audit_local_consistency(eps_local: jnp.ndarray, tol_frac: float = 0.05):
    def _body(x):
        mu = jnp.mean(x)
        sigma = jnp.std(x)
        rel = safe_div(sigma, jnp.abs(mu))
        return rel

    rel = jax.lax.cond(
        jnp.size(eps_local) == 0,
        lambda _: jnp.array(0.0, dtype=jnp.float64),
        _body,
        eps_local,
    )
    passed = rel < tol_frac
    return {
        "pass": passed,
        "rel_std": rel,
    }

def audit_monotone_A(A: jnp.ndarray):
    V_eff = safe_exp(-4.0 * A)
    diffs = jnp.diff(V_eff)

    def _body(d):
        return jnp.all(d <= 0.0)

    nonincreasing = jax.lax.cond(
        jnp.size(diffs) == 0,
        lambda _: jnp.array(True),
        _body,
        diffs,
    )
    return {"nonincreasing": nonincreasing}

# ---------- Analytic RS-superpotential solution ----------
def analytic_phi(y, vUV, c2):
    return vUV * jnp.exp(2.0 * c2 * y)

def analytic_A(y, p: Params, vUV, c2):
    term = (vUV**2 / 12.0) * (jnp.exp(4.0 * c2 * y) - 1.0)
    return p.k * y + term

def error_metrics(phi_num, A_num, y, p: Params, vUV, c2):
    phi_ref = analytic_phi(y, vUV, c2)
    A_ref   = analytic_A(y, p, vUV, c2)
    phi_err = jnp.max(jnp.abs(phi_num - phi_ref))
    A_err   = jnp.max(jnp.abs(A_num - A_ref))
    return {
        "phi_max_error": phi_err,
        "A_max_error": A_err,
    }

# ---------- Redshift & Planck mass ----------
def redshift_outputs(A, p: Params):
    A_IR = A[-1]
    redshift = jnp.exp(-A_IR)
    M_UV = jnp.array(1.0e19, dtype=jnp.float64)
    M_IR = M_UV * redshift
    return {
        "A_IR": A_IR,
        "redshift": redshift,
        "M_IR": M_IR,
    }

def planck_mass_integral(A, y):
    integrand = jnp.exp(2.0 * A)
    dy = jnp.diff(y)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    val = jnp.sum(mid * dy)
    return val

# ---------- Main solver: pure Einstein–scalar RS background ----------
def solve_einstein_scalar_rs(p: Params):
    """
    Pure 5D Einstein–scalar background from a superpotential W(φ) = W0 + c2 φ^2.
    This is a genuine solution of the Einstein–scalar system derived from W.
    """
    y, Ymax = make_stretched_grid(p)
    c2 = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    U0 = jnp.array([p.vUV, 0.0], dtype=jnp.float64)
    W0_const = W0(p)

    def fun(p_, y_, U_, Ymax_, c2_):
        return rhs_system_core(p_, U_, c2_, W0_const)

    Y, sub_counts = integrate_first_order_with_fun(fun, p, y, U0, Ymax, c2)
    phi = Y[:, 0]
    A   = Y[:, 1]

    # Hierarchy & audits
    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "local_consistency": audit_local_consistency(hdict["eps_local"]),
        "monotone_A": audit_monotone_A(A),
    }

    # Analytic error vs superpotential solution
    errors   = error_metrics(phi, A, y, p, p.vUV, c2)

    # Redshift & effective Planck mass
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y)

    # Superpotential constraints, curvature, NEC
    C1, C2, phi_p, A_p, W_val = einstein_constraint_superpotential(phi, A, y, p, c2)
    R = ricci_scalar_from_A(A, y)
    nec = nec_audit(phi_p)
    C1_norm = jnp.max(jnp.abs(C1))
    C2_norm = jnp.max(jnp.abs(C2))

    diagnostics = {
        "C1_phi_prime_minus_dW_dphi": C1,
        "C2_A_prime_minus_k2_over_3_W": C2,
        "C1_norm": C1_norm,
        "C2_norm": C2_norm,
        "ricci_scalar": R,
        "phi_prime": phi_p,
        "A_prime": A_p,
        "W": W_val,
        "NEC": nec,
    }

    return {
        "y": y,
        "phi": phi,
        "A": A,
        "c2": c2,
        "hierarchy": hdict,
        "audits": audits,
        "errors": errors,
        "redshift": redshift,
        "Mpl_eff": Mpl_eff,
        "substeps_last": sub_counts[-1],
        "diagnostics": diagnostics,
    }

solve_einstein_scalar_rs = jax.jit(
    solve_einstein_scalar_rs,
    static_argnames=("p",),
)

# ---------- KK graviton spectrum (tensor modes) ----------
def build_conformal_coordinate(y, A):
    """
    z(y) defined by dz/dy = e^{A(y)}.
    We compute z by cumulative trapezoidal integration.
    """
    eA = jnp.exp(A)
    dy = jnp.diff(y)
    mid = 0.5 * (eA[:-1] + eA[1:])
    dz = mid * dy
    z = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64),
                         jnp.cumsum(dz)])
    return z

def interpolate_linear(x_src, f_src, x_new):
    """
    Simple linear interpolation on a 1D grid (x_src ascending).
    """
    def interp_one(x):
        i = jnp.searchsorted(x_src, x, side="right") - 1
        i = jnp.clip(i, 0, x_src.shape[0]-2)
        x0 = x_src[i]
        x1 = x_src[i+1]
        f0 = f_src[i]
        f1 = f_src[i+1]
        t = safe_div(x - x0, x1 - x0)
        return f0 + t * (f1 - f0)

    return jax.vmap(interp_one)(x_new)

def kk_tensor_potential_from_A_z(z, A_z):
    """
    In conformal coordinate z, tensor modes obey:
      [-d^2/dz^2 + V_T(z)] ψ = m^2 ψ
    with
      V_T(z) = (3/2) A''(z) + (9/4) A'(z)^2
    where A(z) is the warp factor in conformal gauge.
    """
    A_p  = finite_diff_first(z, A_z)
    A_pp = finite_diff_second(z, A_z)
    V = 1.5 * A_pp + 2.25 * (A_p**2)
    return V

def build_schrodinger_operator(z, V):
    """
    Build finite-difference Hamiltonian H on z-grid with Dirichlet BC.
    H ψ = m^2 ψ, where
      H = -d^2/dz^2 + V(z)
    We use second-order central differences.
    """
    Nz = z.shape[0]
    dz = jnp.diff(z)
    dz_mean = jnp.mean(dz)
    dz2 = dz_mean**2

    Nint = Nz - 2
    diag = 2.0 / dz2 + V[1:-1]
    off  = -1.0 / dz2 * jnp.ones((Nint-1,), dtype=jnp.float64)

    H = jnp.zeros((Nint, Nint), dtype=jnp.float64)
    H = H.at[jnp.arange(Nint), jnp.arange(Nint)].set(diag)
    H = H.at[jnp.arange(Nint-1), jnp.arange(1, Nint)].set(off)
    H = H.at[jnp.arange(1, Nint), jnp.arange(Nint-1)].set(off)
    return H

def compute_kk_tensor_spectrum(out_background, Nz=2000, n_modes=5):
    """
    Given a background solution (y, A), construct conformal coordinate z,
    interpolate A(z) on a uniform z-grid, build Schrödinger operator,
    and compute the lowest n_modes eigenvalues m_n^2 (tensor KK masses).
    """
    y = out_background["y"]
    A = out_background["A"]

    z = build_conformal_coordinate(y, A)
    z_min = float(z[0])
    z_max = float(z[-1])

    z_uniform = jnp.linspace(z_min, z_max, Nz)
    A_z = interpolate_linear(z, A, z_uniform)

    V_T = kk_tensor_potential_from_A_z(z_uniform, A_z)
    H = build_schrodinger_operator(z_uniform, V_T)

    H_np = np.array(H)
    evals, evecs = np.linalg.eigh(H_np)

    idx = np.argsort(evals)
    evals_sorted = evals[idx]

    n_take = min(n_modes, evals_sorted.shape[0])
    m2 = evals_sorted[:n_take]
    m = np.sqrt(np.maximum(m2, 0.0))

    return {
        "z": np.array(z_uniform),
        "A_z": np.array(A_z),
        "V_T": np.array(V_T),
        "m2": m2,
        "m": m,
    }

# ---------- Plotting ----------
def plot_background(out, prefix="rs"):
    y = np.array(out["y"])
    phi = np.array(out["phi"])
    A = np.array(out["A"])
    R = np.array(out["diagnostics"]["ricci_scalar"])
    C1 = np.array(out["diagnostics"]["C1_phi_prime_minus_dW_dphi"])
    C2 = np.array(out["diagnostics"]["C2_A_prime_minus_k2_over_3_W"])

    # φ(y)
    plt.figure(figsize=(6,4))
    plt.plot(y, phi)
    plt.title("Scalar field φ(y)")
    plt.xlabel("y")
    plt.ylabel("φ")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{prefix}_phi.png", dpi=150)

    # A(y)
    plt.figure(figsize=(6,4))
    plt.plot(y, A)
    plt.title("Warp factor A(y)")
    plt.xlabel("y")
    plt.ylabel("A")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{prefix}_A.png", dpi=150)

    # Ricci scalar
    plt.figure(figsize=(6,4))
    plt.plot(y, R)
    plt.title("Ricci scalar R(y)")
    plt.xlabel("y")
    plt.ylabel("R")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{prefix}_R.png", dpi=150)

    # Constraints
    plt.figure(figsize=(6,4))
    plt.plot(y, C1, label="C1 = φ' - dW/dφ")
    plt.plot(y, C2, label="C2 = A' - κ²/3 W")
    plt.title("Superpotential constraints")
    plt.xlabel("y")
    plt.ylabel("Constraint value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{prefix}_constraints.png", dpi=150)

    print("PNG plots saved:",
          f"{prefix}_phi.png, {prefix}_A.png, {prefix}_R.png, {prefix}_constraints.png")

def plot_kk_potential(kk, prefix="rs_kk"):
    z = kk["z"]
    V = kk["V_T"]

    plt.figure(figsize=(6,4))
    plt.plot(z, V)
    plt.title("Tensor KK potential V_T(z)")
    plt.xlabel("z")
    plt.ylabel("V_T")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{prefix}_V.png", dpi=150)

    print("KK potential PNG saved:", f"{prefix}_V.png")

# ---------- Export ----------
def export_results(out, filename="results_rs.npz"):
    def to_np(x):
        return np.array(x)

    np.savez(
        filename,
        y=to_np(out["y"]),
        phi=to_np(out["phi"]),
        A=to_np(out["A"]),
        audits={k: {kk: np.array(vv) for kk, vv in v.items()}
                for k, v in out.get("audits", {}).items()},
        errors={k: np.array(v) for k, v in out.get("errors", {}).items()},
        redshift={k: np.array(v) for k, v in out.get("redshift", {}).items()},
        Mpl_eff=np.array(out.get("Mpl_eff", np.nan)),
        diagnostics={k: (np.array(v) if not isinstance(v, dict)
                         else {kk: np.array(vv) for kk, vv in v.items()})
                     for k, v in out.get("diagnostics", {}).items()},
    )

# ---------- Convergence & sensitivity (non-JIT) ----------
def run_study(p: Params):
    base = solve_einstein_scalar_rs(p)
    print("\n=== Convergence study (pure RS Einstein–scalar) ===")
    for Ny in [3001, 6001, 12001]:
        p2 = Params(**{**p.__dict__, "Ny": Ny})
        out2 = solve_einstein_scalar_rs(p2)
        n = min(base["phi"].shape[0], out2["phi"].shape[0])
        dphi = float(jnp.max(jnp.abs(out2["phi"][:n] - base["phi"][:n])))
        dA   = float(jnp.max(jnp.abs(out2["A"][:n]   - base["A"][:n])))
        print(f"Ny={Ny}: ΔΦ={dphi:.2e}, ΔA={dA:.2e}")

    print("\n=== Sensitivity study (vUV, rc) ===")
    for dv in [0.99, 1.01]:
        p2 = Params(**{**p.__dict__, "vUV": p.vUV * dv})
        out2 = solve_einstein_scalar_rs(p2)
        print(f"vUV×{dv}: A_IR={float(out2['redshift']['A_IR']):.2f}, "
              f"redshift={float(out2['redshift']['redshift']):.2e}")

    for rc in [8, 10, 12, 14]:
        p2 = Params(**{**p.__dict__, "rc": rc})
        out2 = solve_einstein_scalar_rs(p2)
        print(f"rc={rc}: A_IR={float(out2['redshift']['A_IR']):.2f}, "
              f"redshift={float(out2['redshift']['redshift']):.2e}")

    print("\n=== Planck mass integral ===")
    print("Effective M_Pl^2 integral:", float(base["Mpl_eff"]))
    return base

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ny", type=int, default=3001)
    parser.add_argument("--rc", type=float, default=12.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--kk", action="store_true",
                        help="Compute KK tensor spectrum on the RS background")
    parser.add_argument("--Nz", type=int, default=2000)
    parser.add_argument("--n_modes", type=int, default=5)
    args, _ = parser.parse_known_args()

    p = Params(
        Ny=args.Ny,
        rc=args.rc,
        rtol=args.rtol,
        atol=args.atol,
    )

    if args.study:
        out = run_study(p)
    else:
        out = solve_einstein_scalar_rs(p)

    print("\n=== Results (pure Einstein–scalar RS) ===")
    print("A_IR:", float(out["redshift"]["A_IR"]))
    print("redshift:", float(out["redshift"]["redshift"]))
    print("Mpl_eff:", float(out["Mpl_eff"]))
    print("c2:", float(out["c2"]))
    print("phi_err_max:", float(out["errors"]["phi_max_error"]))
    print("A_err_max:", float(out["errors"]["A_max_error"]))
    C1_norm = float(out["diagnostics"]["C1_norm"])
    C2_norm = float(out["diagnostics"]["C2_norm"])
    print("max|C1| (phi' - dW/dphi):", C1_norm)
    print("max|C2| (A' - kappa^2/3 W):", C2_norm)
    R = np.array(out["diagnostics"]["ricci_scalar"])
    print("min/max R:", float(np.min(R)), float(np.max(R)))

    # Plot background
    plot_background(out)

    if args.kk:
        print("\n=== KK tensor spectrum (Schrödinger in conformal z) ===")
        kk = compute_kk_tensor_spectrum(out, Nz=args.Nz, n_modes=args.n_modes)
        for i, (m2, m) in enumerate(zip(kk["m2"], kk["m"])):
            print(f"mode {i}: m^2 = {m2:.6e}, m = {m:.6e}")
        plot_kk_potential(kk)

    if args.export:
        export_results(out)
        print("Results exported to results_rs.npz")
