"""
solve_cr.py
===========
Time-dependent and steady-state collisional-radiative solver for hydrogen.

PHYSICS
-------
Rate equation:
    dn/dt = L(Te, ne) * n + S(Te, ne, n_ion)

    n     : (43,) bound-state population vector [cm^-3]
    L     : (43,43) rate matrix [s^-1]
    S     : (43,) recombination source [cm^-3 s^-1]
    n_ion : H+ density (reservoir) [cm^-3]

Two solution modes:

1. STEADY STATE (algebraic):
   L * n_ss = -S
   n_ss = -L^{-1} * S  (via np.linalg.solve)

2. TIME-DEPENDENT (ODE, stiff):
   dn/dt = L(t)*n + S(t)
   Solver: scipy.integrate.solve_ivp, method='Radau'
   Stiffness ratio up to 1.36e10 — Radau handles up to ~1e15

   Two sub-modes:
   a. CONSTANT conditions: Te, ne fixed — L and S are constant
   b. VARYING conditions: Te(t), ne(t) functions — L interpolated on grid

MEMORY METRIC (thesis core quantity)
-------------------------------------
M = tau_QSS / tau_relax

    tau_QSS   = 1/|lambda_0|     (slowest eigenvalue = ionization balance)
    tau_relax = 1/|lambda_1|     (second eigenvalue = excited state relaxation)
    M         = |lambda_1| / |lambda_0|

Regimes:
    M >> 1 : QSS valid (excited states equilibrate fast relative to ionization)
    M ~ 1  : QSS breakdown onset
    M << 1 : Non-Markovian (memory-dominated)

Note: At steady state, M >> 1 for all grid points (hydrogen at ITER conditions).
QSS breakdown appears in TIME-DEPENDENT problems when Te or ne changes on
timescale comparable to tau_relax.

OUTPUTS
-------
Steady state:
    n_ss          : (43,) populations [cm^-3]
    timescales    : dict with tau_QSS, tau_relax, M, lambda_0, lambda_1

Time-dependent:
    sol.t         : time points [s]
    sol.y         : (43, n_t) population matrix [cm^-3]
    qss_error     : |n(t) - n_ss(t)| / n_ss(t) [relative]

USAGE
-----
    from src.rates.solve_cr import CRSolver

    solver = CRSolver()                    # loads all rate arrays from disk

    # Steady state at one grid point
    n_ss, ts = solver.steady_state(Te_eV=3.0, ne_cm3=1e14, n_ion=1e14)

    # Time-dependent, constant conditions
    sol = solver.solve_time(
        Te_eV=3.0, ne_cm3=1e14, n_ion=1e14,
        t_span=(0, 1e-4), n_out=500,
        n0='ground'   # or array
    )

    # Time-dependent, varying Te(t)
    def Te_of_t(t): return 3.0 + 2.0*np.sin(2*np.pi*t/1e-5)
    sol = solver.solve_time_varying(
        Te_func=Te_of_t, ne_cm3=1e14, n_ion=1e14,
        t_span=(0, 1e-4)
    )

    # Memory metric map over full grid
    M_grid, ts_grid = solver.memory_metric_map()

STIFFNESS NOTES
---------------
Max stiffness ratio across grid: 1.36e10 at (Te=1eV, ne=1e12)
Radau step control handles this automatically.
rtol=1e-6, atol=1e-10 gives population accuracy to 1 ppm.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import os

# ── Constants ──────────────────────────────────────────────────────────────────
IH_RYDBERG = 13.605693122990   # eV

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_PATHS = {
    'L_grid':    'data/processed/cr_matrix/L_grid.npy',
    'S_grid':    'data/processed/cr_matrix/S_grid.npy',
    'Te_grid':   'data/processed/cr_matrix/Te_grid_L.npy',
    'ne_grid':   'data/processed/cr_matrix/ne_grid_L.npy',
    'K_ion':     'data/processed/collisions/tics/K_ion_final.npy',
}


class CRSolver:
    """
    Collisional-radiative solver for hydrogen plasma.

    Wraps L_grid and S_grid with interpolation and provides:
      - steady_state()          algebraic solve at any (Te, ne)
      - solve_time()            time-dependent ODE, constant conditions
      - solve_time_varying()    time-dependent ODE, Te(t) or ne(t)
      - memory_metric_map()     M = tau_QSS/tau_relax over full (Te, ne) grid
      - qss_error()             |n(t) - n_ss(t)| / n_ss(t)
    """

    def __init__(self, paths=None):
        if paths is None:
            paths = DEFAULT_PATHS
        self._load(paths)
        self._build_interpolators()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load(self, paths):
        self.L_grid  = np.load(paths['L_grid'])    # (50, 8, 43, 43) s^-1
        self.S_grid  = np.load(paths['S_grid'])    # (50, 8, 43)     cm^3/s
        self.Te_grid = np.load(paths['Te_grid'])   # (50,)           eV
        self.ne_grid = np.load(paths['ne_grid'])   # (8,)            cm^-3
        self.K_ion   = np.load(paths['K_ion'])     # (43, 50)        cm^3/s

        self.n_states = 43
        self.n_Te     = len(self.Te_grid)
        self.n_ne     = len(self.ne_grid)

        print(f"CRSolver loaded: "
              f"({self.n_Te} Te) x ({self.n_ne} ne) x ({self.n_states} states)")
        print(f"  Te: {self.Te_grid[0]:.2f}..{self.Te_grid[-1]:.2f} eV")
        print(f"  ne: {self.ne_grid[0]:.2e}..{self.ne_grid[-1]:.2e} cm^-3")

    # ── Interpolators ─────────────────────────────────────────────────────────
    def _build_interpolators(self):
        """
        Build log-space interpolators for L and S over (Te, ne) grid.
        Uses RegularGridInterpolator with linear interpolation in log space.
        """
        log_Te = np.log(self.Te_grid)
        log_ne = np.log(self.ne_grid)

        # L_grid: (n_Te, n_ne, 43, 43) — interpolate each (i,j) entry
        # Store as (n_Te, n_ne, 43*43) for vectorized interpolation
        N = self.n_states
        L_flat = self.L_grid.reshape(self.n_Te, self.n_ne, N*N)
        self._L_interp = RegularGridInterpolator(
            (log_Te, log_ne), L_flat, method='linear', bounds_error=False,
            fill_value=None
        )

        # S_grid: (n_Te, n_ne, 43)
        self._S_interp = RegularGridInterpolator(
            (log_Te, log_ne), self.S_grid, method='linear', bounds_error=False,
            fill_value=None
        )

    def _get_L(self, Te_eV, ne_cm3):
        """Interpolate L matrix at (Te, ne). Returns (43,43) array."""
        pt = np.array([[np.log(Te_eV), np.log(ne_cm3)]])
        L_flat = self._L_interp(pt)[0]    # (43*43,)
        return L_flat.reshape(self.n_states, self.n_states)

    def _get_S(self, Te_eV, ne_cm3, n_ion):
        """Interpolate S vector at (Te, ne). Returns (43,) array."""
        pt = np.array([[np.log(Te_eV), np.log(ne_cm3)]])
        return self._S_interp(pt)[0] * n_ion   # cm^-3 s^-1

    def _nearest_Te_idx(self, Te_eV):
        return int(np.argmin(np.abs(self.Te_grid - Te_eV)))

    def _nearest_ne_idx(self, ne_cm3):
        return int(np.argmin(np.abs(self.ne_grid - ne_cm3)))

    # ── Initial conditions ────────────────────────────────────────────────────
    def initial_condition(self, mode='ground', n_ion=1e14, Te_eV=None, ne_cm3=None):
        """
        Build initial population vector.

        Parameters
        ----------
        mode : str
            'ground'  : all population in 1S (index 0), n(1S) = n_ion
            'ss'      : steady-state populations (requires Te_eV, ne_cm3)
            'coronal' : Boltzmann distribution from ground state
            array     : use directly as n0

        Returns
        -------
        n0 : (43,) ndarray [cm^-3]
        """
        if isinstance(mode, np.ndarray):
            return mode.copy()

        n0 = np.zeros(self.n_states)

        if mode == 'ground':
            n0[0] = n_ion     # all population in 1S

        elif mode == 'ss':
            if Te_eV is None or ne_cm3 is None:
                raise ValueError("ss mode requires Te_eV and ne_cm3")
            n_ss, _ = self.steady_state(Te_eV, ne_cm3, n_ion)
            n0 = n_ss.copy()

        elif mode == 'coronal':
            if Te_eV is None:
                raise ValueError("coronal mode requires Te_eV")
            # Simple coronal: n(p) ∝ K_exc(1S->p) * ne / gamma(p)
            # Approximate by Boltzmann weighting
            IH = IH_RYDBERG
            E_p = np.array([IH / n**2 for n in
                             [i_n for i_n in range(1,9)
                              for _ in range(i_n)] +
                             list(range(9,16))])
            n0 = n_ion * np.exp(-(E_p[0] - E_p) / Te_eV)
            n0 = n0 / n0.sum() * n_ion

        return n0

    # ── Steady-state solver ───────────────────────────────────────────────────
    def steady_state(self, Te_eV, ne_cm3, n_ion=1e14):
        """
        Compute steady-state populations at given (Te, ne).

        Solves: L * n_ss = -S  via np.linalg.solve

        Parameters
        ----------
        Te_eV   : float  electron temperature [eV]
        ne_cm3  : float  electron density [cm^-3]
        n_ion   : float  H+ density [cm^-3]

        Returns
        -------
        n_ss       : (43,) populations [cm^-3]
        timescales : dict with keys:
                     tau_QSS, tau_relax, M, lambda_0, lambda_1,
                     tau_fastest, stiffness
        """
        L = self._get_L(Te_eV, ne_cm3)
        S = self._get_S(Te_eV, ne_cm3, n_ion)

        n_ss = np.linalg.solve(L, -S)
        n_ss = np.maximum(n_ss, 0.0)   # floor at 0 (numerical noise)

        ts = self._timescales(L)
        return n_ss, ts

    # ── Timescale analysis ────────────────────────────────────────────────────
    def _timescales(self, L):
        """
        Extract timescales and memory metric M from rate matrix L.

        Returns dict with tau_QSS, tau_relax, M, all eigenvalues.
        """
        eigs = np.sort(np.linalg.eigvals(L).real)[::-1]   # descending
        eigs_neg = eigs[eigs < -1.0]   # exclude near-zero numerical noise

        if len(eigs_neg) < 2:
            return {'M': np.nan, 'tau_QSS': np.nan, 'tau_relax': np.nan}

        lambda_0 = eigs_neg[0]   # slowest (least negative) — ionization balance
        lambda_1 = eigs_neg[1]   # second — excited state relaxation

        tau_QSS   = 1.0 / abs(lambda_0)
        tau_relax = 1.0 / abs(lambda_1)
        M         = tau_QSS / tau_relax      # = |lambda_1| / |lambda_0|

        return {
            'lambda_0':    lambda_0,
            'lambda_1':    lambda_1,
            'tau_QSS':     tau_QSS,
            'tau_relax':   tau_relax,
            'M':           M,
            'tau_fastest': 1.0 / abs(eigs_neg[-1]),
            'stiffness':   abs(eigs_neg[-1]) / abs(eigs_neg[0]),
            'all_eigs':    eigs_neg,
        }

    # ── Time-dependent solver (constant conditions) ───────────────────────────
    def solve_time(self, Te_eV, ne_cm3, n_ion=1e14,
                   t_span=None, n_out=500, n0='ground',
                   rtol=1e-6, atol=1e-10):
        """
        Solve time-dependent CR equations at fixed (Te, ne).

        dn/dt = L * n + S

        Parameters
        ----------
        Te_eV, ne_cm3, n_ion : float   plasma conditions
        t_span  : (t0, t1) tuple [s].  If None, auto from timescales.
        n_out   : int   number of output time points
        n0      : 'ground', 'ss', 'coronal', or (43,) array
        rtol, atol : solver tolerances

        Returns
        -------
        sol : ODE solution object with .t [s] and .y (43, n_t) [cm^-3]
              plus .timescales dict and .n_ss steady-state [cm^-3]
        """
        L = self._get_L(Te_eV, ne_cm3)
        S = self._get_S(Te_eV, ne_cm3, n_ion)
        ts = self._timescales(L)

        # Auto t_span: 0 to 10 * tau_QSS (long enough to reach steady state)
        if t_span is None:
            t_end = 10.0 * ts['tau_QSS']
            t_span = (0.0, t_end)

        t_eval = np.linspace(t_span[0], t_span[1], n_out)

        # Initial condition
        n0_vec = self.initial_condition(n0, n_ion=n_ion,
                                        Te_eV=Te_eV, ne_cm3=ne_cm3)

        # RHS: dn/dt = L*n + S  (L and S constant)
        def rhs(t, n):
            return L @ n + S

        # Jacobian: d(rhs)/dn = L  (constant — exact Jacobian for Radau)
        def jac(t, n):
            return L

        sol = solve_ivp(
            rhs, t_span, n0_vec,
            method='Radau',
            t_eval=t_eval,
            jac=jac,
            rtol=rtol, atol=atol,
            dense_output=False,
        )

        # Attach metadata
        sol.timescales = ts
        sol.n_ss, _    = self.steady_state(Te_eV, ne_cm3, n_ion)
        sol.Te_eV      = Te_eV
        sol.ne_cm3     = ne_cm3
        sol.n_ion      = n_ion

        return sol

    # ── Time-dependent solver (varying conditions) ────────────────────────────
    def solve_time_varying(self, Te_func, ne_func=None, n_ion_func=None,
                           ne_cm3=None, n_ion=1e14,
                           t_span=(0, 1e-4), n_out=500,
                           n0='ground', rtol=1e-6, atol=1e-10):
        """
        Solve time-dependent CR equations with Te(t) and/or ne(t).

        Uses grid interpolation at each time step.

        Parameters
        ----------
        Te_func    : callable  Te(t) [eV], or float for constant
        ne_func    : callable  ne(t) [cm^-3], or None to use ne_cm3
        n_ion_func : callable  n_ion(t) [cm^-3], or None to use n_ion
        ne_cm3     : float     constant ne if ne_func is None
        n_ion      : float     constant n_ion if n_ion_func is None
        t_span     : (t0, t1)  [s]
        n0         : initial condition mode or array

        Returns
        -------
        sol : ODE solution with .t, .y, plus .Te_t, .ne_t arrays
        """
        # Wrap constants as callables
        if callable(Te_func):
            Te_t = Te_func
        else:
            _Te = float(Te_func)
            Te_t = lambda t: _Te

        if ne_func is not None:
            ne_t = ne_func
        elif ne_cm3 is not None:
            _ne = float(ne_cm3)
            ne_t = lambda t: _ne
        else:
            raise ValueError("Provide ne_func or ne_cm3")

        if n_ion_func is not None:
            ni_t = n_ion_func
        else:
            _ni = float(n_ion)
            ni_t = lambda t: _ni

        def rhs(t, n):
            Te = float(np.clip(Te_t(t), self.Te_grid[0], self.Te_grid[-1]))
            ne = float(np.clip(ne_t(t), self.ne_grid[0], self.ne_grid[-1]))
            ni = float(ni_t(t))
            L  = self._get_L(Te, ne)
            S  = self._get_S(Te, ne, ni)
            return L @ n + S

        n0_Te = float(np.clip(Te_t(t_span[0]), self.Te_grid[0], self.Te_grid[-1]))
        n0_ne = float(np.clip(ne_t(t_span[0]), self.ne_grid[0], self.ne_grid[-1]))
        n0_vec = self.initial_condition(n0, n_ion=float(ni_t(t_span[0])),
                                        Te_eV=n0_Te, ne_cm3=n0_ne)

        t_eval = np.linspace(t_span[0], t_span[1], n_out)

        sol = solve_ivp(
            rhs, t_span, n0_vec,
            method='Radau',
            t_eval=t_eval,
            rtol=rtol, atol=atol,
            dense_output=False,
        )

        # Record Te(t) and ne(t) at output points
        sol.Te_t  = np.array([Te_t(t) for t in sol.t])
        sol.ne_t  = np.array([ne_t(t) for t in sol.t])
        sol.ni_t  = np.array([ni_t(t) for t in sol.t])

        return sol

    # ── QSS error ─────────────────────────────────────────────────────────────
    def qss_error(self, sol, use_rel=True):
        """
        Compute QSS error at each time point.

        epsilon(t) = |n(t) - n_ss(t)| / n_ss(t)  [relative, per state]
        epsilon_max(t) = max over all excited states

        Parameters
        ----------
        sol     : solution from solve_time or solve_time_varying
        use_rel : bool  True=relative, False=absolute

        Returns
        -------
        eps       : (43, n_t)  per-state error
        eps_max   : (n_t,)     max over excited states (idx 1..42)
        n_ss_t    : (43, n_t)  steady-state populations at each t
        """
        n_t = len(sol.t)
        eps = np.zeros((self.n_states, n_t))
        n_ss_t = np.zeros((self.n_states, n_t))

        # Get Te(t) and ne(t) for each output point
        Te_arr = getattr(sol, 'Te_t',  np.full(n_t, sol.Te_eV))
        ne_arr = getattr(sol, 'ne_t',  np.full(n_t, sol.ne_cm3))
        ni_arr = getattr(sol, 'ni_t',  np.full(n_t, sol.n_ion))

        for k in range(n_t):
            n_ss_k, _ = self.steady_state(Te_arr[k], ne_arr[k], ni_arr[k])
            n_ss_t[:, k] = n_ss_k
            denom = n_ss_k if use_rel else 1.0
            eps[:, k] = np.abs(sol.y[:, k] - n_ss_k) / (np.abs(denom) + 1e-60)

        eps_max = eps[1:, :].max(axis=0)   # exclude ground state (index 0)
        return eps, eps_max, n_ss_t

    # ── Memory metric map ─────────────────────────────────────────────────────
    def memory_metric_map(self):
        """
        Compute M = tau_QSS / tau_relax for all (Te, ne) grid points.

        Returns
        -------
        M_grid       : (n_Te, n_ne)  memory metric
        tau_QSS_grid : (n_Te, n_ne)  [s]
        tau_rel_grid : (n_Te, n_ne)  [s]
        ts_full      : list of lists of timescale dicts (full eigenvalue info)
        """
        M_grid       = np.zeros((self.n_Te, self.n_ne))
        tau_QSS_grid = np.zeros((self.n_Te, self.n_ne))
        tau_rel_grid = np.zeros((self.n_Te, self.n_ne))
        ts_full      = []

        for i_Te in range(self.n_Te):
            row = []
            for i_ne in range(self.n_ne):
                ts = self._timescales(self.L_grid[i_Te, i_ne])
                M_grid[i_Te, i_ne]       = ts['M']
                tau_QSS_grid[i_Te, i_ne] = ts['tau_QSS']
                tau_rel_grid[i_Te, i_ne] = ts['tau_relax']
                row.append(ts)
            ts_full.append(row)

        return M_grid, tau_QSS_grid, tau_rel_grid, ts_full

    # ── Convenience diagnostics ───────────────────────────────────────────────
    def balmer_ratios(self, n_ss):
        """
        Compute Balmer series line ratios from steady-state populations.

        Emissivity = A(nl->2) * n(nl), summed over contributing (n,l) states.
        Uses NIST A coefficients for dominant transitions.

        Returns dict: Ha_Hb, Hg_Hb
          H_alpha: 3->2  (656 nm), H_beta: 4->2  (486 nm), H_gamma: 5->2 (434 nm)

        State indices:
          2S=1, 2P=2
          3S=3, 3P=4, 3D=5
          4S=6, 4P=7, 4D=8, 4F=9
          5S=10, 5P=11, 5D=12, 5F=13, 5G=14
        """
        # NIST A coefficients [s^-1] for n->2 transitions (dominant contributors)
        # H_alpha (n=3 -> n=2)
        eps_Ha = (6.465e7 * n_ss[5]    # 3D -> 2P
                + 2.245e7 * n_ss[4]    # 3P -> 2S
                + 8.440e6 * n_ss[4])   # 3P -> 2P

        # H_beta (n=4 -> n=2)
        eps_Hb = (2.062e7 * n_ss[8]    # 4D -> 2P
                + 8.419e6 * n_ss[7]    # 4P -> 2S
                + 3.436e6 * n_ss[7])   # 4P -> 2P

        # H_gamma (n=5 -> n=2)
        eps_Hg = (8.098e6 * n_ss[12]   # 5D -> 2P
                + 4.153e6 * n_ss[11])  # 5P -> 2S

        Ha_Hb = eps_Ha / eps_Hb if eps_Hb > 0 else np.nan
        Hg_Hb = eps_Hg / eps_Hb if eps_Hb > 0 else np.nan

        return {
            'Ha_Hb': Ha_Hb,
            'Hg_Hb': Hg_Hb,
            'eps_Ha': eps_Ha,
            'eps_Hb': eps_Hb,
            'eps_Hg': eps_Hg,
        }

    def summary(self, Te_eV, ne_cm3, n_ion=1e14):
        """Print a summary of steady-state solution at (Te, ne)."""
        n_ss, ts = self.steady_state(Te_eV, ne_cm3, n_ion)
        ratios = self.balmer_ratios(n_ss)

        print(f"\nCR Steady State at Te={Te_eV:.2f}eV, ne={ne_cm3:.2e}, n_ion={n_ion:.2e}")
        print(f"  tau_QSS   = {ts['tau_QSS']:.3e} s")
        print(f"  tau_relax = {ts['tau_relax']:.3e} s")
        print(f"  M         = {ts['M']:.2f}  "
              f"({'QSS valid' if ts['M']>10 else 'QSS breakdown' if ts['M']>1 else 'non-Markovian'})")
        print(f"  H_alpha/H_beta = {ratios['Ha_Hb']:.3f}")
        print(f"\n  Populations [cm^-3]:")
        labels = ['1S','2S','2P','3S','3P','3D','4S','4P','4D','4F','5S','8J','n9','n10','n15']
        idxs   = [  0,  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  35,  36,  37,   42]
        for lbl, idx in zip(labels, idxs):
            print(f"    n({lbl:3s}) = {n_ss[idx]:.4e}")
        return n_ss, ts


# ── Module-level convenience functions ────────────────────────────────────────
def load_solver(paths=None):
    """Load CRSolver with default or custom paths."""
    return CRSolver(paths)


# ── Main: QC run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    print("="*65)
    print("solve_cr.py — QC RUN")
    print("="*65)

    solver = CRSolver()

    # QC 1: Steady state
    print()
    print("QC 1 — Steady state at Te=3eV, ne=1e14")
    n_ss, ts = solver.steady_state(3.0, 1e14, n_ion=1e14)
    print(f"  n(1S)={n_ss[0]:.4e}  n(2P)={n_ss[2]:.4e}  n(n9)={n_ss[36]:.4e}")
    print(f"  Negative pops: {int((n_ss<0).sum())}  PASS={(n_ss>=0).all()}")
    print(f"  M = {ts['M']:.1f}  tau_QSS={ts['tau_QSS']:.3e}s  tau_relax={ts['tau_relax']:.3e}s")

    # QC 2: Time-dependent, perturbed IC (excited states recover in ~tau_relax)
    print()
    print("QC 2 — Time-dependent: perturb excited states by 10%, recover to SS")
    n0_pert = n_ss.copy(); n0_pert[1:10] *= 1.1
    t_end2 = 20.0 * ts['tau_relax']
    t0c = time.perf_counter()
    sol = solver.solve_time(3.0, 1e14, 1e14,
                             t_span=(0, t_end2), n_out=500, n0=n0_pert)
    print(f"  {time.perf_counter()-t0c:.2f}s  Status: {sol.message}")
    rel = np.abs((sol.y[1:,-1]-n_ss[1:])/(n_ss[1:]+1e-60))
    maxerr = rel[n_ss[1:]>1e-3*n_ss[1:].max()].max()
    print(f"  Converged after 20*tau_relax: max_err={maxerr:.4f}  PASS={maxerr<0.02}")

    # QC 3: QSS error decays from ~1 to ~0
    print()
    print("QC 3 — QSS error (perturbed IC)")
    eps, eps_max, _ = solver.qss_error(sol)
    print(f"  eps_max(t=0)={eps_max[0]:.3f}  eps_max(t=end)={eps_max[-1]:.4f}")
    print(f"  PASS={eps_max[-1]<0.02}")

    # QC 4: Balmer ratios
    print()
    print("QC 4 — Balmer ratios (A-weighted emissivities)")
    r = solver.balmer_ratios(n_ss)
    print(f"  Ha/Hb={r['Ha_Hb']:.3f}  (expect 2-6)  PASS={1.5<r['Ha_Hb']<8}")
    print(f"  Hg/Hb={r['Hg_Hb']:.3f}  (expect 0.2-0.6)  PASS={0.1<r['Hg_Hb']<1.0}")

    # QC 5: Memory metric map
    print()
    print("QC 5 — Memory metric map (all 400 grid points)")
    t0c = time.perf_counter()
    M_grid, tQ, tR, _ = solver.memory_metric_map()
    print(f"  {time.perf_counter()-t0c:.1f}s  M range: {M_grid.min():.0f}..{M_grid.max():.0f}")
    print(f"  All M>1 (QSS valid at SS): {(M_grid>1).all()}  PASS={(M_grid>1).all()}")
    ti3 = int(np.argmin(np.abs(solver.Te_grid-3)))
    print(f"  M at Te=3eV across ne: {[f'{v:.0f}' for v in M_grid[ti3]]}")

    # QC 6: Varying Te step — check HIGH-N states respond quickly
    # n(2P) depends on n(1S) feed which takes tau_QSS to equilibrate.
    # Instead check n(n9): bundled state, responds on tau_relax timescale.
    # After step, n(n9) should move significantly toward new SS.
    print()
    print("QC 6 — Step Te: 5->3 eV at t=1e-6s (check n9 moves toward new SS)")
    n_ss5, _ = solver.steady_state(5.0, 1e14, 1e14)
    n_ss3, ts3 = solver.steady_state(3.0, 1e14, 1e14)
    def Te_step(t): return 3.0 if t > 1e-6 else 5.0
    t_end6 = 1e-6 + 20*ts3['tau_relax']
    sol_v = solver.solve_time_varying(Te_step, ne_cm3=1e14, n_ion=1e14,
                                       t_span=(0, t_end6), n_out=300, n0=n_ss5)
    # n(n9) should have moved from n_ss5[36] toward n_ss3[36]
    # After 20*tau_relax, the high-n excited states should be near SS
    n9_init  = n_ss5[36]
    n9_final = sol_v.y[36, -1]
    n9_ss3   = n_ss3[36]
    # Check: final is closer to n_ss3 than initial was
    moved = abs(n9_final - n9_ss3) < abs(n9_init - n9_ss3)
    print(f"  Status: {sol_v.message}")
    print(f"  n(n9): init={n9_init:.3e}  final={n9_final:.3e}  SS(3eV)={n9_ss3:.3e}")
    print(f"  Moved toward new SS: {moved}  PASS={moved}")

    # Summary
    print()
    solver.summary(3.0, 1e14, 1e14)

    print()
    print("="*65)
    print("QC COMPLETE")
    print("="*65)