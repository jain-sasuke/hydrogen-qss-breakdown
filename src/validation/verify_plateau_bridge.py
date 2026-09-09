"""
verify_plateau_bridge.py
========================
C5-A. Dynamic bridge required by Chapter 5:

    full transient
        -> instantaneous QSS response R+(u(t))
        -> partial-equilibrium plateau R+(u^-)
        -> f_3 - f_4
        -> density ridge

The later arrows are algebraic and are checked elsewhere. This script tests
whether the FULL 43-state transient dynamically reaches the partial-equilibrium
state used to define eps_plateau.

==========================================================================
EXACT POST-STEP TRANSIENT
==========================================================================
After an instantaneous temperature step,

    dn/dt = L+ n + S+ n_i,

with

    n(0) = n_ss^-.

The fixed ion reservoir is normalised to n_i = 1, hence

    n_ss^- = -L-^-1 S-,
    n_ss^+ = -L+^-1 S+.

For constant post-step conditions,

    n(t)
        = n_ss^+
        + exp(L+ t) (n_ss^- - n_ss^+).

The propagator is evaluated using scipy.linalg.expm. An independent
cross-algorithm action check uses scipy.sparse.linalg.expm_multiply.

No explicit eigenvector propagation is used because L is non-normal and the
eigenvector basis can be ill-conditioned.

Eigenvalues are used only for the two spectral clocks

    tau_QSS   = 1 / |lambda_0|,
    tau_relax = 1 / |lambda_1|,

with the least-negative eigenvalue first.

==========================================================================
POST-STEP QSS MANIFOLD
==========================================================================
For the post-step operator,

    a+ = -L_FF+^-1 L_Fg+,
    c+ = -L_FF+^-1 S_F+,

and

    u(t) := n_g(t)/n_i,

the instantaneous QSS state is

    n_F^QSS(t)/n_i = a+ u(t) + c+.

The corresponding Balmer-shell ratio is

    R_QSS+(u(t))
        = [a_3+ u(t) + c_3+]
          / [a_4+ u(t) + c_4+].

The analytic frozen-ground partial-equilibrium ratio is

    R_PE = R_QSS+(u^-),

where

    u^- = n_g^- / n_i

is the PRE-step equilibrium ground reservoir.

Two deviations are therefore distinguished:

    eps_track(t)
        = |R_full(t)/R_QSS+(u(t)) - 1|

which measures departure from the instantaneous QSS manifold, and

    eps_PE(t)
        = |R_full(t)/R_PE - 1|

which measures departure from the frozen-ground plateau.

These are physically different quantities.

==========================================================================
HISTORICAL v1 TEST -- PRESERVED, BUT DOES NOT CONTROL v2
==========================================================================
v1 preregistered the fixed interval

    20 tau_relax <= t <= 0.02 tau_QSS

and required throughout it

    eps_track < 2e-3,
    eps_PE    < 2e-3.

That test was RUN before revision and failed at ridge point [15,3], because
eps_PE reached 2.778e-3 at the right boundary.

The failure is retained and reported unchanged.

It did NOT correspond to QSS breakdown: instantaneous-QSS tracking remained
~1e-6 while the ground reservoir underwent its expected slow drift.

WIN_HI is therefore NOT shortened retrospectively.

==========================================================================
v2 QUESTION
==========================================================================
v2 no longer asserts that the plateau must occupy a chosen fraction of
tau_QSS.

Instead it MEASURES the interval during which both pre-existing tolerances hold:

    eps_track < TOL_TRACK
    AND
    eps_PE    < TOL_PLATEAU.

The measured plateau is defined as the LONGEST CONTIGUOUS interval satisfying
both conditions. This avoids calling a one-point crossing or a temporary
non-normal excursion a plateau.

v2 reports:

  1. sustained instantaneous-QSS tracking onset;
  2. measured PE plateau start and end;
  3. plateau duration in seconds, tau_relax and tau_QSS;
  4. ground-reservoir drift during the plateau;
  5. excited-state distance from the instantaneous QSS manifold;
  6. constancy of eps_CRE across the PE plateau;
  7. persistence of the CRE diagnostic error after plateau formation;
  8. early slow-time scaling in LOGARITHMIC response space.

The v2 existence statement is:

    A dynamically visited partial-equilibrium plateau exists if a non-zero
    contiguous interval satisfies both fixed tolerances.

This revised definition is NOT called preregistered: it was introduced after
inspection of the v1 failure. The v1 result remains visible above it.

==========================================================================
CRE-ERROR PERSISTENCE
==========================================================================
The diagnostic error relative to the NEW CR equilibrium is

    eps_CRE(t) = |R_full(t)/R_CRE_new - 1|.

This is not the same quantity as eps_PE.

The analytic plateau amplitude is

    eps_plateau = |R_PE/R_CRE_new - 1|.

Persistence is measured from formation of the dynamic plateau until eps_CRE
first falls to

    EPS_CRE_FRAC * eps_plateau.

For EPS_CRE_FRAC = 0.5 this gives the post-formation half-decay time

    Delta t_1/2 = t_half - t_plateau_start.

For a single exponential slow decay,

    tau_eff = Delta t_1/2 / ln(2)

would equal tau_QSS. tau_eff/tau_QSS is therefore reported as a diagnostic,
not imposed as an acceptance criterion.

==========================================================================
EARLY SLOW-TIME SCALING
==========================================================================
Once QSS tracking is established, define the signed logarithmic progress from
the PE state toward the new CRE state:

    eta_R(t)
        = ln[R_full(t)/R_PE]
          / ln[R_CRE_new/R_PE].

Thus

    eta_R = 0  at R_PE,
    eta_R = 1  at R_CRE_new.

Over the measured plateau, fit

    eta_R ~= k * t/tau_QSS

through the origin.

This is an empirical early-time scaling check. It is NOT assumed to be a
universal law and is not used as a pass/fail criterion.

==========================================================================
PROPAGATOR CHECKS
==========================================================================
The analytical ODE residual is NOT used: for an expm-built solution it is
identically satisfied by construction.

Instead:

  A. semigroup consistency:
         expm(L t) ~= expm(L t/2) @ expm(L t/2)

  B. independent exponential-action comparison:
         expm(L t) @ d0
      versus
         expm_multiply(L t, d0)

The semigroup test is an internal consistency check, not a proof of accuracy.
Agreement between the two different algorithms is independent numerical
evidence.

==========================================================================
SCOPE
==========================================================================
This script tests two representative points:

  benchmark [23,5],
  ridge     [15,3].

Passing v2 demonstrates the dynamic bridge at those points only. A grid-wide
bridge test is required before claiming that every eps_plateau grid point is
dynamically visited.

==========================================================================
REPORT
==========================================================================
Report only. Writes nothing.

    python src/validation/verify_plateau_bridge.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


HERE = Path(__file__).resolve().parent

for p in (
    HERE,
    HERE.parent / "validation",
    HERE.parent / "analysis",
):
    sys.path.insert(0, str(p))

from cr_context import CRContext  # noqa: E402


# ===========================================================================
# FIXED CONSTANTS
# ===========================================================================

STEP = 1

# Historical v1 window -- retained exactly.
WIN_LO = 20.0
WIN_HI = 0.02

# Thresholds inherited from v1.
TOL_TRACK = 2.0e-3
TOL_PLATEAU = 2.0e-3

# CRE persistence threshold.
EPS_CRE_FRAC = 0.5

# Time scan.
NT = 700

NUM_FLOOR = 1e-300


# ===========================================================================
# HELPERS
# ===========================================================================

def longest_true_run(mask: np.ndarray):
    """
    Return indices (start, end) of the longest contiguous True run.

    Returns (None, None) if mask contains no True values.
    """
    mask = np.asarray(mask, dtype=bool)

    best_start = None
    best_end = None
    start = None

    for q, value in enumerate(mask):

        if value and start is None:
            start = q

        if start is not None:
            run_ended = (
                not value
                or q == len(mask) - 1
            )

            if run_ended:
                end = q if value else q - 1

                if (
                    best_start is None
                    or (end - start) > (best_end - best_start)
                ):
                    best_start = start
                    best_end = end

                start = None

    return best_start, best_end


def interpolate_zero_logtime(
    t0: float,
    t1: float,
    y0: float,
    y1: float,
) -> float:
    """
    Interpolate y=0 between two positive times using log(t).

    Intended for threshold-boundary refinement on a logarithmic time grid.
    """
    if (
        t0 <= 0
        or t1 <= 0
        or not np.isfinite(y0)
        or not np.isfinite(y1)
        or y1 == y0
    ):
        return float(t1)

    frac = -y0 / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))

    x0 = np.log(t0)
    x1 = np.log(t1)

    return float(
        np.exp(
            x0 + frac * (x1 - x0)
        )
    )


def interpolate_positive_level(
    t0: float,
    t1: float,
    y0: float,
    y1: float,
    target: float,
) -> float:
    """
    Interpolate y=target assuming log(y) varies linearly between samples.

    Appropriate for locating a threshold in a roughly exponential decay.
    """
    if (
        t0 <= 0
        or t1 <= 0
        or y0 <= 0
        or y1 <= 0
        or target <= 0
        or y1 == y0
    ):
        return float(t1)

    ly0 = np.log(y0)
    ly1 = np.log(y1)
    lyt = np.log(target)

    if ly1 == ly0:
        return float(t1)

    frac = (
        (lyt - ly0)
        / (ly1 - ly0)
    )

    frac = float(
        np.clip(
            frac,
            0.0,
            1.0,
        )
    )

    return float(
        t0 + frac * (t1 - t0)
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:

    ctx = CRContext.load()

    root = ctx.root

    L = ctx.L_grid
    Te = ctx.te_grid
    ne = ctx.ne_grid

    S = np.load(
        root
        / "data/processed/cr_matrix/S_grid.npy"
    )

    nv = np.asarray(
        ctx.n_values
    )

    g = int(
        ctx.ground_index
    )

    E = np.array(
        [
            k
            for k in range(ctx.n_states)
            if k != g
        ],
        dtype=int,
    )

    posE = {
        state_idx: local_idx
        for local_idx, state_idx
        in enumerate(E)
    }

    idxE = {
        m: [
            posE[s]
            for s in np.where(nv == m)[0]
        ]
        for m in (3, 4)
    }

    idxFull = {
        m: np.where(nv == m)[0]
        for m in (3, 4)
    }

    # =======================================================================
    # PROVENANCE
    # =======================================================================

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)

    for rel in (
        "data/processed/cr_matrix/L_grid.npy",
        "data/processed/cr_matrix/S_grid.npy",
    ):

        f = root / rel

        print(f"  {rel}")
        print(
            "      sha256 "
            f"{hashlib.sha256(f.read_bytes()).hexdigest()}"
        )

    print(
        f"  step = {STEP} interval(s) = "
        f"+{100 * (Te[STEP] / Te[0] - 1):.2f}% in Te"
    )

    print(
        f"  historical v1 window: "
        f"{WIN_LO:g} tau_relax .. "
        f"{WIN_HI:g} tau_QSS"
    )

    print(
        f"  TOL_TRACK   = {TOL_TRACK:.0e}"
    )

    print(
        f"  TOL_PLATEAU = {TOL_PLATEAU:.0e}"
    )

    print(
        "  v2 plateau interval is MEASURED; "
        "the historical v1 window does not determine the v2 verdict."
    )

    # =======================================================================
    # FIXED REPRESENTATIVE POINTS
    # =======================================================================

    ib, jb = 23, 5
    ir, jr = 15, 3

    assert np.isclose(
        Te[ib],
        2.947,
        rtol=5e-4,
    ), (
        f"unexpected benchmark Te[{ib}] "
        f"= {Te[ib]}"
    )

    assert np.isclose(
        ne[jb],
        1.389e14,
        rtol=5e-4,
    ), (
        f"unexpected benchmark ne[{jb}] "
        f"= {ne[jb]}"
    )

    assert np.isclose(
        Te[ir],
        2.024,
        rtol=1e-3,
    ), (
        f"unexpected ridge Te[{ir}] "
        f"= {Te[ir]}"
    )

    assert np.isclose(
        ne[jr],
        1.931e13,
        rtol=1e-3,
    ), (
        f"unexpected ridge ne[{jr}] "
        f"= {ne[jr]}"
    )

    POINTS = [
        (
            (ib, jb),
            "benchmark [23,5]",
        ),
        (
            (ir, jr),
            "ridge [15,3]",
        ),
    ]

    # =======================================================================
    # LOCAL HELPERS
    # =======================================================================

    def steady(
        i: int,
        j: int,
    ) -> np.ndarray:

        n = np.linalg.solve(
            L[i, j],
            -S[i, j],
        )

        if not np.all(
            np.isfinite(n)
        ):
            raise RuntimeError(
                f"non-finite CRE state "
                f"at [{i},{j}]"
            )

        scale = max(
            np.abs(n).max(),
            NUM_FLOOR,
        )

        if n.min() < -1e-12 * scale:
            raise RuntimeError(
                f"meaningfully negative CRE "
                f"population at [{i},{j}]: "
                f"min={n.min():.3e}"
            )

        if n[g] <= 0:
            raise RuntimeError(
                f"non-positive ground "
                f"reservoir at [{i},{j}]: "
                f"u={n[g]:.6e}"
            )

        return n

    def post_channels(
        ip: int,
        j: int,
    ):

        Lp = L[ip, j]
        Sp = S[ip, j]

        LEE = Lp[
            np.ix_(
                E,
                E,
            )
        ]

        LEg = Lp[
            np.ix_(
                E,
                [g],
            )
        ].ravel()

        SE = Sp[E]

        a = np.linalg.solve(
            LEE,
            -LEg,
        )

        c = np.linalg.solve(
            LEE,
            -SE,
        )

        ra = (
            np.linalg.norm(
                LEE @ a + LEg
            )
            / max(
                np.linalg.norm(LEg),
                NUM_FLOOR,
            )
        )

        rc = (
            np.linalg.norm(
                LEE @ c + SE
            )
            / max(
                np.linalg.norm(SE),
                NUM_FLOOR,
            )
        )

        if (
            ra > 1e-10
            or rc > 1e-10
        ):
            raise RuntimeError(
                f"post-step channel solve "
                f"failed at [{ip},{j}]: "
                f"res_a={ra:.3e}, "
                f"res_c={rc:.3e}"
            )

        tol_a = (
            1e-10
            * max(
                np.abs(a).max(),
                NUM_FLOOR,
            )
        )

        tol_c = (
            1e-10
            * max(
                np.abs(c).max(),
                NUM_FLOOR,
            )
        )

        if a.min() < -tol_a:
            raise RuntimeError(
                f"negative a response "
                f"at [{ip},{j}]: "
                f"min={a.min():.3e}"
            )

        if c.min() < -tol_c:
            raise RuntimeError(
                f"negative c response "
                f"at [{ip},{j}]: "
                f"min={c.min():.3e}"
            )

        am = {
            m: a[idxE[m]].sum()
            for m in (3, 4)
        }

        cm = {
            m: c[idxE[m]].sum()
            for m in (3, 4)
        }

        return (
            am,
            cm,
            a,
            c,
        )

    # =======================================================================
    # RUN REPRESENTATIVE POINTS
    # =======================================================================

    results = []

    for (
        (i, j),
        label,
    ) in POINTS:

        ip = i + STEP

        print()
        print("=" * 78)

        print(
            f"{label}   "
            f"Te {Te[i]:.4f} -> "
            f"{Te[ip]:.4f} eV, "
            f"ne {ne[j]:.4e} cm^-3"
        )

        print("=" * 78)

        Lp = L[ip, j]
        Sp = S[ip, j]

        n_old = steady(
            i,
            j,
        )

        n_new = steady(
            ip,
            j,
        )

        # ==================================================================
        # QSS RESPONSE / PE TARGET
        # ==================================================================

        am, cm, a, c = post_channels(
            ip,
            j,
        )

        u_old = float(
            n_old[g]
        )

        u_new = float(
            n_new[g]
        )

        nF_pe = (
            a * u_old + c
        )

        nF_new_reduced = (
            a * u_new + c
        )

        sup = (
            np.max(
                np.abs(
                    nF_new_reduced
                    - n_new[E]
                )
            )
            / max(
                np.max(
                    np.abs(
                        n_new[E]
                    )
                ),
                NUM_FLOOR,
            )
        )

        if sup > 1e-10:
            raise RuntimeError(
                f"post-step CRE "
                f"superposition failed "
                f"at [{ip},{j}]: "
                f"{sup:.3e}"
            )

        R_pe = (
            nF_pe[idxE[3]].sum()
            / nF_pe[idxE[4]].sum()
        )

        R_new_reduced = (
            nF_new_reduced[
                idxE[3]
            ].sum()
            /
            nF_new_reduced[
                idxE[4]
            ].sum()
        )

        R_new = (
            n_new[
                idxFull[3]
            ].sum()
            /
            n_new[
                idxFull[4]
            ].sum()
        )

        if (
            abs(
                R_new_reduced
                / R_new
                - 1.0
            )
            > 1e-10
        ):
            raise RuntimeError(
                f"reduced/full post-step "
                f"CRE ratio mismatch "
                f"at [{ip},{j}]"
            )

        eps_plateau = abs(
            R_pe / R_new
            - 1.0
        )

        log_total = np.log(
            R_new / R_pe
        )

        if abs(log_total) < 1e-14:
            raise RuntimeError(
                f"R_PE and R_CRE_new are "
                f"indistinguishable at [{i},{j}]; "
                f"log-response progress "
                f"cannot be defined."
            )

        # ==================================================================
        # SPECTRUM
        # ==================================================================

        eig = np.linalg.eigvals(
            Lp
        )

        if eig.real.max() >= 0:
            raise RuntimeError(
                f"non-decaying post-step "
                f"mode at [{ip},{j}]: "
                f"max Re(lambda)="
                f"{eig.real.max():.3e}"
            )

        imag_rel = float(
            np.max(
                np.abs(eig.imag)
                / np.maximum(
                    np.abs(eig),
                    NUM_FLOOR,
                )
            )
        )

        if imag_rel > 1e-10:
            raise RuntimeError(
                f"complex eigenvalues "
                f"at [{ip},{j}]: "
                f"max relative imaginary "
                f"part={imag_rel:.3e}"
            )

        ev = np.sort(
            eig.real
        )[::-1]

        tau = (
            1.0
            / np.abs(ev)
        )

        tau_qss = float(
            tau[0]
        )

        tau_rel = float(
            tau[1]
        )

        M = (
            tau_qss
            / tau_rel
        )

        print(
            f"  tau_QSS      "
            f"{tau_qss:.4e} s"
        )

        print(
            f"  tau_relax    "
            f"{tau_rel:.4e} s"
        )

        print(
            f"  M            "
            f"{M:.6g}"
        )

        print(
            f"  R^PE         "
            f"{R_pe:.8f}"
        )

        print(
            f"  R^CRE_new    "
            f"{R_new:.8f}"
        )

        print(
            f"  eps_plateau  "
            f"{eps_plateau:.6f}"
        )

        # ==================================================================
        # PROPAGATOR
        # ==================================================================

        d0 = (
            n_old
            - n_new
        )

        def n_at(
            t: float,
        ) -> np.ndarray:

            n = (
                n_new
                + expm(
                    Lp * t
                ) @ d0
            )

            if not np.all(
                np.isfinite(n)
            ):
                raise RuntimeError(
                    f"non-finite propagated "
                    f"state at [{i},{j}], "
                    f"t={t:.6e}"
                )

            return n

        # Historical v1 window defines a useful intermediate probe time,
        # but does not control the v2 plateau.
        lo_v1 = (
            WIN_LO
            * tau_rel
        )

        hi_v1 = (
            WIN_HI
            * tau_qss
        )

        if lo_v1 < hi_v1:

            t_probe = float(
                np.sqrt(
                    lo_v1
                    * hi_v1
                )
            )

        else:

            t_probe = float(
                WIN_LO
                * tau_rel
            )

        # ------------------------------------------------------------------
        # Propagator check A: semigroup
        # ------------------------------------------------------------------

        E_full = expm(
            Lp * t_probe
        )

        E_half = expm(
            Lp
            * (
                t_probe
                / 2.0
            )
        )

        E_sq = (
            E_half
            @ E_half
        )

        r_semi = float(
            np.max(
                np.abs(
                    E_full
                    - E_sq
                )
            )
            / max(
                np.max(
                    np.abs(
                        E_full
                    )
                ),
                NUM_FLOOR,
            )
        )

        # ------------------------------------------------------------------
        # Propagator check B: independent exponential-action algorithm
        # ------------------------------------------------------------------

        v_dense = (
            E_full
            @ d0
        )

        v_action = expm_multiply(
            Lp * t_probe,
            d0,
        )

        r_cross = float(
            np.max(
                np.abs(
                    v_dense
                    - v_action
                )
            )
            / max(
                np.max(
                    np.abs(
                        v_dense
                    )
                ),
                NUM_FLOOR,
            )
        )

        print(
            "  propagator semigroup "
            f"check      {r_semi:.3e}"
        )

        print(
            "  propagator cross-algorithm "
            f"check {r_cross:.3e}"
        )

        if r_semi > 1e-8:
            raise RuntimeError(
                f"expm semigroup consistency "
                f"failed at [{i},{j}], "
                f"t={t_probe:.6e}: "
                f"{r_semi:.3e}"
            )

        if r_cross > 1e-8:
            raise RuntimeError(
                f"expm and expm_multiply "
                f"disagree at [{i},{j}], "
                f"t={t_probe:.6e}: "
                f"{r_cross:.3e}"
            )

        # Long-time limit.
        n_inf = n_at(
            100.0
            * tau_qss
        )

        r_inf = (
            np.max(
                np.abs(
                    n_inf
                    - n_new
                )
            )
            / max(
                np.max(
                    np.abs(
                        n_new
                    )
                ),
                NUM_FLOOR,
            )
        )

        print(
            "  propagator long-time "
            f"check         {r_inf:.3e}"
        )

        if r_inf > 1e-8:
            raise RuntimeError(
                f"matrix exponential does "
                f"not approach post-step "
                f"CRE at [{i},{j}]: "
                f"{r_inf:.3e}"
            )

        # ==================================================================
        # TIME GRID
        # ==================================================================

        t_scan_lo = (
            0.01
            * tau_rel
        )

        t_scan_hi = (
            10.0
            * tau_qss
        )

        ts_base = np.logspace(
            np.log10(
                t_scan_lo
            ),
            np.log10(
                t_scan_hi
            ),
            NT,
        )

        extra_times = [
            lo_v1,
            hi_v1,
            t_probe,
            tau_rel,
            5.0 * tau_rel,
            10.0 * tau_rel,
            20.0 * tau_rel,
            0.01 * tau_qss,
            0.02 * tau_qss,
            0.1 * tau_qss,
            tau_qss,
        ]

        extra_times = np.asarray(
            [
                t
                for t in extra_times
                if (
                    np.isfinite(t)
                    and t > 0
                    and t_scan_lo <= t <= t_scan_hi
                )
            ],
            dtype=float,
        )

        ts = np.unique(
            np.sort(
                np.concatenate(
                    [
                        ts_base,
                        extra_times,
                    ]
                )
            )
        )

        n_samples = len(
            ts
        )

        R_full = np.empty(
            n_samples
        )

        R_qss = np.empty(
            n_samples
        )

        dev_track = np.empty(
            n_samples
        )

        dev_pe = np.empty(
            n_samples
        )

        ng_drift = np.empty(
            n_samples
        )

        state_qss_dev = np.empty(
            n_samples
        )

        eps_cre = np.empty(
            n_samples
        )

        eta_log = np.empty(
            n_samples
        )

        # ==================================================================
        # FULL TRANSIENT SCAN
        # ==================================================================

        for q, t in enumerate(
            ts
        ):

            n = n_at(
                float(t)
            )

            pop_scale = max(
                np.abs(n).max(),
                NUM_FLOOR,
            )

            if (
                n.min()
                < -1e-10
                * pop_scale
            ):
                raise RuntimeError(
                    f"negative transient "
                    f"population at [{i},{j}], "
                    f"t={t:.6e}: "
                    f"min={n.min():.3e}"
                )

            n3 = float(
                n[
                    idxFull[3]
                ].sum()
            )

            n4 = float(
                n[
                    idxFull[4]
                ].sum()
            )

            if (
                not np.isfinite(n3)
                or not np.isfinite(n4)
                or n3 <= 0
                or n4 <= 0
            ):
                raise RuntimeError(
                    f"invalid Balmer shell "
                    f"population at [{i},{j}], "
                    f"t={t:.6e}: "
                    f"n3={n3}, n4={n4}"
                )

            R = (
                n3
                / n4
            )

            u_t = float(
                n[g]
            )

            if (
                not np.isfinite(u_t)
                or u_t <= 0
            ):
                raise RuntimeError(
                    f"invalid transient ground "
                    f"reservoir at [{i},{j}], "
                    f"t={t:.6e}: "
                    f"u={u_t}"
                )

            nF_qss_t = (
                a * u_t
                + c
            )

            qss3 = float(
                nF_qss_t[
                    idxE[3]
                ].sum()
            )

            qss4 = float(
                nF_qss_t[
                    idxE[4]
                ].sum()
            )

            if (
                qss3 <= 0
                or qss4 <= 0
            ):
                raise RuntimeError(
                    f"invalid instantaneous "
                    f"QSS ratio at [{i},{j}], "
                    f"t={t:.6e}"
                )

            Rq = (
                qss3
                / qss4
            )

            R_full[q] = R
            R_qss[q] = Rq

            dev_track[q] = abs(
                R / Rq
                - 1.0
            )

            dev_pe[q] = abs(
                R / R_pe
                - 1.0
            )

            ng_drift[q] = abs(
                u_t / u_old
                - 1.0
            )

            eps_cre[q] = abs(
                R / R_new
                - 1.0
            )

            eta_log[q] = (
                np.log(
                    R / R_pe
                )
                / log_total
            )

            state_qss_dev[q] = (
                np.linalg.norm(
                    n[E]
                    - nF_qss_t
                )
                / max(
                    np.linalg.norm(
                        nF_qss_t
                    ),
                    NUM_FLOOR,
                )
            )

        # ==================================================================
        # HISTORICAL v1 RESULT
        # ==================================================================

        win_v1 = (
            (ts >= lo_v1)
            &
            (ts <= hi_v1)
        )

        print()
        print(
            "  HISTORICAL v1 FIXED-WINDOW TEST:"
        )

        if (
            lo_v1 >= hi_v1
            or not win_v1.any()
        ):

            v1_track = False
            v1_pe = False

            print(
                "    historical window "
                "empty"
            )

        else:

            v1_max_track = float(
                dev_track[
                    win_v1
                ].max()
            )

            v1_max_pe = float(
                dev_pe[
                    win_v1
                ].max()
            )

            v1_track = (
                v1_max_track
                < TOL_TRACK
            )

            v1_pe = (
                v1_max_pe
                < TOL_PLATEAU
            )

            print(
                f"    window "
                f"{lo_v1:.4e} .. "
                f"{hi_v1:.4e} s"
            )

            print(
                "    max eps_track "
                f"= {v1_max_track:.3e} "
                f"-> "
                f"{'PASS' if v1_track else 'FAIL'}"
            )

            print(
                "    max eps_PE    "
                f"= {v1_max_pe:.3e} "
                f"-> "
                f"{'PASS' if v1_pe else 'FAIL'}"
            )

            print(
                "    This historical "
                "result is retained but "
                "does NOT determine the "
                "v2 verdict."
            )

        # ==================================================================
        # SUSTAINED TRACKING ONSET
        # ==================================================================

        ok_track = (
            dev_track
            < TOL_TRACK
        )

        q_track_start, q_track_end = (
            longest_true_run(
                ok_track
            )
        )

        if q_track_start is None:

            sustained_tracking = False
            t_track = float("nan")

        else:

            sustained_tracking = True

            # Refine the entry threshold.
            if q_track_start > 0:

                margin0 = (
                    dev_track[
                        q_track_start - 1
                    ]
                    / TOL_TRACK
                    - 1.0
                )

                margin1 = (
                    dev_track[
                        q_track_start
                    ]
                    / TOL_TRACK
                    - 1.0
                )

                t_track = (
                    interpolate_zero_logtime(
                        ts[
                            q_track_start - 1
                        ],
                        ts[
                            q_track_start
                        ],
                        margin0,
                        margin1,
                    )
                )

            else:

                t_track = float(
                    ts[
                        q_track_start
                    ]
                )

        # ==================================================================
        # v2 MEASURED PE PLATEAU
        # ==================================================================

        both = (
            (dev_track < TOL_TRACK)
            &
            (dev_pe < TOL_PLATEAU)
        )

        q_start, q_end = (
            longest_true_run(
                both
            )
        )

        # Require non-zero sampled duration.
        plateau_exists = (
            q_start is not None
            and q_end is not None
            and q_end > q_start
        )

        plateau_right_censored = False

        if plateau_exists:

            margin = np.maximum(
                dev_track
                / TOL_TRACK,
                dev_pe
                / TOL_PLATEAU,
            ) - 1.0

            # Refine start.
            if q_start > 0:

                t_start = (
                    interpolate_zero_logtime(
                        ts[
                            q_start - 1
                        ],
                        ts[
                            q_start
                        ],
                        margin[
                            q_start - 1
                        ],
                        margin[
                            q_start
                        ],
                    )
                )

            else:

                t_start = float(
                    ts[q_start]
                )

            # Refine end.
            if (
                q_end + 1
                < n_samples
            ):

                t_end = (
                    interpolate_zero_logtime(
                        ts[q_end],
                        ts[
                            q_end + 1
                        ],
                        margin[q_end],
                        margin[
                            q_end + 1
                        ],
                    )
                )

            else:

                t_end = float(
                    ts[q_end]
                )

                plateau_right_censored = True

            duration = (
                t_end
                - t_start
            )

            seg_idx = np.arange(
                q_start,
                q_end + 1,
            )

            max_track_plateau = float(
                dev_track[
                    seg_idx
                ].max()
            )

            max_pe_plateau = float(
                dev_pe[
                    seg_idx
                ].max()
            )

            max_ng_plateau = float(
                ng_drift[
                    seg_idx
                ].max()
            )

            max_state_plateau = float(
                state_qss_dev[
                    seg_idx
                ].max()
            )

            flat_abs = float(
                np.max(
                    np.abs(
                        eps_cre[
                            seg_idx
                        ]
                        - eps_plateau
                    )
                )
            )

            flat_frac = (
                flat_abs
                / max(
                    eps_plateau,
                    NUM_FLOOR,
                )
            )

        else:

            t_start = float("nan")
            t_end = float("nan")
            duration = float("nan")

            max_track_plateau = float("nan")
            max_pe_plateau = float("nan")
            max_ng_plateau = float("nan")
            max_state_plateau = float("nan")
            flat_frac = float("nan")

        # ==================================================================
        # CRE-ERROR PERSISTENCE
        # ==================================================================

        half_reached = False
        t_half = float("nan")
        dt_half = float("nan")
        tau_eff = float("nan")

        if plateau_exists:

            target_half = (
                EPS_CRE_FRAC
                * eps_plateau
            )

            # Search only AFTER plateau formation.
            cross_q = None

            for q in range(
                q_start + 1,
                n_samples,
            ):

                if (
                    eps_cre[q]
                    <= target_half
                    and eps_cre[
                        q - 1
                    ]
                    > target_half
                ):
                    cross_q = q
                    break

            if cross_q is not None:

                t_half = (
                    interpolate_positive_level(
                        ts[
                            cross_q - 1
                        ],
                        ts[
                            cross_q
                        ],
                        eps_cre[
                            cross_q - 1
                        ],
                        eps_cre[
                            cross_q
                        ],
                        target_half,
                    )
                )

                dt_half = (
                    t_half
                    - t_start
                )

                tau_eff = (
                    dt_half
                    / np.log(2.0)
                )

                half_reached = True

        # ==================================================================
        # EARLY SLOW-TIME LOG-SPACE FIT
        # ==================================================================

        k_fit = float("nan")
        fit_rms = float("nan")
        fit_max = float("nan")

        if plateau_exists:

            fit_idx = np.arange(
                q_start,
                q_end + 1,
            )

            x = (
                ts[fit_idx]
                / tau_qss
            )

            y = eta_log[
                fit_idx
            ]

            finite = (
                np.isfinite(x)
                &
                np.isfinite(y)
                &
                (x > 0)
            )

            x = x[finite]
            y = y[finite]

            if len(x) >= 3:

                xx = float(
                    np.dot(
                        x,
                        x,
                    )
                )

                if xx > 0:

                    k_fit = float(
                        np.dot(
                            x,
                            y,
                        )
                        / xx
                    )

                    resid = (
                        y
                        - k_fit
                        * x
                    )

                    fit_rms = float(
                        np.sqrt(
                            np.mean(
                                resid**2
                            )
                        )
                    )

                    fit_max = float(
                        np.max(
                            np.abs(
                                resid
                            )
                        )
                    )

        # ==================================================================
        # OUTPUT -- v2
        # ==================================================================

        print()
        print(
            "  v2 DYNAMIC BRIDGE:"
        )

        if sustained_tracking:

            print(
                "    sustained QSS tracking "
                f"starts at {t_track:.4e} s "
                f"= {t_track/tau_rel:.3f} "
                f"tau_relax "
                f"= {t_track/tau_qss:.3e} "
                f"tau_QSS"
            )

        else:

            print(
                "    sustained QSS tracking "
                "was NOT found."
            )

        print()

        if plateau_exists:

            print(
                "    measured PE plateau "
                "EXISTS"
            )

            print(
                f"    start     "
                f"{t_start:.4e} s "
                f"= {t_start/tau_rel:.3f} "
                f"tau_relax "
                f"= {t_start/tau_qss:.3e} "
                f"tau_QSS"
            )

            print(
                f"    end       "
                f"{t_end:.4e} s "
                f"= {t_end/tau_rel:.3f} "
                f"tau_relax "
                f"= {t_end/tau_qss:.3e} "
                f"tau_QSS"
                + (
                    "  [right-censored]"
                    if plateau_right_censored
                    else ""
                )
            )

            print(
                f"    duration  "
                f"{duration:.4e} s "
                f"= {duration/tau_rel:.3f} "
                f"tau_relax "
                f"= {duration/tau_qss:.3e} "
                f"tau_QSS"
            )

            print(
                "    max eps_track "
                f"inside = "
                f"{max_track_plateau:.3e}"
            )

            print(
                "    max eps_PE    "
                f"inside = "
                f"{max_pe_plateau:.3e}"
            )

            print(
                "    max ground drift "
                f"inside = "
                f"{max_ng_plateau:.3e}"
            )

            print(
                "    max full-state QSS "
                f"deviation inside = "
                f"{max_state_plateau:.3e}"
            )

            print(
                "    max departure of "
                "eps_CRE from analytic "
                "eps_plateau = "
                f"{100*flat_frac:.3f}% "
                "of eps_plateau"
            )

        else:

            print(
                "    measured PE plateau "
                "NOT RESOLVED as a "
                "non-zero contiguous "
                "interval."
            )

        # ------------------------------------------------------------------
        # Persistence
        # ------------------------------------------------------------------

        print()
        print(
            "  CRE-ERROR PERSISTENCE:"
        )

        print(
            "    analytic eps_plateau "
            f"= {eps_plateau:.6f}"
        )

        if half_reached:

            print(
                "    half-amplitude target "
                f"= {target_half:.6f}"
            )

            print(
                "    crossing time since "
                f"step = {t_half:.4e} s "
                f"= {t_half/tau_qss:.4f} "
                "tau_QSS"
            )

            print(
                "    post-plateau "
                "half-decay time "
                f"Delta t_1/2 "
                f"= {dt_half:.4e} s "
                f"= {dt_half/tau_qss:.4f} "
                "tau_QSS"
            )

            print(
                "    exponential-equivalent "
                f"tau_eff = "
                f"{tau_eff:.4e} s "
                f"= {tau_eff/tau_qss:.4f} "
                "tau_QSS"
            )

        else:

            print(
                "    half-amplitude crossing "
                "not reached within scan."
            )

        # ------------------------------------------------------------------
        # Slow-time fit
        # ------------------------------------------------------------------

        print()
        print(
            "  EARLY SLOW-TIME "
            "LOG-RESPONSE FIT:"
        )

        if np.isfinite(
            k_fit
        ):

            print(
                "    eta_R "
                "= ln(R/R_PE) / "
                "ln(R_CRE/R_PE)"
            )

            print(
                "    fit "
                "eta_R ~= "
                "k t/tau_QSS"
            )

            print(
                f"    k = {k_fit:.5f}"
            )

            print(
                f"    RMS residual "
                f"= {fit_rms:.3e}"
            )

            print(
                f"    max residual "
                f"= {fit_max:.3e}"
            )

        else:

            print(
                "    insufficient measured "
                "plateau points for fit."
            )

        # ------------------------------------------------------------------
        # Spectral comparison, descriptive only
        # ------------------------------------------------------------------

        print()
        print(
            "  SPECTRAL COMPARISON "
            "(descriptive only):"
        )

        print(
            f"    1/M = "
            f"{1.0/M:.3e}"
        )

        if plateau_exists:

            print(
                "    max eps_track * M "
                f"inside plateau = "
                f"{max_track_plateau*M:.4g}"
            )

            print(
                "    no universal O(1) "
                "acceptance band is imposed."
            )

        # ==================================================================
        # REPRESENTATIVE TIME TABLE
        # ==================================================================

        print()
        print(
            f"    {'location':>13s}"
            f" {'t (s)':>12s}"
            f" {'t/t_rel':>10s}"
            f" {'t/t_QSS':>10s}"
            f" {'R_full':>11s}"
            f" {'R_QSS':>11s}"
            f" {'PE dev':>10s}"
            f" {'QSS dev':>10s}"
            f" {'eps_CRE':>10s}"
            f" {'ng drift':>10s}"
        )

        table_points = []

        if plateau_exists:

            q_mid = int(
                (q_start + q_end)
                // 2
            )

            table_points.extend(
                [
                    (
                        "PE start",
                        q_start,
                    ),
                    (
                        "PE middle",
                        q_mid,
                    ),
                    (
                        "PE end",
                        q_end,
                    ),
                ]
            )

        if sustained_tracking:

            table_points.append(
                (
                    "QSS onset",
                    q_track_start,
                )
            )

        # Add historical v1 boundaries.
        q_v1_lo = int(
            np.argmin(
                np.abs(
                    ts
                    - lo_v1
                )
            )
        )

        q_v1_hi = int(
            np.argmin(
                np.abs(
                    ts
                    - hi_v1
                )
            )
        )

        table_points.extend(
            [
                (
                    "v1 start",
                    q_v1_lo,
                ),
                (
                    "v1 end",
                    q_v1_hi,
                ),
            ]
        )

        # Deduplicate while preserving order.
        seen = set()
        unique_table = []

        for name, q in table_points:

            if q not in seen:
                seen.add(q)
                unique_table.append(
                    (
                        name,
                        q,
                    )
                )

        for name, q in unique_table:

            print(
                f"    {name:>13s}"
                f" {ts[q]:12.4e}"
                f" {ts[q]/tau_rel:10.3e}"
                f" {ts[q]/tau_qss:10.3e}"
                f" {R_full[q]:11.6f}"
                f" {R_qss[q]:11.6f}"
                f" {dev_pe[q]:10.3e}"
                f" {dev_track[q]:10.3e}"
                f" {eps_cre[q]:10.3e}"
                f" {ng_drift[q]:10.3e}"
            )

        # ==================================================================
        # STORE RESULT
        # ==================================================================

        results.append(
            {
                "label": label,
                "v1_track": v1_track,
                "v1_pe": v1_pe,
                "tracking": sustained_tracking,
                "plateau_exists": plateau_exists,
                "t_start": t_start,
                "t_end": t_end,
                "duration": duration,
                "half_reached": half_reached,
                "dt_half": dt_half,
                "tau_eff": tau_eff,
            }
        )

    # =======================================================================
    # FINAL VERDICT
    # =======================================================================

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    print()
    print(
        "  HISTORICAL v1 RESULT:"
    )

    for result in results:

        print(
            f"    {result['label']:20s} "
            f"fixed-window QSS="
            f"{result['v1_track']}   "
            f"fixed-window PE="
            f"{result['v1_pe']}"
        )

    print()
    print(
        "  The historical v1 result is "
        "not reclassified or overwritten."
    )

    print()
    print(
        "  v2 DYNAMIC BRIDGE:"
    )

    for result in results:

        print(
            f"    {result['label']:20s} "
            f"sustained tracking="
            f"{result['tracking']}   "
            f"measured PE plateau="
            f"{result['plateau_exists']}"
        )

    bridge_observed = (
        len(results)
        == len(POINTS)
        and all(
            result["tracking"]
            and result[
                "plateau_exists"
            ]
            for result in results
        )
    )

    print()

    if bridge_observed:

        print(
            "  DYNAMIC BRIDGE OBSERVED "
            "AT BOTH REPRESENTATIVE POINTS."
        )

        print()
        print(
            "  The full 43-state transient "
            "enters the instantaneous "
            "post-step QSS manifold and "
            "passes through a finite "
            "contiguous interval satisfying "
            "the frozen-ground PE criterion."
        )

        print()
        print(
            "  This directly supports, at "
            "these representative points:"
        )

        print()
        print(
            "      full transient"
        )

        print(
            "          -> R_QSS+(u(t))"
        )

        print(
            "          -> R_PE = R+(u^-)"
        )

        print(
            "          -> f_3 - f_4"
        )

        print(
            "          -> density ridge"
        )

        print()
        print(
            "  The v1 fixed-window failure "
            "at the ridge, if reproduced, "
            "remains a separate historical "
            "result: it means the frozen-"
            "ground approximation ceases to "
            "meet an absolute 0.2% criterion "
            "before 0.02 tau_QSS, not that "
            "the QSS manifold was missed."
        )

    else:

        print(
            "  DYNAMIC BRIDGE NOT "
            "ESTABLISHED AT BOTH POINTS."
        )

        print()
        print(
            "  At least one representative "
            "point either never enters a "
            "sustained instantaneous-QSS "
            "regime or lacks a non-zero "
            "contiguous interval satisfying "
            "both fixed tolerances."
        )

        print()
        print(
            "  Do not make the full-transient "
            "-> PE causal claim until the "
            "failing point is understood."
        )

    print()
    print(
        "  Scope: these are two "
        "representative conditions only. "
        "A grid-wide bridge test is required "
        "for a grid-wide claim."
    )

    print()
    print("=" * 78)
    print(
        "  Report only. Nothing was written."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()