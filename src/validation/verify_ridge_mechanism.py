"""
verify_ridge_mechanism.py
=========================
T1. eps_plateau has an interior maximum in DENSITY at column j = 3,
ne = 1.931e13 cm^-3, at every temperature from 2 to 9 eV, while in temperature
it falls monotonically. Why that column?

HISTORY
-------
v1 is WITHDRAWN and retained only as:
    verify_ridge_mechanism_v1_WITHDRAWN_20260824.py

v1 mixed a pre-step b_1 with a post-step Saha coefficient. Its horizontal
coordinate was therefore wrong by Z+/Z-, and its displacement contained the
spurious term ln(Z+/Z-), comparable to the signal being measured.

The present script avoids that problem entirely by working internally in

    u := n_g / n_i,

the ground-state population per unit ion density. The pipeline normalises the
fixed ion reservoir to n_i = 1, so numerically u equals the ground component
returned by solve(L, -S), but conceptually it remains n_g/n_i.

--------------------------------------------------------------------------
THE EXACT STRUCTURE, from Chapter 3
--------------------------------------------------------------------------
For the POST-step operator,

    n_F = n_i (a+ u + c+),

with

    a+ = -L_FF+^-1 L_Fg+,
    c+ = -L_FF+^-1 S_F+,
    u  = n_g / n_i.

For shell m,

    n_m / n_i = a_m+ u + c_m+.

Therefore

    R+(u) = (a_3+ u + c_3+) / (a_4+ u + c_4+),

and

    S+(u) := d ln R+ / d ln u
           = f_3+(u) - f_4+(u),

where

    f_m+(u) = a_m+ u / (a_m+ u + c_m+).

The partial-equilibrium plateau freezes the ground reservoir at its PRE-step
physical value while allowing the excited manifold to equilibrate under the
POST-step operator:

    u^-     = n_g^- / n_i,
    u^+     = n_g^+ / n_i,

    R^PE       = R+(u^-),
    R^CRE_new  = R+(u^+).

Hence, with no approximation,

    ln(R^PE / R^CRE_new)
        = INT_{ln u+}^{ln u-} S+(u) d ln u
        = Sbar * Dln,

where

    Dln  := ln(u^- / u^+),

    Sbar := ln(R^PE / R^CRE_new) / Dln

is the EXACT finite-interval mean logarithmic sensitivity.

Thus

    eps_plateau
        = |R^PE/R^CRE_new - 1|
        = |exp(Sbar * Dln) - 1|.

--------------------------------------------------------------------------
WHAT IS AND IS NOT BEING TESTED
--------------------------------------------------------------------------
The identity above is not an independent model validation.

Likewise,

    |S+(u^-)| |Dln|

is merely the local, first-order approximation to the exact integral and is
not used as evidence for the mechanism.

The analysis here is an ATTRIBUTION analysis of the exact finite-step
decomposition:

    log-response = mean sensitivity x ground-reservoir displacement.

The question is whether the density dependence of the plateau-error ridge is
carried primarily by

    |Sbar|

or by

    |Dln|.

The physical interpretation of |Sbar| is independently wired to Chapter 3 by
integrating the kernel f_3 - f_4 over the traversed interval and checking that
it reproduces the endpoint-defined Sbar.

--------------------------------------------------------------------------
PREREGISTRATION
--------------------------------------------------------------------------
H:
    The density ridge is SENSITIVITY-DOMINATED: the column maximising
    eps_plateau is also the column maximising exact |Sbar| because the frozen
    ground reservoir traverses the mixed-supply region where the two shells
    differ most strongly in ground-fed fraction.

ACCEPT H if, over rows with starting Te in [2, 9] eV and over ALL density
columns,

    argmax_j |Sbar| == argmax_j eps

in at least 80% of rows,

AND

    argmax_j |Dln| == argmax_j eps

in fewer than 50% of rows.

REJECT H in favour of a DISPLACEMENT-DOMINATED ridge if those conditions swap.

INCONCLUSIVE if both agreement fractions exceed 50%.

Otherwise neither attribution is established cleanly and the two factors are
reported separately.

The row-wise overlap of the two criteria is descriptive only; it does not
determine the verdict.

--------------------------------------------------------------------------
HARD / WIRING CHECKS
--------------------------------------------------------------------------
1. Channel residuals:
       L_FF a + L_Fg = 0,
       L_FF c + S_F  = 0.

2. a and c must be non-negative up to numerical tolerance.

3. Full post-step CRE populations must satisfy
       n_F^CRE = a u^+ + c
   and the resulting shell ratio must agree with the independently solved
   43-state CRE ratio.

4. Chapter 3 kernel wiring:
       Sbar(endpoint ratio)
   must agree with
       (1/Dln) INT (f_3-f_4) d ln u.

5. Chapter 3 gives
       |S+(u)| <= tanh(|Delta+|/4),
   so necessarily
       |ln(R^PE/R^CRE_new)|
       <= tanh(|Delta+|/4) |Dln|.

Violation of any of these means the implementation is internally inconsistent.

--------------------------------------------------------------------------
STEP-SIZE ROBUSTNESS
--------------------------------------------------------------------------
Both STEP = 1 and STEP = 4 are run.

Their ridge locations and factor maxima are compared only over starting
temperature rows accessible to BOTH step sizes.

A ridge that moves with step amplitude is not automatically an artifact:
changing the step changes both the traversed reservoir interval and the
post-step operator S+(u). Such movement would instead establish genuine
finite-amplitude dependence.

--------------------------------------------------------------------------
REPORT
--------------------------------------------------------------------------
Report only. Writes nothing.

    python src/validation/verify_ridge_mechanism.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Imports / paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

for p in (
    HERE,
    HERE.parent / "validation",
    HERE.parent / "analysis",
):
    sys.path.insert(0, str(p))

from cr_context import CRContext  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHI_H = 13.605693122994
H_PLANCK = 6.62607015e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19

TE_LO = 2.0
TE_HI = 9.0

STEPS = (1, 4)

ACCEPT_FRAC = 0.80

# Printed for local inspection only. The actual test uses ALL density columns.
NEIGHBOURHOOD = (2, 3, 4, 5)

# Dense quadrature is cheap because the integrand is an analytic two-shell
# response, not another CR solve.
NQUAD = 2001

NUM_FLOOR = 1e-300


# ---------------------------------------------------------------------------
# Saha coefficient -- DISPLAY ONLY in this analysis
# ---------------------------------------------------------------------------

def saha_Z(p: int, te_ev: float) -> float:
    """
    Saha-Boltzmann coefficient for hydrogen shell p [cm^3], Te in eV.

    Internally the mechanism analysis works in u = n_g/n_i and therefore does
    not need Z. Z appears only when converting u to a post-step b_1 coordinate
    for human-readable output.
    """
    return (
        1e6
        * p**2
        * (
            H_PLANCK**2
            / (2.0 * np.pi * M_E * EV_TO_J * te_ev)
        ) ** 1.5
        * np.exp((CHI_H / p**2) / te_ev)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ctx = CRContext.load()

    root = ctx.root

    L = ctx.L_grid
    Te = ctx.te_grid
    ne = ctx.ne_grid

    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    S = np.load(S_path)

    nv = np.asarray(ctx.n_values)

    g = int(ctx.ground_index)

    E = np.array(
        [k for k in range(ctx.n_states) if k != g],
        dtype=int,
    )

    posE = {
        state_idx: local_idx
        for local_idx, state_idx in enumerate(E)
    }

    # Local indices into the excited block.
    idxE = {
        m: [
            posE[s]
            for s in np.where(nv == m)[0]
        ]
        for m in (3, 4)
    }

    # Full 43-state indices.
    idxFull = {
        m: np.where(nv == m)[0]
        for m in (3, 4)
    }

    nT = len(Te)
    nN = len(ne)

    # NumPy 1.x / 2.x compatibility.
    trapz = (
        np.trapezoid
        if hasattr(np, "trapezoid")
        else np.trapz
    )

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)

    for rel in (
        "data/processed/cr_matrix/L_grid.npy",
        "data/processed/cr_matrix/S_grid.npy",
    ):
        f = root / rel
        sha = hashlib.sha256(f.read_bytes()).hexdigest()

        print(f"  {rel}")
        print(f"      sha256 {sha}")

    print(
        f"  all {nN} density columns tested; "
        f"starting rows {TE_LO} <= Te <= {TE_HI} eV"
    )

    print(
        f"  acceptance threshold: "
        f"{100 * ACCEPT_FRAC:.0f}% row-wise argmax agreement"
    )

    print(
        "  v1 withdrawn for a pre/post Saha-coordinate error; "
        "current analysis uses u = n_g/n_i throughout."
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def steady(i: int, j: int) -> np.ndarray:
        """
        Full 43-state CR-equilibrium solution per unit ion density.

        Because S is the source per unit ion density, solve(L, -S) returns
        n/n_i. In particular n[g] = n_g/n_i = u.
        """
        n = np.linalg.solve(
            L[i, j],
            -S[i, j],
        )

        if not np.all(np.isfinite(n)):
            raise RuntimeError(
                f"non-finite CRE population at [{i},{j}]"
            )

        if n[g] <= 0:
            raise RuntimeError(
                f"non-positive ground population at [{i},{j}]: "
                f"u={n[g]:.6e}"
            )

        return n

    def post_channels(
        ip: int,
        j: int,
    ):
        """
        Post-step response vectors

            a = -L_EE^-1 L_Eg
            c = -L_EE^-1 S_E

        and their n=3,4 shell sums.
        """

        Lij = L[ip, j]

        LEE = Lij[np.ix_(E, E)]

        LEg = Lij[
            np.ix_(E, [g])
        ].ravel()

        SE = S[ip, j][E]

        a = np.linalg.solve(
            LEE,
            -LEg,
        )

        c = np.linalg.solve(
            LEE,
            -SE,
        )

        # ---------------------------------------------------------------
        # Independent channel-solve residuals
        # ---------------------------------------------------------------

        ra = (
            np.linalg.norm(LEE @ a + LEg)
            / max(np.linalg.norm(LEg), NUM_FLOOR)
        )

        rc = (
            np.linalg.norm(LEE @ c + SE)
            / max(np.linalg.norm(SE), NUM_FLOOR)
        )

        if ra > 1e-10 or rc > 1e-10:
            raise RuntimeError(
                f"channel solve failed at [{ip},{j}]: "
                f"res_a={ra:.3e}, res_c={rc:.3e}"
            )

        # ---------------------------------------------------------------
        # Physical non-negativity check
        # ---------------------------------------------------------------

        tol_a = (
            1e-10
            * max(np.abs(a).max(), NUM_FLOOR)
        )

        tol_c = (
            1e-10
            * max(np.abs(c).max(), NUM_FLOOR)
        )

        if a.min() < -tol_a:
            raise RuntimeError(
                f"negative ground-fed response at [{ip},{j}]: "
                f"min(a)={a.min():.3e}"
            )

        if c.min() < -tol_c:
            raise RuntimeError(
                f"negative recombination-fed response at [{ip},{j}]: "
                f"min(c)={c.min():.3e}"
            )

        am = {
            m: a[idxE[m]].sum()
            for m in (3, 4)
        }

        cm = {
            m: c[idxE[m]].sum()
            for m in (3, 4)
        }

        return am, cm, a, c

    # Rows reachable by the largest step.
    common_rows = [
        i
        for i in range(nT - max(STEPS))
        if TE_LO <= Te[i] <= TE_HI
    ]

    # Save row-wise argmax locations for cross-step comparison.
    across = {}

    # =======================================================================
    # LOOP OVER STEP SIZE
    # =======================================================================

    for k in STEPS:

        print()
        print("=" * 78)

        pct_step = 100.0 * (
            Te[k] / Te[0] - 1.0
        )

        print(
            f"STEP = {k} grid interval(s)   "
            f"(+{pct_step:.2f}% in Te)"
        )

        print("=" * 78)

        rows = [
            i
            for i in range(nT - k)
            if TE_LO <= Te[i] <= TE_HI
        ]

        n_r = len(rows)

        # ------------------------------------------------------------------
        # Storage
        # ------------------------------------------------------------------

        eps = np.zeros((n_r, nN))

        # Exact endpoint-defined mean sensitivity.
        Sbar = np.zeros((n_r, nN))

        # Independently integrated Chapter-3 kernel.
        Sker = np.zeros((n_r, nN))

        # Reservoir displacement ln(u-/u+).
        Dln = np.zeros((n_r, nN))

        # Local sensitivity at the frozen old reservoir under the new operator.
        Sloc = np.zeros((n_r, nN))

        # Exact log-ratio.
        lnr = np.zeros((n_r, nN))

        # Diagnostic / display only.
        b1_u = np.zeros((n_r, nN))
        b1_pk = np.zeros((n_r, nN))

        tiny = np.zeros(
            (n_r, nN),
            dtype=bool,
        )

        # ===================================================================
        # GRID CALCULATION
        # ===================================================================

        for r, i in enumerate(rows):

            for j in range(nN):

                ip = i + k

                # -----------------------------------------------------------
                # POST-step response structure
                # -----------------------------------------------------------

                am, cm, a, c = post_channels(
                    ip,
                    j,
                )

                # -----------------------------------------------------------
                # PRE- and POST-step full CRE states
                # -----------------------------------------------------------

                n_old = steady(i, j)
                n_new = steady(ip, j)

                # u := n_g/n_i.
                u_old = float(n_old[g])
                u_new = float(n_new[g])

                # -----------------------------------------------------------
                # Independent reduced-vs-full CRE superposition check
                # -----------------------------------------------------------

                nE_reduced_new = (
                    a * u_new + c
                )

                sup = (
                    np.max(
                        np.abs(
                            nE_reduced_new
                            - n_new[E]
                        )
                    )
                    / max(
                        np.max(np.abs(n_new[E])),
                        NUM_FLOOR,
                    )
                )

                if sup > 1e-10:
                    raise RuntimeError(
                        f"CRE superposition failed at [{ip},{j}]: "
                        f"relative deviation={sup:.3e}"
                    )

                R_new_reduced = (
                    am[3] * u_new + cm[3]
                ) / (
                    am[4] * u_new + cm[4]
                )

                R_new_full = (
                    n_new[idxFull[3]].sum()
                    / n_new[idxFull[4]].sum()
                )

                rel_R_new = abs(
                    R_new_reduced / R_new_full
                    - 1.0
                )

                if rel_R_new > 1e-10:
                    raise RuntimeError(
                        f"reduced/full R_CRE disagreement at "
                        f"[{ip},{j}]: "
                        f"R_reduced={R_new_reduced:.12e}, "
                        f"R_full={R_new_full:.12e}, "
                        f"rel={rel_R_new:.3e}"
                    )

                # -----------------------------------------------------------
                # Partial-equilibrium plateau
                #
                # New excited operator, OLD physical ground reservoir.
                # -----------------------------------------------------------

                R_pe = (
                    am[3] * u_old + cm[3]
                ) / (
                    am[4] * u_old + cm[4]
                )

                R_new = R_new_full

                # -----------------------------------------------------------
                # Finiteness / positivity BEFORE logs
                # -----------------------------------------------------------

                vals = np.array(
                    [
                        u_old,
                        u_new,
                        am[3],
                        am[4],
                        cm[3],
                        cm[4],
                        R_pe,
                        R_new,
                    ],
                    dtype=float,
                )

                if (
                    not np.all(np.isfinite(vals))
                    or np.any(vals <= 0)
                ):
                    raise RuntimeError(
                        f"non-finite or non-positive quantity "
                        f"at [{i},{j}]: "
                        f"{vals}"
                    )

                # -----------------------------------------------------------
                # Exact displacement and observable response
                # -----------------------------------------------------------

                d = np.log(
                    u_old / u_new
                )

                lr = np.log(
                    R_pe / R_new
                )

                # All experiments considered here are upward Te steps.
                # Do not silently assume the expected sign.
                if d <= 0:
                    raise RuntimeError(
                        f"heating did not deplete the ground "
                        f"reservoir at [{i},{j}]: "
                        f"u_old={u_old:.6e}, "
                        f"u_new={u_new:.6e}, "
                        f"Dln={d:.6e}"
                    )

                Dln[r, j] = d
                lnr[r, j] = lr

                eps[r, j] = abs(
                    R_pe / R_new - 1.0
                )

                # -----------------------------------------------------------
                # Chapter-3 sensitivity kernel
                # -----------------------------------------------------------

                Delta = np.log(
                    (am[3] / am[4])
                    / (cm[3] / cm[4])
                )

                s_peak = np.tanh(
                    abs(Delta) / 4.0
                )

                def f_of(
                    m: int,
                    uu,
                ):
                    return (
                        am[m] * uu
                        / (
                            am[m] * uu
                            + cm[m]
                        )
                    )

                Sloc[r, j] = abs(
                    f_of(3, u_old)
                    - f_of(4, u_old)
                )

                # -----------------------------------------------------------
                # Exact mean sensitivity:
                # 1. endpoint route
                # 2. independent kernel-integration route
                # -----------------------------------------------------------

                if abs(d) < 1e-12:

                    um = np.sqrt(
                        u_old * u_new
                    )

                    s_mid = (
                        f_of(3, um)
                        - f_of(4, um)
                    )

                    Sbar[r, j] = s_mid
                    Sker[r, j] = s_mid

                    tiny[r, j] = True

                else:

                    # Endpoint route
                    Sbar[r, j] = (
                        lr / d
                    )

                    # Independent Chapter-3 kernel route
                    lg = np.linspace(
                        np.log(u_new),
                        np.log(u_old),
                        NQUAD,
                    )

                    uq = np.exp(lg)

                    Sq = (
                        am[3] * uq
                        / (
                            am[3] * uq
                            + cm[3]
                        )
                        -
                        am[4] * uq
                        / (
                            am[4] * uq
                            + cm[4]
                        )
                    )

                    Sker[r, j] = (
                        trapz(Sq, lg)
                        / d
                    )

                # -----------------------------------------------------------
                # Display-only b1 coordinate
                #
                # IMPORTANT: both use the SAME POST-step Z.
                # -----------------------------------------------------------

                Zn = (
                    saha_Z(
                        1,
                        Te[ip],
                    )
                    * ne[j]
                )

                b1_u[r, j] = (
                    u_old / Zn
                )

                b1_pk[r, j] = (
                    np.sqrt(
                        (cm[3] / am[3])
                        * (cm[4] / am[4])
                    )
                    / Zn
                )

                if not np.isclose(
                    b1_u[r, j] * Zn,
                    u_old,
                    rtol=1e-12,
                    atol=0.0,
                ):
                    raise RuntimeError(
                        f"b1/u coordinate reconstruction "
                        f"failed at [{i},{j}]"
                    )

                # -----------------------------------------------------------
                # Hard Chapter-3 sensitivity bound
                # -----------------------------------------------------------

                bound = (
                    s_peak
                    * abs(d)
                )

                if (
                    abs(lr)
                    > bound * (1.0 + 1e-9)
                ):
                    raise RuntimeError(
                        f"tanh bound violated at [{i},{j}]: "
                        f"|ln(R_PE/R_CRE)|={abs(lr):.6e} > "
                        f"S_peak|Dln|={bound:.6e}"
                    )

        # ===================================================================
        # NUMERICAL / WIRING CHECKS
        # ===================================================================

        # ------------------------------------------------------------------
        # Arithmetic reconstruction
        #
        # Deliberately tautological for non-tiny points because
        # Sbar := lnr/Dln.
        # ------------------------------------------------------------------

        mask = ~tiny

        if np.any(mask):

            recon = np.abs(
                np.expm1(
                    Sbar[mask]
                    * Dln[mask]
                )
            )

            eps_mask = eps[mask]

            dev = (
                np.abs(
                    recon - eps_mask
                )
                / np.maximum(
                    eps_mask,
                    NUM_FLOOR,
                )
            )

            max_recon_dev = (
                dev.max()
            )

        else:

            max_recon_dev = 0.0

        print(
            "  algebraic reconstruction "
            "(tautological arithmetic check only): "
            f"max rel dev {max_recon_dev:.2e}"
        )

        # ------------------------------------------------------------------
        # Independent kernel wiring check
        # ------------------------------------------------------------------

        # At tiny-displacement points Sbar and Sker are BOTH assigned s_mid,
        # so kdev is identically zero there and the comparison is vacuous.
        # Including them would understate max_kdev and overstate how many
        # points the check actually constrained. Mask them out.
        kmask = ~tiny
        n_constrained = int(kmask.sum())

        den = np.maximum(
            np.maximum(
                np.abs(Sbar),
                np.abs(Sker),
            ),
            1e-14,
        )

        kdev = np.where(
            kmask,
            np.abs(Sker - Sbar) / den,
            np.nan,
        )

        if n_constrained:

            max_kdev = float(
                np.nanmax(kdev)
            )

        else:

            max_kdev = 0.0

        print(
            "  WIRING CHECK "
            "Sbar(endpoint) vs Sbar(kernel integral): "
            f"max rel dev {max_kdev:.2e} "
            f"over {n_constrained} of {tiny.size} points "
            f"({int(tiny.sum())} tiny, vacuous, excluded)"
        )

        if n_constrained == 0:

            raise RuntimeError(
                "every point used the tiny-displacement limit, so the "
                "kernel wiring check constrained nothing. The step is too "
                "small to measure anything at this grid resolution."
            )

        if max_kdev > 1e-6:

            ij = np.unravel_index(
                int(np.nanargmax(kdev)),
                kdev.shape,
            )

            raise RuntimeError(
                "Chapter-3 kernel does not reproduce "
                "the endpoint mean sensitivity: "
                f"max rel dev={max_kdev:.3e} "
                f"at local row/column {ij}"
            )

        print(
            f"  tanh bound honoured at all "
            f"{n_r * nN} points; "
            f"{int(tiny.sum())} point(s) used "
            f"the tiny-displacement limit"
        )

        # ===================================================================
        # RIDGE LOCATIONS
        # ===================================================================

        j_eps = np.argmax(
            eps,
            axis=1,
        )

        j_logerr = np.argmax(
            np.abs(lnr),
            axis=1,
        )

        j_sb = np.argmax(
            np.abs(Sbar),
            axis=1,
        )

        j_dl = np.argmax(
            np.abs(Dln),
            axis=1,
        )

        j_loc = np.argmax(
            Sloc,
            axis=1,
        )

        # Exact decomposition lives in logarithmic response.
        same_log = int(
            (
                j_logerr
                == j_eps
            ).sum()
        )

        print()
        print(
            "  argmax ordinary eps == "
            "argmax |Delta ln R| : "
            f"{same_log}/{n_r} "
            f"({100 * same_log / n_r:.0f}%)"
        )

        if same_log != n_r:
            print(
                "  WARNING: ordinary relative-error and "
                "logarithmic-error ridges differ on some rows."
            )
            print(
                "  The exact Sbar*Dln factorisation is in log space; "
                "those rows require separate interpretation."
            )

        # ------------------------------------------------------------------
        # Row-wise argmax agreement
        # ------------------------------------------------------------------

        a_sb = int(
            (
                j_sb == j_eps
            ).sum()
        )

        a_dl = int(
            (
                j_dl == j_eps
            ).sum()
        )

        both = int(
            (
                (j_sb == j_eps)
                &
                (j_dl == j_eps)
            ).sum()
        )

        print()
        print(
            f"  rows tested {n_r}   "
            f"(Te {Te[rows[0]]:.3f} "
            f"to {Te[rows[-1]]:.3f} eV)"
        )

        print(
            "  argmax eps == argmax |Sbar| : "
            f"{a_sb}/{n_r} "
            f"({100 * a_sb / n_r:.0f}%)"
        )

        print(
            "  argmax eps == argmax |Dln|  : "
            f"{a_dl}/{n_r} "
            f"({100 * a_dl / n_r:.0f}%)"
        )

        print(
            "  both simultaneously          : "
            f"{both}/{n_r}"
        )

        def mode(v):
            return int(
                np.bincount(
                    v,
                    minlength=nN,
                ).argmax()
            )

        print(
            "  modal argmax:"
            f" eps j={mode(j_eps)},"
            f" |Sbar| j={mode(j_sb)},"
            f" |Dln| j={mode(j_dl)},"
            f" local |f3-f4| j={mode(j_loc)}"
        )

        # ===================================================================
        # CONTRAST / DYNAMIC RANGE
        # ===================================================================

        C_S = np.array(
            [
                abs(
                    Sbar[
                        r,
                        j_eps[r],
                    ]
                )
                /
                max(
                    np.median(
                        np.abs(Sbar[r])
                    ),
                    NUM_FLOOR,
                )
                for r in range(n_r)
            ]
        )

        C_D = np.array(
            [
                abs(
                    Dln[
                        r,
                        j_eps[r],
                    ]
                )
                /
                max(
                    np.median(
                        np.abs(Dln[r])
                    ),
                    NUM_FLOOR,
                )
                for r in range(n_r)
            ]
        )

        rng_S = np.array(
            [
                np.abs(Sbar[r]).max()
                /
                max(
                    np.abs(Sbar[r]).min(),
                    NUM_FLOOR,
                )
                for r in range(n_r)
            ]
        )

        rng_D = np.array(
            [
                np.abs(Dln[r]).max()
                /
                max(
                    np.abs(Dln[r]).min(),
                    NUM_FLOOR,
                )
                for r in range(n_r)
            ]
        )

        print(
            "  contrast at argmax(eps): "
            f"|Sbar| {np.median(C_S):.3f}x median, "
            f"|Dln| {np.median(C_D):.3f}x median"
        )

        print(
            "  dynamic range across ne: "
            f"|Sbar| {np.median(rng_S):.2f}x, "
            f"|Dln| {np.median(rng_D):.2f}x"
        )

        print(
            "  -> a factor with dynamic range near unity "
            "cannot be shaping a strong ridge merely because "
            "its argmax coincides with eps."
        )

        # ------------------------------------------------------------------
        # Correlation between the two factors across density
        # ------------------------------------------------------------------

        cors = []

        for r in range(n_r):

            aa = np.abs(
                Sbar[r]
            )

            bb = np.abs(
                Dln[r]
            )

            if (
                aa.std() > 0
                and bb.std() > 0
            ):
                cors.append(
                    np.corrcoef(
                        aa,
                        bb,
                    )[0, 1]
                )

        if cors:

            print(
                "  corr_j(|Sbar|, |Dln|): "
                f"median {np.median(cors):+.3f}"
            )

            print(
                "  -> strong anticorrelation means the two-factor "
                "attribution is intrinsically difficult to separate."
            )

        # ===================================================================
        # NORMALISED DENSITY PROFILES
        # ===================================================================

        print()
        print(
            "  normalised density profiles, median over rows "
            "(each row scaled to its own maximum):"
        )

        def prof(M):
            out = []

            for j in range(nN):

                vals = []

                for r in range(n_r):

                    denom = max(
                        np.abs(M[r]).max(),
                        NUM_FLOOR,
                    )

                    vals.append(
                        np.abs(M[r, j])
                        / denom
                    )

                out.append(
                    np.median(vals)
                )

            return out

        print(
            "     j        "
            + "".join(
                f"{j:>8d}"
                for j in range(nN)
            )
        )

        for label, M in (
            ("eps    ", eps),
            ("|Sbar| ", Sbar),
            ("|Dln|  ", Dln),
        ):

            print(
                f"     {label}  "
                + "".join(
                    f"{v:8.3f}"
                    for v in prof(M)
                )
            )

        # ===================================================================
        # PREREGISTERED VERDICT
        # ===================================================================

        print()

        f_sb = (
            a_sb / n_r
        )

        f_dl = (
            a_dl / n_r
        )

        if (
            f_sb > 0.5
            and f_dl > 0.5
        ):

            print(
                "  VERDICT: INCONCLUSIVE."
            )

            print(
                "  Both factors track the eps ridge in a majority "
                "of rows; this attribution test cannot separate them."
            )

            print(
                "  Describe the ridge as arising from their combined "
                "finite-step product, attributing it to neither alone."
            )

            print(
                f"  (row-wise overlap {both}/{n_r}; descriptive only)"
            )

        elif (
            f_sb >= ACCEPT_FRAC
            and f_dl < 0.5
        ):

            print(
                "  VERDICT: SENSITIVITY-DOMINATED RIDGE. "
                "H ACCEPTED."
            )

            print(
                "  Across density, the plateau-error maximum follows "
                "the exact mean reservoir sensitivity more consistently "
                "than the magnitude of the ground-state displacement."
            )

            print(
                "  Since Sbar is the interval average of f3-f4, "
                "the ridge is attributed to passage through the "
                "mixed-supply region of the post-step atomic response."
            )

            print(
                "  The local atomic model therefore generates this "
                "ridge without requiring spatial divertor structure."
            )

            print(
                "  This does not imply that spatial structure is "
                "irrelevant to an experimental line-integrated measurement."
            )

        elif (
            f_dl >= ACCEPT_FRAC
            and f_sb < 0.5
        ):

            print(
                "  VERDICT: DISPLACEMENT-DOMINATED RIDGE. "
                "H REJECTED."
            )

            print(
                "  Across density, the plateau-error maximum follows "
                "the magnitude of the ground-state displacement more "
                "consistently than the exact mean sensitivity."
            )

            print(
                "  The Chapter-5 mechanism should therefore be described "
                "primarily in terms of how far the ground reservoir moves."
            )

        else:

            print(
                "  VERDICT: NEITHER CRITERION MET."
            )

            print(
                "  H is neither established nor cleanly refuted; "
                "report the two exact factors separately."
            )

        # ===================================================================
        # LOCAL DETAIL AROUND THE OBSERVED RIDGE
        # ===================================================================

        print()

        print(
            f"  detail over {NEIGHBOURHOOD} "
            f"(display only; test used all {nN} columns)"
        )

        print(
            f"    {'Te':>7s} "
            + "".join(
                f"{'j=%d' % j:>9s}"
                for j in NEIGHBOURHOOD
            )
            + f"  {'argmax eps':>10s}"
            + f" {'|Sbar|':>7s}"
            + f" {'|Dln|':>6s}"
        )

        stride = max(
            1,
            n_r // 8,
        )

        for r in range(
            0,
            n_r,
            stride,
        ):

            cells = "".join(
                f"{100 * eps[r, j]:8.2f}%"
                for j in NEIGHBOURHOOD
            )

            print(
                f"    {Te[rows[r]]:7.3f} "
                f"{cells}  "
                f"{j_eps[r]:10d} "
                f"{j_sb[r]:7d} "
                f"{j_dl[r]:6d}"
            )

        # Store cross-step argmax information.
        across[k] = {
            i: (
                int(j_eps[r]),
                int(j_sb[r]),
                int(j_dl[r]),
            )
            for r, i in enumerate(rows)
        }

        # ------------------------------------------------------------------
        # Show whether j=3 actually lies near its sensitivity peak
        # ------------------------------------------------------------------

        sample_rows = list(
            range(n_r)
        )[::max(1, n_r // 5)]

        print()
        print(
            "  ridge column j=3: "
            "where does the frozen reservoir lie "
            "relative to its sensitivity peak?"
        )

        print(
            f"    {'Te':>7s}"
            f" {'b1(u_old)':>11s}"
            f" {'b1_peak':>11s}"
            f" {'ratio':>8s}"
            f" {'|Sbar|':>8s}"
            f" {'|Dln|':>8s}"
            f" {'eps':>8s}"
        )

        for r in sample_rows:

            print(
                f"    {Te[rows[r]]:7.3f}"
                f" {b1_u[r, 3]:11.3e}"
                f" {b1_pk[r, 3]:11.3e}"
                f" {b1_u[r, 3] / b1_pk[r, 3]:8.3f}"
                f" {abs(Sbar[r, 3]):8.4f}"
                f" {abs(Dln[r, 3]):8.4f}"
                f" {100 * eps[r, 3]:7.3f}%"
            )

    # =======================================================================
    # CROSS-STEP COMPARISON
    # =======================================================================

    print()
    print("=" * 78)
    print("CROSS-STEP COMPARISON")
    print("=" * 78)

    n_each = {
        k: sum(
            1
            for i in range(nT - k)
            if TE_LO <= Te[i] <= TE_HI
        )
        for k in STEPS
    }

    print(
        f"  STEP={STEPS[0]} reaches {n_each[STEPS[0]]} rows; "
        f"STEP={STEPS[1]} reaches {n_each[STEPS[1]]} rows."
    )

    print(
        f"  Direct comparison uses the "
        f"{len(common_rows)} starting rows accessible to both."
    )

    k1, k4 = STEPS

    same_eps = sum(
        across[k1][i][0]
        == across[k4][i][0]
        for i in common_rows
    )

    same_sb = sum(
        across[k1][i][1]
        == across[k4][i][1]
        for i in common_rows
    )

    same_dl = sum(
        across[k1][i][2]
        == across[k4][i][2]
        for i in common_rows
    )

    same_sensitivity_agreement = sum(
        (
            across[k1][i][1]
            == across[k1][i][0]
        )
        ==
        (
            across[k4][i][1]
            == across[k4][i][0]
        )
        for i in common_rows
    )

    denom = max(
        len(common_rows),
        1,
    )

    print(
        "  eps ridge at same density column       : "
        f"{same_eps}/{len(common_rows)} "
        f"({100 * same_eps / denom:.0f}%)"
    )

    print(
        "  |Sbar| maximum at same density column  : "
        f"{same_sb}/{len(common_rows)} "
        f"({100 * same_sb / denom:.0f}%)"
    )

    print(
        "  |Dln| maximum at same density column   : "
        f"{same_dl}/{len(common_rows)} "
        f"({100 * same_dl / denom:.0f}%)"
    )

    print(
        "  sensitivity/eps argmax agreement status: "
        f"{same_sensitivity_agreement}/{len(common_rows)} "
        f"({100 * same_sensitivity_agreement / denom:.0f}%)"
    )

    print()
    print(
        "  READING:"
    )

    print(
        "  A ridge that moves with step amplitude is not thereby an artifact."
    )

    print(
        "  The step changes both the traversed reservoir interval and the "
        "post-step operator S+(u), so movement can be genuine finite-amplitude "
        "physics."
    )

    print(
        "  The relevant questions are therefore whether the eps ridge moves, "
        "whether the |Sbar| and |Dln| maxima move with it, and whether the "
        "factor attribution survives the change in perturbation amplitude."
    )

    # =======================================================================
    # End
    # =======================================================================

    print()
    print("=" * 78)
    print("  Report only. Nothing was written.")
    print("=" * 78)


if __name__ == "__main__":
    main()