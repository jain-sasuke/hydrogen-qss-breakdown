# Code Reading Guide

A reading order for understanding this repository from scratch.

**The one idea to hold onto:** the entire repo exists to build one matrix and
then ask one question about it.

```
    dn/dt = L(Te, ne) · n  +  S(Te, ne, n_ion)
            ^^^^^^^^^^^^      ^^^^^^^^^^^^^^^
            43×43 matrix      recombination source
```

Everything upstream of `L` is *making its numbers*. Everything downstream is
*analysing its eigenvalues*. If you always know which side of `L` a script is
on, you cannot get lost.

---

## Stage 0 — Orientation (30 min, no code)

Read in this order:

| # | File | Why |
|---|---|---|
| 0.1 | `CLAUDE.md` | The rules. Especially "report, do not repair" and the known-issues table. |
| 0.2 | `outputs/master_plan_v2.md` | What the thesis is arguing and what is still open. |
| 0.3 | `outputs/thesis_grade_results_backlog.md` | The result register. Which numbers are 💡 vs ✅. |
| 0.4 | `run_pipeline.sh` | The 5-stage skeleton. Skim only — it is just a runner. |

You now know the vocabulary: `Te`, `ne`, `L`, `τ_QSS`, `τ_relax`, `M`, `ε`.

---

## Stage 1 — The spine (read these three, in this order)

This is the 20% that explains 80%. Do not read anything else until these
three make sense.

### 1.1 `src/validation/cr_context.py` (262 lines)

**Start here.** It is short, has no physics, and teaches you the *shape* of
everything: 50 Te points, 8 ne points, 43 states, and how state index → (n, ℓ)
works.

Run it and look at the output:
```sh
/opt/anaconda3/envs/cr/bin/python -c "
import sys; sys.path.insert(0,'src/validation')
from cr_context import CRContext; print(CRContext.load().describe())"
```

Key things to extract:
- `ctx.n_values` — principal quantum number of each of the 43 states
- states 0–35 are ℓ-resolved (n=1–8); states 36–42 are **ℓ-bundled** (n=9–15)
- `n=3 → [3,4,5]` (Hα upper), `n=4 → [6,7,8,9]` (Hβ upper)

> ⚠️ `n_values` is `float64`, not int. Bites people writing `==` comparisons.

### 1.2 `src/rates/assemble_cr_matrix.py` → the `build_L()` function (359 lines)

**This is the hub of the whole repo.** Read `PATHS` (line ~85) then `build_L()`.

`build_L` adds five physical terms into one 43×43 array:

| term | physics | sign |
|---|---|---|
| 1 | electron-impact excitation / de-excitation | off-diagonal gain, diagonal loss |
| 2 | spontaneous emission (Einstein A) | downward only |
| 3 | electron-impact ionisation | pure loss (leaves the manifold) |
| 4 | recombination | enters as `S`, **not** in `L` |
| 5 | proton-impact ℓ-mixing (PSM20) | within-shell, n=2–8 only |

Pair with: **`outputs/derivation_01_rate_matrix.md`**.

Two structural facts that drive every later result — make sure you see *why*
in the code:
- `colsum(L) = −K_ion·ne`. Ionisation is the only true loss. `L` is **not**
  conservative, because the ion is outside the 43 states.
- Recombination is a **fixed reservoir** in `S`, proportional to `n_ion`. So
  `eig(L)` does not depend on `n_ion` at all. (This is exactly why the "is
  τ_QSS an open-boundary artifact?" question is a real one.)

### 1.3 `src/rates/solve_cr.py` (650 lines)

How `L` gets used. Only two modes matter:

```python
n_ss = -inv(L) @ S            # steady state, algebraic
solve_ivp(..., method='Radau') # time-dependent, stiff (ratio up to 1e10)
```

Pair with: **`outputs/derivation_03_qss_steady_state.md`**.

**Checkpoint.** You should now be able to answer: *where do the 43 states come
from, what are the five terms in L, and why is Radau used instead of RK45?*
If not, re-read 1.2 before going on.

---

## Stage 2 — Where L's numbers come from (upstream)

Now walk backwards from `PATHS` in `assemble_cr_matrix.py`. Every entry there
is produced by exactly one script. Read them in data-flow order:

| # | Script | Produces | Physics |
|---|---|---|---|
| 2.1 | `src/parsers/parse_ccc.py` | `ccc_crosssections.h5` | Raw CCC cross-sections σ(E) from Bray's database. Read the FILE CONVENTION docstring — the `2P.1S` naming and the `:` = n=10 trick. |
| 2.2 | `src/rates/compute_K_CCC.py` | `K_CCC_exc_table.npy` | σ(E) → K(Te) by Maxwellian averaging, then **detailed balance** for the reverse rate. |
| 2.3 | `src/rates/radiative_rates.py` | `A_resolved.npy` | Einstein A coefficients, Hoang-Binh (1993). |
| 2.4 | `src/rates/ionization_rates.py` | `K_ion_final.npy` | Ionisation — the term that makes `L` non-conservative. |
| 2.5 | `src/rates/recombination_rates.py` | `alpha_RR_*.npy` | Radiative + three-body recombination → the `S` vector. |
| 2.6 | `src/rates/compute_lmix.py` | `K_lmix.npy` | Proton-impact ℓ-mixing, PSM20 Debye formula. |
| 2.7 | `src/rates/assemble_K_exc.py` | `K_exc_full.npy` | Merges CCC (accurate, n≤8) with V&S (~20%, n=11–15) into one table. |

Pair 2.2 with **`outputs/derivation_02_detailed_balance.md`**, and 2.6 with
**`outputs/derivation_04b_ell_mixing.md`** — that one documents the `F(U_m)`
bug that was found and fixed, which is the model for what good work looks like
here.

Two things to notice as you read, because they matter later:

- **`compute_lmix.py` is NOT in `run_pipeline.sh`.** It produces `K_lmix.npy`,
  which `assemble_cr_matrix.py` *requires*. So `run_pipeline.sh` cannot
  actually rebuild the repo from scratch. Know this before you trust it.
- `K_lmix` is non-zero only for n=2–8. There is **no ℓ-mixing rate anywhere in
  the pipeline for the bundled n=9–15 block** — statistical ℓ-equilibrium there
  is *assumed*, not computed.

---

## Stage 3 — The thesis question (downstream)

Now you can read the part the thesis is actually about.

### 3.1 `src/validation/qss_analysis.py` (473 lines)

Read these three functions in order:

1. `compute_timescale_map()` — eigenvalues of `L` at all 400 grid points.
   `τ_QSS = 1/|λ₀|` (slowest), `τ_relax = 1/|λ₁|` (next), `M = τ_QSS/τ_relax`.
2. `epsilon_after_step()` — the error metric. Steps Te by +0.6 eV, integrates,
   compares the actual excited/ground ratio to the QSS ratio.
3. the `__main__` block — how the breakdown map is assembled.

Pair with **`outputs/derivation_04_two_timescales.md`** and
**`outputs/derivation_05_qss_breakdown.md`**.

> ⚠️ **Two live issues in this file**, both documented and neither to be
> "fixed" casually:
> - `ε` is a `max` over all 42 excited states, normalised by the ground state.
>   At the cold corner that max is attained at `n15` — the top *bundled*
>   state — which makes `eps_res ≈ 0.99` nearly automatic. See
>   `outputs/derivation_07_transient_diagnostic_error.md`.
> - The `eps_bar` decay law assumes `ε(t) = eps_res·exp(−t/τ_QSS)` using the
>   *pre-step* operator's eigenvalue. Direct integration disagrees.

### 3.2 `src/validation/verify_timescales.py` (473 lines)

The independent re-derivation of 3.1. Reading these two side by side is the
single most instructive thing in the repo: same physics, two implementations,
and the places they disagree are where the real results live.

Pair with **`outputs/derivation_04c_non_normality.md`** — `L` is strongly
non-normal (numerical abscissa `+1.3e11` vs spectral abscissa `−4.4e4`), so
"eigenvalues tell you the dynamics" is an assumption that has to be *checked*,
not assumed.

---

## Stage 4 — The verification layer

These exist to attack Stage 3's numbers. Read after you believe Stage 3.

| Script | Asks |
|---|---|
| `verify_ramp_vs_step.py` | Does the instantaneous-step idealisation apply to a real ELM ramp? (Test 3 is the key one.) |
| `verify_boundary_descent.py` | Is the Griem boundary where it should be? |
| `verify_bundling_psm20.py` | Is the n=9–15 bundling valid? **Never been run** — `outputs/bundling/` does not exist. |
| `validate_gates.py` | The pass/fail gate battery (A–E). |
| `anderson_benchmark_qc.py` | Benchmark against published Anderson results. |

---

## Stage 5 — Analysis and figures (read last)

`src/analysis/physics_tests.py` → `test_scaling.py` → `unified_scaling.py` →
`plot_results.py`. These consume everything above and produce the thesis
figures. Safe to skim; nothing conceptually new.

---

## Appendix — What to ignore, and why

`src/rates/` contains ~25 scripts, most of which are **not** part of the
pipeline. Per `CLAUDE.md`, this is the environment that bred the −46% Hα
artifact. Do not read these while learning, and confirm which is current
before ever running one:

- `Balmer_transient_ratio.py` **and** `Balmer_transient_ratio_v3_before_paper_label_patch.py` **and** `backup_scripts/Balmer_transient_ratio_old_*.py` — three near-copies
- `mori_zwanzig.py`, `mori_zwanzig_weekc.py`, `mori_zwanzing_weekb.py` (note the typo) — three near-copies
- `S_criterion.py`, `S_criterion_fixed.py`, `S_criterion_3P.py` — three near-copies
- `check.py`, `check_mz.py`, `qsscheck.py`, `diagnostic.py` — 8–16 line scratch files

---

## Suggested schedule

| Session | Content | Outcome |
|---|---|---|
| 1 | Stage 0 + 1.1 | You know the grid, the 43 states, the index map |
| 2 | 1.2 (`build_L`) + derivation_01 | You can name the 5 terms and explain `colsum(L)` |
| 3 | 1.3 + derivation_03 | You can compute a steady state yourself |
| 4 | Stage 2 (2.1–2.4) | You can trace one rate from raw σ(E) to its slot in `L` |
| 5 | Stage 2 (2.5–2.7) | You understand the CCC/V&S merge and the ℓ-mixing gap |
| 6 | 3.1 + derivation_04, 05 | You can explain `M`, `τ_QSS`, `ε` |
| 7 | 3.2 + derivation_04c | You know why non-normality threatens the modal reading |
| 8 | Stage 4 | You can attack the results |

**The exercise that proves you understand it:** pick one number — say
`τ_QSS = 22.73 µs` at the ITER reference point — and trace it all the way back
to a raw cross-section file in `data/raw/ccc/`, naming every script it passed
through. That is the standard `CLAUDE.md` asks for on every number.
