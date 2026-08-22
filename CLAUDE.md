# CLAUDE.md — Project context for Claude Code

**Project:** Time-dependent collisional-radiative modelling of hydrogen plasma;
quantifying quasi-steady-state validity in ITER divertor conditions.
M.Tech thesis, Chemical Engineering, IIT Kanpur. **Defense 1 September 2026.**

---

## Read these first

```
outputs/master_plan.md                      roadmap, standing decisions, schedule
outputs/thesis_grade_results_backlog.md     result register + open ideas
outputs/derivation_*.md                     the verified dossier
```

These files are the project's memory and **take precedence over any chat
history**. If something you find contradicts them, they win — several earlier
numbers in this project turned out to be stale or artifacts, and re-importing
them has caused real problems.

---

## Your role here: REPORT, DO NOT REPAIR

This repository is under **verification**, not development. The default is to
investigate and report, never to fix.

**Do not, unless explicitly asked:**
- adjust a tolerance so a check passes
- substitute a default, stand-in, or synthetic value for missing data
- "helpfully" correct a script whose output looks wrong
- regenerate a data file to make something run
- silently widen a filter, clip a range, or drop outliers

**A failing check is information.** Report it, state what it means, and stop.
A check that was made to pass is a check that no longer verifies anything.

**Missing data:** stop and say so. Never proceed with a placeholder. The whole
point of these scripts is that the numbers trace to real atomic data; a
stand-in produces plausible-looking output that means nothing.

---

## Non-negotiable rules

1. **No hardcoding.** Grids, state ordering, and matrices load from the
   pipeline's own files via `src/validation/cr_context.py`. If a script needs a
   grid, it loads `Te_grid_L.npy` / `ne_grid_L.npy` — never redefines them. A
   script that hardcodes a grid will silently mislabel every result when the
   pipeline changes.

2. **Fail loudly.** Missing file, mismatched shape, stale grid → raise with a
   clear message naming which file is wrong. Never fall back to a guess.

3. **No black boxes.** Any script whose output enters the thesis must be
   readable line by line. When asked to audit, explain what each block computes,
   what physics it implements, and what would break if written differently.
   Trace every magic number to a source.

4. **Provenance on every number.** State where it came from: which script, which
   file, which grid point, which run. "τ_relax = 2.28 ns" is not a result;
   "τ_relax = 2.277 ns from eig(L_grid[23,5]), L_grid regenerated 14 Jul after
   the F(U_m) fix" is.

5. **Ground-truth hierarchy: physics → math → code → documents.** Documents are
   consistency checks, never authorities.

---

## Known issues — do not "fix" without asking

| Location | Issue |
|---|---|
| `src/validation/qss_analysis.py` | Contains `eigs = eigs[eigs < -1.0]`, which silently truncates τ_QSS at ~1 s. Real values reach 67 s in the cold corner (Te=1 eV, ne=1e12). **Report the effect; do not change the filter without discussion** — it may be load-bearing elsewhere. |
| `src/rates/` | Multiple near-duplicate scripts (`Balmer_transient_ratio*.py`, `mori_zwanzig*.py`, `backup_scripts/`). Confirm which is current before running. This is the environment that bred the −46% Hα artifact. |
| Chapters 3–5 | Contain stale numbers (25 ns, M=611, −46% Hα, ne^-1.00 scaling). See backlog §B. Do not propagate. |

---

## Verified reference values

ITER reference point: **Te = 2.947 eV** (index 23), **ne = 1.389×10¹⁴ cm⁻³**
(index 5), from `L_grid.npy` regenerated 14 Jul 2026 after the ℓ-mixing
F(U_m) correction:

| Quantity | Value |
|---|---|
| τ_QSS = 1/\|λ₀\| | 22.73 µs |
| τ_relax = 1/\|λ₁\| | 2.277 ns |
| M = τ_QSS/τ_relax | 9982 |
| μ(L) numerical abscissa | +1.28×10¹¹ s⁻¹ |
| spectral abscissa | −4.40×10⁴ s⁻¹ |

Convention: **M = τ_QSS/τ_relax**, large = timescale separation holds.

If a run disagrees with these, **report the disagreement** — do not assume the
stored value is right, and do not assume the new one is either.

---

## Verification scripts

```
src/validation/cr_context.py               loads grids + state ordering (import this)
src/validation/verify_timescales.py        spectrum, framings, eigenvectors, ODE, grid scan
src/validation/verify_boundary_descent.py  Griem boundary descent + sensitivity
src/validation/verify_ramp_vs_step.py      does the step picture apply? (Test 3 is the key one)
```

Outputs go to `validation/` and `figures/`. All auto-discover the repo root.

---

## When reporting results

State, every time:
- which script, which arguments, which files it loaded
- whether it reproduced a previously recorded value, and to how many digits
- what **would have** refuted the claim, and whether that appeared
- which numbers are robust and which depend on a definition, norm, or weighting

Then append the outcome to the relevant backlog entry with its graduation state
(💡 → 📐 → 💻 → ▶️ → ✅). **Nothing reaches ✅ without a sensitivity check and
written caveats.**

---

## What good work looks like here

Finding a bug, tracing it to the published source equation, quantifying its
size, and proving the headline result is insensitive to it — that is a *result*,
not a setback. It is what happened with the ℓ-mixing F(U_m) error, and it is the
standard.

Making a check pass is not.
