# Change Report

**10 September 2026.** Every modification to a pre-existing file this session,
built from `git diff fa18595 HEAD` rather than from memory, plus the physics
breaks and hardcoded values found but deliberately not repaired.

---

## 1. No source code was modified. No data was touched.

`git diff --name-status fa18595 HEAD -- src/` returns **no `M` entries**. Every
script under `src/` is an addition. Two new files were written:
`src/validation/verify_lyman_trapping.py` and
`src/analysis/make_ch5_figures.py`. Nothing existing was edited, and no
tolerance, filter, threshold or gate was changed anywhere.

`data/` is untouched. `L_grid.npy` still carries its 21 Jul 20:44 mtime and
hashes to `2d92b58e1693107d…`, the canonical value.

## 2. Pre-existing files that were modified — the complete list

| file | change | why |
|---|---|---|
| `.gitignore` | +8 lines | Ignore the 20 third-party skills installed into `.claude/skills/`; three negations keep the project's own skills tracked |
| `outputs/thesis_grade_results_backlog.md` | +132 lines, append only | The verifier appended its run outcome at ▶️ Run. `CLAUDE.md`'s reporting section requires this |
| `validation/plateau_gridmap/plateau_gridmap.csv` | 1 line | Generation timestamp only. **Already dirty before the session began**, present in the opening `git status` |
| `validation/plateau_gridmap/plateau_gridmap.txt` | 1 line | Same |
| `outputs/thesis_ready.md` | register corrections | Seven documented corrections, each recorded in place as `An earlier version read…` rather than silently overwritten |
| `thesis_tex/chapter3.tex` | 43 insertions, 13 deletions | Three surgical corrections: the detachment claim, the unbacked ADAS promise, the unscoped 0.06% figure |
| `thesis_tex/thesis_main.tex` | wiring | `\input` for chapters 1, 2, 6; `graphicspath` to reach `../figures/`; bibliography enabled against `references.bib`; `csquotes` added for `\enquote` |

## 3. Environment changes outside the repository

`siunitx` and `csquotes` installed from CTAN into `~/Library/texmf`. This needs
no admin rights and does not modify the system TeX tree at
`/usr/local/texlive/2026basic`. **`sudo tlmgr install siunitx` is no longer
needed.**

---

## 4. Physics that breaks, found and reported not repaired

### 4.1 An error I propagated: σ₀ high by √2

`findings_10` ADDENDUM A quotes σ₀ = 7.74×10⁻¹⁴ cm² for Lyman-α at 1 eV, and
`chapter6.tex:143` repeats it. **Both are wrong by exactly √2.** The Doppler
width was built from √(kT/m) instead of √(2kT/m). Settled by direct
computation:

| source | σ₀ (cm², 1 eV) |
|---|---|
| `verify_lyman_trapping.py` run log | 5.4705×10⁻¹⁴ |
| `escape_factor.lyman_alpha_sigma0`, independent implementation | 5.4737×10⁻¹⁴ |
| direct evaluation with √(2kT/m) | 5.4779×10⁻¹⁴ |
| **ADDENDUM A and `chapter6.tex:143`** | **7.7469×10⁻¹⁴** |

7.7469/5.4779 = 1.4142.

**The script results are unaffected.** `verify_lyman_trapping.py` derives σ₀
from the repo's own A-values and cross-checks against the `escape_factor`
module to 0.059%; it used the correct value throughout. The trapping census
(45/448 above 2 eV, invariant to 0.5% across a twentyfold slab range) stands.
What is wrong is the σ₀ and τ figures quoted in prose.

Consequences: τ per cm at [0,4] is **80.4 cm⁻¹, not 114**. Over 5 cm the depth
is 402 full-path or 201 half-slab, and **7900 does not follow from either**.
`chapter6.tex:150-166` states 114 and 7900 in the same passage, inconsistent
with each other by a factor 13.9, and neither appears in the artifact it cites.

### 4.2 The conservation gate is tautological

`chapter3.tex:222-225` claims that "an error in a single off-diagonal element,
a transposed index, or a missing back-reaction all break this immediately."
Three faults were injected and the gate caught **none** of them:

| injected fault | max ΔL (s⁻¹) | conservation residual |
|---|---|---|
| baseline | — | 2.238×10⁻¹² |
| 10× error in C(1s→2p) | 6.9×10⁵ | 2.238×10⁻¹² unchanged |
| all de-excitation deleted | 1.9×10¹¹ | 2.171×10⁻¹² unchanged |
| K_exc input array transposed | 2.7×10¹¹ | 1.837×10⁻¹² unchanged |
| A halved but γ not | 3.1×10⁸ | 3.63×10¹ **caught** |

The diagonal is constructed as minus the column sum of the same off-diagonal
array, so the check is exact by construction. It detects one defect only: an
A/γ inconsistency. `chapter3.tex:286` states "A test that a wrong result would
also have passed is not a test", which condemns its own claim three paragraphs
above.

### 4.3 Chapters 3 and 6 contradict each other about the same operator

`chapter3.tex:1024-1030` asserts the two-channel split and the tanh bound "hold
for a trapped-line rate matrix exactly as they do for an optically thin one".
That is false. Trapping makes A depend on n(1s), so L_FF and L_Fg acquire an
n(1s) dependence, the split stops being affine, the logistic loses unit width,
and the bound degrades to tanh(mΔ/4) with m unknown. `chapter6.tex:202-203`
says the opposite and is right. The same defect appears at
`chapter6.tex:570-575` for molecular states, which would be a **third** supply
channel and change the functional form rather than perturb it.

### 4.4 Three-body recombination at n_e = 10¹²

`chapter3.tex:190-193` says three-body supplies "under 10% of the feed into any
level" at 10¹². Measured: **30 of 36 levels exceed 10%** at 1.00 and 2.95 eV,
26 of 36 at 10 eV. It is essentially the whole feed into the bundled shells.
Flagged independently by the Chapter 2 agent and by Gate 4.

### 4.5 Other wrong statements in written LaTeX

M_eff floor 77.6 should be **77.3** (it multiplied two extrema from different
grid points, the same defect already caught once). "Nine orders of magnitude"
should be 4.8. "Eight orders spanned by the entries" should be about twelve
decades. α_RR does not peak near Te ~ I_n; it decreases monotonically across
the whole grid. The brightness equation is off by 10⁴ in its stated units. R is
defined twice, incompatibly, in Chapters 1 and 3.

---

## 5. Hardcoded values, by class

Full sweep of all 78 `.py` files under `src/`. None was repaired.

**Grid or state hardcoding.** `TE_GRID = np.logspace(...)` is written
independently in **nine** producer files that build L. Textually identical
today; only `pre_assembly_check.py` guards against divergence, it is not
invoked by `assemble_cr_matrix.py`, and it has no entry for the ℓ-mixing grid.
`make_ch3_figures.py:160` and `verify_plateau_bridge.py:519` protect the
benchmark index with `assert`, **which `python -O` deletes**.
`verifych3_gb.py:86` hardcodes `g = 0` instead of `ctx.ground_index`.
`plot_results.py:338` and `physics_tests.py:306` hardcode the entire state→n
map. `check.py:7-8` re-derives the grid formula and prints the retracted
"Thesis M = 611".

**Untraceable thresholds.** `TOL_PLATEAU = 2.0e-3` with the plateau duration
linear in it and no scan. `BREAKDOWN_THRESHOLD = 0.10` declared in three
separate files. `win_lo = win_hi = 30`, and a fourth incompatible window
convention in `verify_plateau_bridge.py`. The isolation gate at
`make_ch3_figures.py:155` is set to 10.0 against a measured minimum of 24.33.

**Hardcoded fits.** `verify_partition.py:134` and `plot_results.py:459` carry
two mutually inconsistent exponential fits both named `eps_step`, 9.6% apart at
Te = 3 eV, neither traceable to a run. `verify_partition.py:156` evaluates its
fit at the wrong index, so a whole table column is constant at 52.4%.

**Duplicated constants.** The Rydberg energy appears with **eight distinct
values across 17 sites**. Bounded impact below 0.011% everywhere it is live,
and exactly zero inside `corrcoef`. A traceability failure, not a numerical
one. `CHI_H` names R∞hc but reads as the ionisation potential, and it is used
in two live Saha-Boltzmann expressions.

**Silent fallback.** `verify_bundling_psm20.py:160-165` synthesises grids when
files are missing, printing a warning and continuing. This violates
`CLAUDE.md` rule 2 directly. It never fires today because the files exist.

**Retracted numbers still in code.** `eigs[eigs < -1.0]` survives at
`solve_cr.py:269` and `check_mz.py:10`. At the cold corner it turns M = 1.729×10⁹
into **M = 1.467**, a wrong answer that reads as a physics finding. Confirmed
imported by nothing. That is luck, not design.

**Clean, verified rather than assumed:** `cr_context.py`, `preflight.py` (its
pinned SHAs match byte-for-byte), `verify_eps_gridmap.py`,
`verify_ridge_mechanism.py`, `verify_lyman_trapping.py`,
`verify_plateau_gridmap.py`, `verify_timescales.py`,
`verify_boundary_descent.py`, and `make_ch5_figures.py`, which reads the step
size from the CSV header rather than re-declaring it and raises rather than
asserts.
