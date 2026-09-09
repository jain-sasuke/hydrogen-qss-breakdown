# RESUME HERE

**Written 10 September 2026**, when a session limit killed five agents
mid-flight. This file exists so the next session starts in ten minutes rather
than two hours. **Read this first, then `plan_to_submission.md`.**

Everything below is committed. `git log --oneline -8` shows the session's work.

---

## 1. How to restart

```bash
cd /Users/phi/Desktop/non_markovian_cr
git log --oneline -8              # this session's work
git branch --show-current         # backup/verification-session-2026-09-10
cat outputs/RESUME_HERE.md        # this file
```

Then open, in order: `findings_10_four_agent_review.md` (incl. **ADDENDUM A**
Lyman trapping and **ADDENDUM B** first-principles re-derivation),
`plan_to_submission.md`, `writing_specification.md`.

**Prompt to paste into the new session:**

> M.Tech thesis, IIT Kanpur, time-dependent collisional-radiative modelling of
> hydrogen for divertor Balmer-ratio diagnostics. Defence ~21 Sep 2026.
> Read `outputs/RESUME_HERE.md` first, then `outputs/plan_to_submission.md`
> and `outputs/writing_specification.md`. Standing method: check every number,
> no hardcoded numbers, verify against literature and books, derive from first
> principles, physics and math outrank code and documents. Report, do not
> repair.

---

## 2. Two blockers that are not mine to clear

1. **`sudo tlmgr install siunitx`** — the only missing LaTeX package of the
   sixteen the preamble needs. **Nothing can be compiled until this runs.**
   Verified by probe: `pdflatex` stops with `File 'siunitx.sty' not found`.
   The Chapter 6 agent reported a clean 19-page compile; that does not
   reproduce and should be treated as unverified.
2. **The submission deadline is still unconfirmed** — open since 23 August
   (`master_plan_v2.md:398`). If examiners need the document a week before a
   21 Sep defence, the real deadline is **14 September** and Tier 2/3 of the
   plan must be cut, not scheduled.

---

## 3. State of the thesis

| Chapter | File | Lines | State |
|---|---|---|---|
| 1 | `chapter1.tex` | 747 | **PARTIAL — agent killed mid-draft.** Do not treat as complete |
| 2 | `chapter2.tex` | 1585 | **PARTIAL — agent killed mid-draft.** Do not treat as complete |
| 3 | `chapter3.tex` | 1640 | **complete and corrected** (3 surgical edits, commit `ec3d890`) |
| 4 | — | — | not started |
| 5 | `chapter5.tex` + `chapter5_C5D.tex` | 59 + 327 | **not started.** Register now fixed, so it is unblocked |
| 6 | `chapter6.tex` | 1077 | **drafted**, 10 sections, 8 `\todo`s. LaTeX unvalidated |
| 7 | — | — | not started |

Figures: all four Chapter 5 figures built and committed
(`fig5_1_eps_map`, `fig5_2_eps_vs_ne`, `fig5_3_M_vs_eps`, `fig5_4_trapping`)
plus `figures/fig5_captions.tex`. `references.bib` has 6 verified entries.

`thesis_main.tex` still has **no `\include` lines**. Nothing is wired together.

---

## 4. What was verified this session, and what it changed

### 4.1 The mathematics is correct (ADDENDUM B)

All six Chapter 3 core results re-derived independently, code unopened,
predictions written first. Every number in `chapter3.tex`
§sec:ground_fed_fraction reproduces. **This is the single most important
result of the session** — the foundation is sound.

### 4.2 The register was falsified and is now fixed (commit `a4ae962`)

A8–A11 demoted ✅→▶️; τ_QSS floor 1.18 µs → 75.4 ns; A10 column j=4 restored;
detachment retracted; A11 restated inside the Te ≥ 2 eV scope; W2's
M/ε colocation claim corrected. **Chapter 5 may now be written from it.**

### 4.3 Lyman trapping computed (ADDENDUM A, commit in `verify_lyman_trapping.py`)

Above 2 eV the ELM breakdown count does not move (45/448) and the worst case
moves 0.5% across a twentyfold slab range. Below 2 eV the 38.7% becomes
11.6–15.7%. §3.5.3's Te ≥ 2 eV cut is a **measured boundary**, not a hedge.

---

## 5. Findings from the three completed verification agents

**These are recorded here because the agents are gone. Act on them.**

### 5.1 The escape-factor geometry is mislabelled — affects ADDENDUM A

ADAS214 eq. 3.14.14, obtained and read directly from
`https://www.adas.ac.uk/man/chap2-14.pdf`, is the **isotropic / sphere-centre**
case (ADAS geometry **g1**), *not* a slab. `src/analysis/escape_factor.py`
documents it as "a homogeneous slab of half-thickness b". A true slab (g2)
requires `Θ_slab(τ) = ∫₀¹ Θ_iso(τ/μ) dμ`, which is **≈2.1× smaller** at large
τ_c (ratio → 1/2 asymptotically):

| τ_c | Θ_P (3.14.14, sphere) | Θ_P slab (g2) | ratio |
|---|---|---|---|
| 1 | 0.513929 | 0.271806 | 0.529 |
| 100 | 2.5307e-3 | 1.2077e-3 | 0.477 |
| 1000 | 2.0830e-4 | 1.0086e-4 | 0.484 |

**Decision needed:** either relabel as isotropic/spherical, or — if the
divertor is genuinely a slab — eq. 3.14.14 is the wrong ADAS case and Θ_P is
too large by ~2× in the thick regime. **This propagates into
`verify_lyman_trapping.py` and therefore into ADDENDUM A's numbers.** The
Te ≥ 2 eV conclusion is safe either way (Θ_P ≈ 1 there); the cold-corner
numbers are not.

**Everything else in that module verified correct against the primary source:**
τ_c = κ₀·(half-thickness) is exactly ADAS's convention (ADAS states three times
that b is edge-to-centre); the Ladenburg constant πe²/m_ec = 2.65400885e-2
cm² Hz; φ(ν₀) = 1/(√π Δν_D) with FWHM = 2√(ln2)Δν_D (ADAS's own constants
0.60056 and 1.06446 are exactly this); v_th = √(2kT/m) reproduces ADAS's
quoted 7.7 pm (H) and 5.4 pm (D); the Holstein leading constant is **1**, and
a factor 1.6 is *not* missing (proved numerically — a 1.6 would show as a
ratio → 0.625; measured ratios converge to 1).

### 5.2 Atomic data — one real error, one 33% trap

**WRONG:** `e = 4.80326e-10` esu. CODATA 2022 gives **4.803204713e-10** — a
+0.00115% error that looks like a digit transposition. Propagates as e² into
every A-coefficient (+0.0023%). Numerically immaterial; still a wrong constant.

**WRONG IDENTITY:** `m_H = 1.67262e-24 g` is the **proton** mass. The hydrogen
**atom** is 1.6735328407e-24 g, 0.0545% heavier. Neutral atoms Doppler-broaden
with the atom mass, so the width is +0.027% too wide. Small, but the wrong
physical object.

**THE 33% TRAP — audit this.** A(2p→1s) = 6.2649e8 s⁻¹ is an **ℓ-resolved**
level rate. A(n=2→n=1) = 4.6986e8 = (6/8)×6.2649e8 is the **n-resolved**
shell rate; the 3/4 is the 2s fraction that cannot radiate to 1s by E1.
**Putting the 2p value on an n-resolved element overestimates Lyman-α by
33.3%.** Given this repo's ℓ-mixing history, check the radiative block's
ℓ-resolution against `cr_context.py`'s state ordering.

**The A-values are not the NIST set.** The four stored Lyman A's are the exact
**non-relativistic, infinite-nuclear-mass** hydrogenic values (reproduced to
1–4 parts in 10⁵), sitting +0.058% above the NIST/Wiese–Fuhr recommended
values. Legitimate and self-consistent — **but do not cite NIST as their
source.** Same for the f-values (0.4162 = 2¹³/3⁹ exactly).

**Both hydrogen energies verified correct for what they are:**
`13.605693122994` is CODATA-2018 R_∞hc; `13.598434599702` is exactly the NIST
ASD H I ionization energy. **But `CHI_H` is a dangerous name for the former** —
χ_H universally means the ionization potential, which is the latter. If any
Saha or Boltzmann expression reads `CHI_H`, it is 0.0534% too large.
Note also: Y is *not* X times the reduced-mass factor (that gets 5 digits then
stops); the residual +1.4733e-4 eV is Dirac + Lamb shift, agreeing to 5 sf.

**Minor:** Ly-δ `f = 0.0139` is −0.32% low (3-sf rounding of 0.0139383) and
inconsistent with the stored Ly-δ A by 0.28%. A round-trip test (recompute A
from stored f, compare to stored A) flags n=5 at 0.28%, n=4 at 0.03%.

### 5.3 The hardcoded sweep — the headline finding

**ε_plateau is LINEAR in the step size, and the step size is a grid artifact.**

| step | max ε_plateau (all) | Te≥2 eV | Te≥2 & ne≥1e14 |
|---|---|---|---|
| k=1 (+4.81%) | **38.690%** | 18.073% | 10.355% |
| k=2 (+9.85%) | 87.410% | 38.203% | 23.380% |
| k=4 (+20.68%) | 190.376% | 81.133% | 58.531% |

"+4.81%" exists only because someone chose 50 log-spaced points between 1 and
10 eV. **"38.7%" means "38.7% per 4.8% temperature step" and is meaningless
quoted bare.** An examiner asking "what if the ELM is 10%?" gets 87%, and
there is no principled answer for why 4.81% is right.

**The invariant quantity is dε/d ln Te, and no script headlines it.** Report
that. Inside the defensible box the value is **10.4%**, a factor 3.7 lower.

Other findings worth acting on:

- **`assert` is not a guard.** `make_ch3_figures.py:160` and
  `verify_plateau_bridge.py:519` protect the benchmark index with `assert`,
  which `python -O` deletes. `make_ch5_figures.py` and `verify_ch3_claims.py`
  already use `raise`. Adopt `raise` everywhere.
- **`verify_partition.py:156` is a bug, not a style issue.** Inside
  `for te_new in [4,5,6,7,8,10]` it evaluates the fit at `ti_old`, not `ti_n`,
  so the `eps_step(%)` column of its Section 5 table is **constant at 52.4%
  for all six rows** and the `N_A/eps` column is meaningless.
- **The Rydberg energy appears with EIGHT distinct values across 17 sites**
  (13.605693122994 / …122990 / 13.605693 / 13.6058 / 13.6057 / 13.605 / 13.6,
  plus the legitimately different 13.598434599702). Bounded impact: ≤0.011%
  anywhere it is live, and **exactly zero** inside `corrcoef`. A traceability
  failure, not a numerical one.
- **`TE_GRID = np.logspace(...)` is written independently in NINE producer
  files** that build L. Textually identical today. Only
  `pre_assembly_check.py` guards against divergence, it is not invoked by
  `assemble_cr_matrix.py`, and **it has no entry for the ℓ-mixing grid.**
- **`verify_bundling_psm20.py:160-165` silently synthesises grids** if the
  files are missing — a direct violation of CLAUDE.md rule 2. Never fires
  today because the files exist.
- **`eigs[eigs < -1.0]` in `solve_cr.py:269` and `check_mz.py:10`**: confirmed
  still present, and confirmed **imported by nothing**. At the cold corner it
  turns M = 1.729e9 into **M = 1.467** — a wrong answer that reads as a physics
  finding. Latent, not live. That is luck, not design.
- **Three mutually inconsistent "ITER divertor" boxes are live**
  (`qss_analysis.py:362`, `plot_results.py:89`, `verify_eps_gridmap.py:58`).
  Only the last carries a citation (Guillemaut 2011).
- **`make_ch5_figures.py` is the best-engineered script in the repo** — reads
  the step from the CSV header rather than re-declaring it, derives indices in
  log space, raises rather than asserts. Use it as the template.
- **Clean, verified not assumed:** `cr_context.py`, `preflight.py` (its pinned
  SHAs match byte-for-byte), `verify_eps_gridmap.py`, `verify_ridge_mechanism.py`,
  `verify_lyman_trapping.py`, `verify_plateau_gridmap.py` (Class A clean),
  `verify_timescales.py`, `verify_boundary_descent.py`.

**Benchmark re-verified from the hash-checked matrix:** τ_QSS 22.7280 µs,
τ_relax 2.2769 ns, M 9981.9, cold corner 67.233 s, 19 points > 1 s. All ✓.

---

## 6. Do this next, in order

1. `sudo tlmgr install siunitx`, then compile `chapter3.tex` + `chapter6.tex`
   against `thesis_main.tex` to find macro collisions early.
2. Email the supervisor: submission deadline, and 7 chapters or 5.
3. **Decide the escape-factor geometry** (§5.1). It changes ADDENDUM A's
   cold-corner numbers.
4. **Audit the radiative block's ℓ-resolution** (§5.2, the 33% trap).
5. Re-run the Chapter 1 and Chapter 2 agents — both drafts are partial.
6. Write Chapter 5. Register is fixed; figures exist; use the **position-effect**
   framing from ADDENDUM B §B.2, and quote the density maximum as a **range**
   (7×10¹² – 5×10¹³ cm⁻³), which two independent agents now recommend.
7. Write Chapter 4 (Gate D reported as failing, plus the two survived attacks).
8. Chapters 7 + abstract, `\include` wire-up, consistency audit.

**Standing rule for every number written from here:** state its scope, its
operator (pre- or post-step), and its step size. All three have now produced
errors in this project.
