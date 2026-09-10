# Claim Hierarchy and Thesis Architecture

**Written 10 September 2026.** This document **supersedes**
`outputs/handover/thesis_architecture.md`, which predates `findings_09` and
`findings_10` and still carries the retracted framing (QSS breakdown, the
detachment identification, the M-colocation sentence, "38.7% in the divertor").

**Sources of truth, in precedence order:**

1. `outputs/findings_10_four_agent_review.md` incl. **ADDENDUM A** (Lyman
   trapping) and **ADDENDUM B** (first-principles re-derivation)
2. `outputs/handover/findings_09_central_quantity_misnamed.md`
3. `outputs/thesis_ready.md` (register, corrected 10 Sep — A8–A11 at ▶️)
4. `outputs/RESUME_HERE.md` §5 (three completed verification agents)
5. `outputs/CH5_EVIDENCE.md`, `outputs/derivation_*.md`

**Report only.** No chapter, script, data file or register entry was modified in
producing this document.

**Graduation states** (from `CLAUDE.md` / the backlog):
💡 idea · 📐 derived · 💻 implemented · ▶️ run · ✅ verified with sensitivity
check and written caveats.

---

# PART 0 — The central question and the one-sentence answer

> **Central question.** When can a steady-state collisional–radiative
> spectroscopic diagnostic be trusted after the plasma state changes, and what
> physics determines its failure *before ionisation balance has relaxed*?

## The thesis claim

> **A steady-state collisional–radiative line-ratio diagnostic becomes
> trustworthy only after the *ground-state reservoir* has relaxed — not after
> the excited states have. The excited manifold re-equilibrates in
> nanoseconds onto a reservoir that has not moved, and for the whole ionisation
> time $\tau_{\rm QSS}$ thereafter the emitted line ratio reports that
> *partial-equilibrium* state. A table indexed only on $(T_e, n_e)$ silently
> assumes equilibrium ionisation balance, so during that window it assigns the
> wrong physical meaning to a correctly measured ratio. The size of the
> misassignment is set exactly by the difference between the two emitting
> shells' ground-fed fractions, $d\ln R/d\ln u = f_3 - f_4$, and is therefore
> largest not where the timescale separation is largest, but where the
> ground-fed and recombination-fed supply channels are evenly matched — at the
> density where Griem's boundary level descends through the emitting shells.**

**Three things this sentence deliberately does not say**, each because the
verification work forbids it:

| Forbidden | Why |
|---|---|
| "quasi-steady state breaks down" | The QSS closure is exact to $10^{-8}$ on the plateau at the worst point on the grid (`findings_09` §1). Naming a breakdown that does not occur is an own goal. |
| "the error is largest where $M$ is largest" | $M_{\max}$ and $\varepsilon_{\max}$ are **52× apart in density** (`findings_09` W2, index corrected in `findings_09_index_correction.md`). |
| "…which is detachment" | Retracted. The ridge sits 5.2× below Guillemaut's detached band; at citable detached densities the ELM-averaged bound never reaches 10% (`findings_10` §1.3). |

## The infrastructure / thesis boundary

These are **not results** and must never be presented as such:

- "I built a 43-state hydrogen CR model" → Chapter 2, infrastructure.
- "I calculated hydrogen level populations" → Chapter 3, a solve.
- "I computed $\tau_{\rm QSS}$ and $\tau_{\rm relax}$ across a grid" →
  Chapter 3, a measurement of a *precondition*.

Each of these is a **premise of the argument**, and each earns its place only
by being needed for Claims D–F. A chapter that reports one of them without
saying which claim it feeds has failed the architecture.

---

# PART 1 — The claim hierarchy

Each claim carries: **(a)** the evidence, by file and section; **(b)** its
graduation state; **(c)** what is still missing. Where the evidence contradicts
the skeleton I was given, that is stated in a **⚠ Skeleton correction** block.

---

## CLAIM A — The model is physically credible

> The 43-state, $\ell$-resolved-to-$n{=}8$, $n$-bundled-9–15 hydrogen CR matrix
> is assembled from traceable atomic data, is dimensionally consistent, conserves
> particles, satisfies detailed balance in the appropriate limit, is converged in
> its state space to a stated tolerance, and reproduces published coefficients
> and timescales from an independent code.

**State: ▶️ Run — not ✅.** Four sub-claims pass; three are open; one external
gate (D) *fails* and must be reported as failing.

### A.1 Atomic data are traceable to named sources ✅ — the strongest rung in the project

- **Every coefficient traces to a named, dated source that survives into on-disk
  `source` columns** (`src/rates/*.py` headers, `data/**/*meta.csv`,
  `data/raw/adas/PROVENANCE.md`). The tallies are exact and reproducible:
  $K_{\rm exc}$ = 546 CCC + 36 CCC_n9 + 36 CCC_n10 + 201 V&S = **819**;
  $K_{\rm ion}$ = 36 CCC_TICS + 1 + 6 Lotz1968; RR = 43 × Johnson1972.
  **A provenance table for Chapter 2 can be generated from the repo, not
  reconstructed.**
- **Weak spots, all minor but all real:** `A_NP_TOTAL_FACTOR = 1.3`
  (`verify_bundling_psm20.py:56-62`) has **no source**; Anderson Table 2 is
  hand-transcribed (`anderson_benchmark_qc.py:67-68`) with no checksum;
  `prepare_adas.py` writes `*_interpolated.csv` files that **interpolate
  nothing** (it renames a column — a misleading filename); `README.md:64,203`
  and `radiative_rates.py:12` **cite the wrong Hoang-Binh paper** (1993/1990;
  the actual source per `data/raw/hoang_binh/Hoang_binh_README.md:46` is
  Hoang-Binh 2005, CPC **166**, 191). A dormant `Kramers_fallback` with a bare
  `1e-10` floor sits at `compute_K_VS.py:236-241`, verified never taken
  (237/237 rows are HoangBinh).
- **Contradicting evidence on the $A$-values, `RESUME_HERE.md` §5.2:** the stored
  Lyman $A$'s are the exact **non-relativistic infinite-nuclear-mass hydrogenic**
  values, reproduced to 1–4 parts in $10^5$, sitting **+0.058% above** the
  NIST/Wiese–Fuhr recommended set. Self-consistent and legitimate — **but NIST
  must not be cited as their source**, and `data/raw/wiese_fuhr.pdf` is committed
  and **never read by any code**.

### A.2 Units and dimensions ▶️ — the mislabel is fixed; the dangerous name is live

- **`S_grid` is no longer a defect, and the register is stale.** Commit
  `ffe1768` fixed the cm³/s → s⁻¹ label at all four sites
  (`assemble_cr_matrix.py:58,199,217,347`, `solve_cr.py:131`).
  **`thesis_ready.md` B8 still lists it as an open blocker — remove it.** The
  s⁻¹ reading was confirmed *physically*, not by inspection, by the cross-check
  $n(1s)/n_{\rm ion}\times n_e$ at [0,7] $= 2.05\times10^{16}$ against
  `CH5_EVIDENCE.md:282`'s independently measured $2.0\times10^{16}$.
- **Dangerous name, and it is LIVE, not hypothetical.** `CHI_H` is bound to
  $R_\infty hc$, not to the ionisation potential (13.598434599702, NIST ASD), at
  four sites — `verifych3_gb.py:57`, `verify_ridge_mechanism.py:224`,
  `make_ch3_figures.py:66`, `verify_fujimoto_table41.py:205` — and **all four
  use it inside a Saha–Boltzmann exponent**, where it is **0.0534% too large**.
  This is not a naming complaint; it is a wrong constant in an exponent.
- **Traceability failure:** the Rydberg energy appears with **eight distinct
  values across 40 independent literals**. Note in particular
  `13.605693122990` is baked into `L_grid` via `assemble_cr_matrix.py:76` while
  the Chapter 3 validators use CODATA-2018 `...994` — the matrix and its
  checkers disagree in the last digit. Bounded impact $\le0.011\%$, exactly zero
  inside `corrcoef`.
- **There is no dimensional gate at all.** No `pint`, no `unyt`, no
  `src/config/constants.py`; `astropy` is in `requirements.in:26` with zero hits
  in `src/`. Every one of the 40 sites is an independent literal.
- **Wrong constants:** $e = 4.80326\times10^{-10}$ esu (CODATA 2022:
  4.803204713) and $m_H$ set to the **proton** mass (0.0545% light for a neutral
  atom, +0.027% on Doppler width). Numerically immaterial; wrong objects.
- **Missing:** a units audit whose result can be stated in one line in Ch. 4,
  and the decision on whether to rename `CHI_H`.

### A.3 Detailed balance ✅ — and there are **two** results, which the docs conflate

- **Gate A** (`validate_gates.py:90-122`): max error **< 0.01%** over 819×3
  pairs. But `K_deexc` is *derived from* `K_exc`, so Gate A verifies detailed
  balance **by construction** and can only catch a wiring error. It belongs on
  the ladder as an implementation check, not as physics.
- **The physics content is in `src/parsers/qc_ccc.py:87-168`**, which tests
  whether Bray's **raw** cross sections satisfy microscopic reversibility:
  ratio **mean 0.9995, std 0.0032, 98.7% within 1%** (`ccc_qc_report.png`).
  That is a 0.05% mean bias with 0.32% scatter on data the project did not
  produce — a genuine external check, and it is currently invisible.
- **This closes an open action item.** `derivation_02:206` records the number as
  *"~0.43% (or 0.05%?)"* with the question unresolved. It is 0.05% mean /
  0.32% scatter. Neither reading was 0.43%.
- **Still open:** `derivation_02` §7 Steps 4–6 are marked **Pending**.

### A.4 Particle conservation ▶️ — correctly formulated, under-covered, unpersisted

- **The formulation is right and a previous misreading was caught.** Columns are
  *not* expected to sum to zero: $L$ is the **open** 43-state bound manifold and
  leaks to the ion, so $\sum_i L_{ij} = -K_{\rm ion}n_e$.
  `verify_timescales.py:207-213` contains the negative result correcting an
  earlier version that misread column sums as conservation. Keep that in Ch. 4.
- **Measured:** max relative column-sum error **2.59×10⁻¹¹**
  (`assemble_cr_matrix.py:269-282`, "Check D").
- **Three caveats that must be stated or closed:**
  1. Check D samples **20 of 400** grid points (`range(0,n_Te,10)`,
     `ne_grid[::2]`), while `derivation_01:142` claims "all 50×8". **The
     document overstates the code.**
  2. The closed **44**-state construction exists only as **prose** in
     `findings_10` §3.1. `grep` for `L_closed|np.block` returns **zero hits**;
     no residual was ever printed to a file. The five-digit agreement with
     1.6226 s is real but unreproducible from the repo as it stands.
  3. 2.59×10⁻¹¹ appears in **no file under `validation/`** — it survives only in
     markdown. Per `findings_10` §11 item 10: *"two files describing the same
     quantity, neither stamped, is how three retractions happened."*
- **Naming collision to fix:** "Gate D" means two different things in this repo
  — the ADAS external gate, and the assembler's conservation Check D. Rename one
  before Chapter 4 is written.
- Caveat that must not be softened: the closure adds an ion reservoir but does
  **not** close the electron balance ($n_e$ held fixed while the ion population
  moves) and adds no transport.

### A.5 State-space convergence and truncation ▶️ — one confound untested

- Measured: $n_{\max}$ truncation moves the observable **0.04% per top shell**
  (`findings_09` §6.1).
- **Open:** `verify_bundling_psm20.py` **has never been run** — no
  `outputs/bundling/` directory exists and a repo-wide search returns the script
  and nothing else. It closes the truncation confound that
  `chapter5_C5D.tex §sec:ridge_alternatives` admits is untested.
- ⚠ **But "it is one command" is wrong, and running it naively is a trap.**
  Two defects sit in the script: (i) lines 160–166 **silently synthesise grids**
  if the files are missing — a direct violation of `CLAUDE.md` rule 2; and
  (ii) `derivation_07:301` states `K_lmix` has non-zero entries only for
  $n = 2$–8, consistent with `compute_lmix.py:465` looping `for j in range(36)`.
  If so, the script reads **zeros** for indices 36–42 and returns a **false
  "bundling INVALID" from missing data**. Read those two blocks before running,
  or the result is uninterpretable either way.
- `master_plan_v2.md:343` records an explicit decision to declare C6 a scope
  limitation rather than test it, and `verify_fujimoto_table41.py:130-135` names
  this as a blocker on its own validation. **The honest options are: run it with
  the two defects understood, or declare it a scope limitation in Ch. 6. Not
  both, and not silence.**
- **Open, quantified:** the terminal shell is bad — $r_1(15)$ high by 4.9–6.2×
  against published values, while $n=10$ agrees to 4–20%
  (`thesis_architecture.md` Ch. 6 contents). Must be stated, not hidden.

### A.6 Radiative-transfer assumption ▶️ — measured, not assumed

- ADDENDUM A is the strongest single piece of Claim-A evidence in the project:
  trapping applied **self-consistently** ($\Theta_P$ depends on $n(1s)$ which
  depends on $\Theta_P$), all 14 Lyman channels, converged at all 400 points,
  swept over slab thickness rather than fixed, with **three validation gates
  passed before any result was read** — including one that FAILED at 1156% on
  first run and caught a $4\pi$ CGS/SI error.
- **The untrapped rebuild reproduces the canonical matrix exactly**
  ($\max|{\rm rebuilt}-{\rm canonical}| = 0$). Every trapped number is a
  difference against the real matrix.
- **Live defect, `RESUME_HERE.md` §5.1:** ADAS214 eq. 3.14.14 is the
  **isotropic / sphere-centre** case (ADAS g1), not a slab; a true slab is
  ≈2.1× smaller at large $\tau_c$. `escape_factor.py` documents it as a slab.
  This propagates into ADDENDUM A's **cold-corner** numbers. The $T_e\ge2$ eV
  conclusion is safe either way because $\Theta_P\approx1$ there.
- **Missing:** the geometry decision (relabel as isotropic, or redo as g2).

### A.7 Numerical validation ▶️ — sound where it exists, but one chapter claim has no script

- **Solid:** the superposition residual gate, **3.075×10⁻¹⁴ over 784 pairs**,
  grid-wide, raising rather than warning. This is the healthy sub-item.
- **Solid but narrow:** eigenvalue condition numbers $1/|y^Hx|$ are $O(1)$
  (1.71/1.98 at [23,5], 3.26/2.59 at the cold corner) **despite**
  $\|L\|_1 = 2.0\times10^{14}$ — at **3 of 400 points**, narrative only.
  Perturbation and float32 evidence likewise comes from ADDENDUM B, not from a
  repo script; `cond(L_{EE})$ on disk covers **4 point-variants at $T_e=10$ eV**,
  spanning $7.2\times10^3$–$1.6\times10^5$.
- **Tolerance evidence that is real:** rtol $10^{-9}$ vs $10^{-12}$ differ by
  $1.20\times10^{-8}$; `expm` vs `expm_multiply` by $1.13\times10^{-10}$
  (2 points, not written to disk).
- ⚠ **TRACEABILITY FAIL, and it is in a chapter.** `chapter3.tex:719-721`
  asserts $\kappa(L_{FF})\in[1.48\times10^3, 1.74\times10^5]$ **grid-wide**.
  **No script in the repo computes it** — `np.linalg.cond` appears at exactly
  two sites, neither on the full 43×43 $L$. Either produce it or delete the
  sentence. This is the single most quotable unsupported number currently
  sitting in written LaTeX.
- **Also unlocated:** A7's rtol $10^{-6}$–$10^{-11}$ convergence table has no
  identified producing script (`master_plan_v2.md:121` still lists finding it as
  open). The semigroup propagator check was **withdrawn by the author as
  tautological** (returns 0.000e+00) — a good call, and worth reporting as one.
- The caveat against quoting "67.2 s" to three figures is **physics**
  ($d\ln\tau_{\rm QSS}/d\ln T_e = -13.3$), not roundoff.

### A.8 External comparison ▶️ / one FAIL

- **PASS:** Fujimoto *Plasma Spectroscopy* App. 4B at a point that lies exactly
  on this grid — excited response 0.34×, ground depletion 1.77×
  (`thesis_ready.md` A2). Agreement to a factor of a few is the meaningful
  comparison; Fujimoto's $t_{\rm res} = \max_p t_{\rm tr}(p)$ is not $1/|\lambda_1|$.
- **PASS:** Fujimoto Table 4.1 $r_0$ to 0.1–1% for $n\ge4$; $r_0(2) = 0.7392$ vs
  0.730, the discriminating case.
- **OPEN, and it gates the paper:** the low-$p$ $r_1$ deficit — factor **8.3** at
  $p=3$ and **4.4** at $p=4$ against Fujimoto Table 4.1(b) at $n_e = 10^{12}$,
  one grid interval from the ridge (`findings_09` §9 item 4; `findings_10`
  §4.4, §10 item 3). B1 ruled out the excitation data ($\Delta\ell=1$ agrees at
  median −5.0% over 120 comparisons). **A different suspect is required.**
- **PASS, and it is a *self-caught* one worth advertising:**
  `verify_fujimoto_table41.py`'s no-proton-$\ell$-mixing variant produces
  $r_0(2) = 104.5$, flagged by the script itself as *"unphysical, a level cannot
  exceed Saha equilibrium"*, while production gives 0.7392 against Fujimoto's
  0.730. A check that fires on a deliberately broken input is severe; say so.
- **FAIL, and must be reported as failing:** Gate D against ADAS SCD/ACD.
  Measured $\eta = {\rm SCD_{model}}/{\rm SCD_{ADAS}} \in [5.73, 8346]$;
  **0 of 400 points within a factor 2** (`validation/gate_summary.txt`).
  `acd_adas` is loaded at `validate_gates.py:299` and **never used again**, so
  the recombination half is unimplemented despite the docstring, and the
  docstring names files (`data/processed/ADAS/SCD96_long.csv`) the code does not
  read. Diagnosis: `SCD_model` sums ionisation over the **full** steady state,
  but ADAS SCD is the *ionizing* coefficient. **Do not make it pass.**
- ⚠ **Gate D's status is reported three incompatible ways** — `README.md:119`
  "DOCUMENTED", `validate_gates.py:47` "PASS at $T_e\ge5$ eV",
  `gate_summary.txt` "FAIL (0%)". Pick one, which must be FAIL, and delete the
  other two. This is the same two-files-one-quantity condition under which the
  −46% H$\alpha$ artifact propagated.
- ⚠ **`data/processed/adas/report_adas.md` cannot be cited.** Its explanation
  quotes $\eta\in10^2$–$10^4$ scaling as $n_e^{1.5}$, which does **not** match
  the current post-fix CSV; and its fallback claim (H$\alpha$/H$\beta$ = 4.1
  against "ADAS PEC 2.8–4.5, Summers 2006") rests on a PEC/adf15 file that
  **exists nowhere in `data/`**, with no producing script.
- Rewrite `chapter3.tex:1377-1387`, which promises a comparison that does not
  exist.

### A.9 Independent reproduction ▶️ — strong, but the register's flagship is not what it claims

- **Bit-for-bit reproduction ✅.** A1/A2 grids bit-identical (including the
  LSODA-derived `eps_res`); A8/A9/A11 **byte-for-byte** (680/680; 784 rows at
  0.000e+00); A12 exactly (19/400 differ, and 19/19 have
  $\tau_{\rm relax}^{\rm new} = \tau_{\rm QSS}^{\rm old}$ exactly, confirming the
  ladder shift). `make_ch5_figures.py:378-458` enforces
  recomputation-vs-CSV at rtol $10^{-9}$ on **every run** and raises — that is
  the pattern the rest of the repo should copy.
- **First-principles re-derivation ✅** (ADDENDUM B): all six Chapter 3 core
  results re-derived with code unopened and predictions written first. Every
  number in `chapter3.tex §sec:ground_fed_fraction` reproduces.
- ⚠ **A5, the "strongest single check", is not a check — it is a citation.**
  `thesis_ready.md:135` grades the May/August cross-validation ✅ and calls it
  *"the strongest single check"*. But **the May run was never re-run.**
  `derivation_04_two_timescales.md:375-381` says so explicitly — *"a reported
  result, not independently re-run… Treat it as strong documented evidence, not
  as self-verified"* — and `HANDOVER.md:37` puts the source document on the skip
  list. **Demote A5 to ▶️**, or re-run `Balmer_transient_ratio.py` and earn the
  mark. As it stands the thesis's headline credibility claim rests on a document,
  which `CLAUDE.md`'s ground-truth hierarchy ranks last.
- ⚠ **Two integrity items that belong in Ch. 4's corrections section:**
  `audit_writers.py` ran and **reports an unfixed failure** (D.4 — three `.npy`
  paths × 2 writers × 6 readers); and commit `ffe1768` carries a **false message**
  claiming the `eigs < -1.0` filter was removed. It survives at
  `src/rates/solve_cr.py:269` and `src/rates/check_mz.py:10` (imported by
  nothing — latent, not live; at the cold corner it would turn $M = 1.73\times10^9$
  into $M = 1.467$, a wrong answer that reads as a physics finding).
  **`CLAUDE.md`'s known-issue table points at the wrong file** —
  `qss_analysis.py:137` now reads `eigs < 0.0`. Fix the table, not the code.

### What Claim A still needs

1. Run `verify_bundling_psm20.py` (one command).
2. Decide the escape-factor geometry (§A.6).
3. Audit the radiative block's $\ell$-resolution against `cr_context.py` state
   ordering — the **33% trap**: $A(2p\to1s) = 6.2649\times10^8$ is $\ell$-resolved,
   $A(n{=}2\to n{=}1) = 4.6986\times10^8 = \tfrac34\times$ that is $n$-resolved.
   Putting the 2p value on an $n$-resolved element overestimates Ly-$\alpha$ by
   33.3% (`RESUME_HERE.md` §5.2). Given this repo's $\ell$-mixing history this is
   the single highest-risk unaudited item in Claim A.
4. A named detailed-balance residual with its tolerance.
5. One provenance table for atomic data.

---

## CLAIM B — The excited states possess a fast relaxation scale

> The excited manifold has a relaxation time $\tau_{\rm relax} = 1/|\lambda_1|$
> of order nanoseconds, and adiabatic elimination of that manifold — the
> quasi-steady-state closure, $\dot{\mathbf n}_E = 0$ — is **exact** on the
> plateau, not approximate.

**State: ✅ Verified.** This is the most solidly established claim in the thesis
and it is the *reversal*.

### Evidence

- $\tau_{\rm relax} = 2.2769$ ns at the benchmark $L[23,5]$; grid range
  0.87–38.9 ns (`thesis_ready.md` A1, re-verified `RESUME_HERE.md` §5).
- Three independent implementations agree bit-for-bit (`qss_analysis.py`,
  `validate_gates.py` Gate E, `timescales_unfiltered_CHECK.npz`); Framing A
  ($\lambda_1$ of the full matrix) agrees with Framing B to 0.0098% at the
  benchmark and <0.35% grid-wide.
- **The closure residual, measured along the trajectory** (`findings_09` §1):

  | point | error vs CRE (what was reported) | **QSS closure error** |
  |---|---|---|
  | benchmark [23,5] | $6.34\times10^{-2}$ | $8.66\times10^{-6}$ |
  | worst [0,4] | $3.869\times10^{-1}$ | $6.73\times10^{-9}$ |

  **Four orders of magnitude. That gap is the thesis.**
- Mathematical basis for exactness (ADDENDUM B §B.1): $-L_{EE}$ is a Z-matrix
  with strictly positive column sums, hence a non-singular M-matrix, hence
  $(-L_{EE})^{-1}\ge0$ elementwise; equivalently $L_{EE}$ is Metzler and Hurwitz.
  Entry $(p,q)$ is the expected time an atom created in $q$ spends in $p$ before
  leaving the manifold. **Measured: 0 negative entries at all 400 points**,
  minimum component $+6.019\times10^{-13}$.
- Why no intermediate mode can spoil it: at [1,4] the eigenvalues are
  $-4.29$, $-1.60\times10^8$, $-3.44\times10^8$ s⁻¹ — a gap of $3.7\times10^7$
  (`findings_10` §3.2).

### The distinction that *is* the thesis

$$\text{QSS} \equiv \dot{\mathbf n}_E = 0 \qquad \textbf{not} \qquad \dot n_g = 0$$

Every sentence in every chapter must respect this. `chapter4.tex`
§`sec:qss_ratio` violates it, and that single false sentence — *"these ratios
depend on $(T_e,n_e)$ but not on the absolute normalisation of $\mathbf n$"* —
is the origin of the entire retracted framing (`findings_09` §2).

### Caveats to carry

- Under strong Lyman trapping the fast scale is **not** invariant: at [0,3],
  $\tau_{\rm relax}$ goes $8.88\times10^{-9}\to2.43\times10^{-7}$ s
  (**×27.3**) at $D = 20$ cm, while at the benchmark it moves **+0.78%**
  (ADDENDUM A §A.5). `thesis_ready.md:388` ("within 1%") and
  `derivation_07:299-300` ("$M$ falls 158×") are **both correct in different
  regimes and neither states its regime.** Both sentences must carry scope.

### What Claim B still needs

Nothing to establish it. It needs *placement*: Chapter 3 must state it before
anything else, because it frames everything downstream.

---

## CLAIM C — The reservoir evolves on a different scale

> The ground state and ion reservoir turn over on $\tau_{\rm QSS} = 1/|\lambda_0|$,
> which exceeds $\tau_{\rm relax}$ by $M = \tau_{\rm QSS}/\tau_{\rm relax}
> \sim 10^2$–$10^9$. That separation is the *necessary condition* licensing
> Claim B's closure. **It is not a predictor of diagnostic error.**

**State: ▶️ Run.** The numbers reproduce; the *interpretation* required three
retractions.

### Evidence

- Benchmark $L[23,5]$: $\tau_{\rm QSS} = 22.728$ µs, $M = 9981.9$
  (`CLAUDE.md`, re-verified `RESUME_HERE.md` §5 and ADDENDUM B §B.1 to four
  digits).
- Grid-wide, over **all 400 points**: $M$ from **86.77** to
  $1.72928\times10^9$; $\tau_{\rm QSS}$ from **75.4 ns** ([49,7]) to 67.2 s.
- Physical identification: $\lambda_0 = 2.286\times K_{\rm ion}(1S)\cdot n_e$ —
  **the CR ionisation time of ground-state hydrogen, not an equilibration time**
  (`thesis_ready.md` A1).
- Fujimoto App. 4B independently gives ground-state depletion $\sim10^{-4}$ s at
  a point on this grid against the model's $1.7657\times10^{-4}$ s.

### ⚠ Skeleton correction 1 — Claim C must be prevented from implying prediction

The skeleton reads "the reservoir evolves on a different scale", which is true,
but the thesis's failure mode has been to slide from *there is a separation* to
*the separation locates the failure*. The verification work forbids that:

- $M_{\max} = 1.73\times10^9$ at **[0,0]**, $n_e = 10^{12}$;
  $\varepsilon_{\max}$ at **[0,4]**, $n_e = 5.18\times10^{13}$ — **52× apart in
  density.** At the $M$ maximum $\varepsilon = 0.12$, the 75th percentile
  (`findings_09` W2; index per `findings_09_index_correction.md`).
- $\mathrm{corr}(\log M,\log\varepsilon) = +0.757$ raw, $+0.33$ (findings_09) /
  $+0.274$ (independent recomputation) controlling linearly for $(T_e,n_e)$, and
  **flips sign to $-0.16$ / $-0.442$** under quadratic control. A bare
  $e^{13.6/T_e}$, containing no dynamics whatever, correlates at $+0.708$.
  $\log M$ is **93% explained by $(\log T_e,\log n_e)$ alone.**
  **Quote the sign, not the magnitude** — over eight scope/basis combinations
  the quadratic partial ran $-0.22$ to $-0.70$.
- **Greenland (2001)**, *J. Nucl. Mater.* **290–293**, 615, already concluded
  that CR validity criteria "are not related to the equilibrium time-scales" and
  that "the eigenvalues have secondary importance" — in general form, 25 years
  ago. **The thesis's contribution is quantification for a specific diagnostic,
  not the discovery.** Cite him in Chapter 1, not in a footnote.

### ⚠ Skeleton correction 2 — three separate scope traps live inside Claim C

1. **$M$ floor.** $M\ge86.8$ quotes the floor of a set that excludes it: that
   point has `window_ok = False`. Over the analysed set the floor is $M\ge902$
   — and that floor is **imposed**, since `window_ok` $\equiv M > 900$
   (win_lo × win_hi = 30×30). A constructed cut is not a measurement.
2. **$\tau_{\rm QSS}$ floor.** 1.18 µs is the minimum over the 346-point
   $M>900$ subset; the unrestricted minimum is **75.4 ns** and **46 of 400
   points lie below 1.18 µs**. `verify_ch3_claims.py` reports this as its sole
   FAIL. `chapter3.tex` Eq. `M_range` is already correct.
3. **Pre- vs post-step operator.** $M$ at "the benchmark point" has **three**
   values in three files — 9982 ($L[23,5]$), 8243 ($L[24,5]$, post +4.81% step),
   4856 (post +0.6 eV step). Each individually correct; none distinguished.
   **Every $\tau_{\rm QSS}$, $\tau_{\rm relax}$ and $M$ in the thesis must name
   which operator it belongs to.**

### Caveats that bound Claim C's *physical* content

- **Boundary-dependence.** Closing with a 44th ion state changes $\tau_{\rm QSS}$
  from 67.2 s to 1.62 s at the cold corner — a factor **41**. But grid-wide the
  factor is min 1.00, median 1.00, max 41.4, and it is **unity everywhere
  $\tau_{\rm QSS}$ approaches the event duration** (`findings_10` §3.1).
  $\tau_{\rm relax}$ is unchanged by closure.
- **Transport, and this is the real threat.** $\tau_{\rm QSS}$ at [0,4] is the CR
  ionisation time of a *stationary* neutral. A 1–3 eV D atom crosses a 10 cm
  plasma in **6–10 µs** against the model's **233 ms**. An effective ground-state
  renewal time of ≈26 µs already drops the [0,4] bound below 10%. **This, not
  closure, is what can kill the cold-corner magnitude** (`findings_10` §4.2).
  It cannot be settled inside a 0-D model. State it as a conditional: *if* the
  ground-state neutral density is frozen for the duration of the event, then…

### What Claim C still needs

1. A single notation decision fixing pre-/post-step $M$, applied everywhere.
2. A stated scope on every extremum (`writing_specification.md` N2).
3. The Greenland citation, in the introduction.

---

## CLAIM D — Timescale separation creates a partial-equilibrium state

> Between $\tau_{\rm relax}$ and $\tau_{\rm QSS}$ the system occupies a
> well-defined intermediate state: the excited manifold in equilibrium with the
> **post-step operator** but the **pre-step reservoir**,
> $$\mathbf n_E^{\rm PE} = -L_{EE}^{-1}\!\left(L_{Eg}\,n_g^{\rm old} + \mathbf S_E\right),$$
> with $L_{EE}$, $L_{Eg}$, $\mathbf S_E$ evaluated at the **new** $(T_e,n_e)$.
> It is computable in one linear solve and requires no stiff integration.

**State: ✅ Verified.**

### Evidence

| | analytic | measured (LSODA) | agreement |
|---|---|---|---|
| cold corner [0,0] | 0.240751 | 0.240747 | **6 digits** |
| benchmark [23,5] | 0.329030 | 0.324170 | 1.5%, explained |

The 1.5% is window sampling, not model error: back-extrapolation recovers
0.329045 (ratio **1.000047**); the fitted decay time is $1.079\,\tau_{\rm QSS}$
*without being told* $\tau_{\rm QSS}$; $R^2 = 0.9999999$; the observed window
deficit 0.98527 matches the predicted 0.98406 to $1.2\times10^{-3}$
(`thesis_ready.md` A4).

**Window sensitivity done** (method 4.4): repeated at window factors 10, 30, 100;
$\tau_{\rm fit}/\tau_{\rm QSS} = 1.0774/1.0791$; every $\varepsilon$ identical.
30 is a default, not load-bearing — but it is the right default, because at 100
the $M>10^4$ guard skips the benchmark point itself.

**Superposition, the severe check:** $\mathbf n_E(x) = \mathbf n^{(0)} + x\,\mathbf n^{(1)}$
holds to $3.075\times10^{-14}$ across all 784 grid-point/direction pairs
(`thesis_ready.md` A3). Per ADDENDUM B §B.3 item 2, **this and the
reduced-vs-full $R$ test are the severe checks** — they catch a deliberate 3↔4
shell swap. The $\tanh$ gate does not and must not be printed as validation.

### The naming, which is the whole correction

What was called "QSS breakdown" is the distance from the partial-equilibrium
state to the **CR-equilibrium** state. `derivation_07` §5 already had the honest
sentence — *"after Stage 1 the excited manifold is in equilibrium with a stale
reservoir"* — and it went unacted-on for four days (`findings_09` §2.1).

### Caveats

- The name of the metric must change throughout; patch `chapter4.tex`
  §`sec:qss_ratio` first and everything downstream inherits
  (`findings_09` §9 item 1).
- `findings_09` §5.1: §`sec:eps_bar` models the decay as
  $\varepsilon_{\rm res}e^{-t/\tau_{\rm QSS}}$ and labels it "the Stage 1 QSS
  error decay" while using the **Stage 2** constant. The label and the constant
  disagree inside one sentence.

### What Claim D still needs

A notation decision ($\varepsilon^{\rm QSS}$ vs $\varepsilon^{\rm CRE}$, or
better names) fixed once — and the resolution of the **ten**-way `eps_step` name
collision (`findings_10` §5.11), not the "at least six" in `thesis_ready.md` B6.
Two of the ten are mutually inconsistent hardcoded fits
(`verify_partition.py:134`, `plot_results.py:459`) giving a **9.6% spread** at
$T_e = 3$ eV, neither traceable to any run.

---

## CLAIM E — Spectroscopy sees that partial-equilibrium state

> The observable is a line ratio built from the same populations, so the
> partial-equilibrium state is what the spectrometer reports. A table indexed
> only on the *final* equilibrium $(T_e,n_e)$ therefore assigns the wrong
> physical meaning to a correctly measured Balmer ratio for a time
> $\sim\tau_{\rm QSS}$.

**State: ▶️ Run.** The observable link is ✅; the *magnitude* is the part that
was demoted.

### Evidence that the observable link holds

- **A6, and this is the strongest robustness result in the project.**
  $\ell$-populations move as a **rigid body**: $f(4S) = 0.0608$ vs
  $f(4F) = 0.0606$; the 4F fraction of the $n=4$ shell is 0.4361–0.4375 against
  the statistical 14/32 = 0.4375, so proton-impact $\ell$-mixing drives $n=4$
  statistical to <0.3% everywhere on the grid. Populations come from the solve —
  grep for `statistical`, `stat_weight`, `(2l+1)` returns **zero hits**.
- **Shell ratio vs A-weighted line ratio:** $H\alpha/H\beta$ gives 0.386683
  against the shell value 0.386903 — **0.06%** (`findings_09` §6.1);
  $\varepsilon_{\rm step}$ agrees to **0.014%** (`thesis_ready.md` A5).
  Photon- vs energy-weighting: $6.7\times10^{-16}$.
- 4F is **correctly absent** from H$\beta$ ($\Delta\ell = 2$, E1-forbidden), so
  the $n=4$ shell sum contains a state the line cannot see — and the measured
  consequence is $\varepsilon$(Balmer)/$\varepsilon$(shell) = 0.978–0.9999,
  worst case 2.2% at the lowest density.
- **C8 is closed with a number**, and the caveat in `divertor_map.txt` should be
  deleted and replaced by these numbers.

### Evidence for the magnitude, restated inside its scope

| scope | pairs | ELM lower bound > 10% | worst |
|---|---|---|---|
| all `window_ok` (**as previously published**) | 680 | 202 | 0.3868 |
| **$T_e\ge2$ eV — the defensible range** | 448 | **45** | **0.1748** |
| $T_e\ge2$ eV **and** $n_e\ge10^{14}$ (citable divertor density) | 108 | **0** | 0.0717 |

**Quote the middle row.** 45 of 448 pairs above 2 eV; worst **17.5%** at
$T_e = 2.02$ eV, $n_e = 1.93\times10^{13}$; at citable divertor densities the
ELM-averaged lower bound peaks at 7.2%. The benchmark point does **not** break
down (1.2% at ELM timescales).

### ⚠ Skeleton correction 3 — the magnitude is a *rate*, not a number

**$\varepsilon_{\rm plateau}$ is LINEAR in the temperature step, and the step
size is a grid artifact** (`RESUME_HERE.md` §5.3):

| step | max $\varepsilon$ (all) | $T_e\ge2$ eV | $T_e\ge2$ & $n_e\ge10^{14}$ |
|---|---|---|---|
| $k=1$ (+4.81%) | 38.690% | 18.073% | 10.355% |
| $k=2$ (+9.85%) | 87.410% | 38.203% | 23.380% |
| $k=4$ (+20.68%) | 190.376% | 81.133% | 58.531% |

"+4.81%" exists only because someone chose 50 log-spaced points between 1 and
10 eV. **"38.7%" means "38.7% per 4.81% temperature step" and is meaningless
quoted bare.** An examiner asking "what if the ELM is 10%?" gets 87%.

> **The invariant is $d\varepsilon/d\ln T_e$ — and no script currently
> headlines it.** This is an architecture-level instruction: the headline
> quantity of Chapter 5 must be the derivative, with the per-step value quoted
> underneath it as an illustration at a stated step size.

Note also (`findings_10` §8.5) that the four-interval numbers are exact solves
but sit **3.8× beyond** A9's validated linearisation ceiling; the
"do not use the linearised form" instruction must travel with them.

### Further scope corrections carried into Claim E

- These are **(point, direction) pairs**, not grid points: 105 heat + 97 cool
  over **113 distinct points of 400**, and the count is **201**, not 202
  (`findings_09` W5, W3).
- The "confined to $T_e\le2.947$ eV" boundary is a **level set of two arbitrary
  constants** (10.0 eV at 1%, 2.947 at 10%, 1.76 at 20%), monotone and
  knee-free. That it lands on benchmark index 23 is coincidence and will not
  read as one (`findings_09` W4).
- **The 10% threshold is doing real work:** counts at 5% / 10% / 20% are
  **348 / 202 / 61**. The count is not robust to the threshold; the worst case
  is. The threshold's *rationale* is sound (matched to combined ADAS PEC ~5% and
  atomic-data ~5% uncertainty) and should be kept.
- **The ELM time-average is close to a no-op on the counted set:**
  LOWER/UPPER > 0.99 at 127 of 202, median 0.9971 — the 202 are effectively
  *selected by* $\tau_{\rm QSS} > \tau_{\rm ELM}$ (`findings_09` §4.2).
  Do not imply the averaging does work.
- **But the bound is near-exact for this observable:** direct 43-state
  eigen-propagation gives true/lower = **0.9999–1.0066** at seven points
  including the worst (`findings_10` §3.2). W3's "not a bound on either side"
  and this are both true: it is not a *guaranteed* bound, and it is empirically
  accurate to 0.7%. Say both.
- **The worst case is edge-truncated:** all eight $n_e$ columns peak at the
  lowest available $T_e$; $d\ln\varepsilon/d\ln T_e\approx-1.27$ and still
  rising at the edge. Must be *"at least"*. The $n_e$ direction **is** genuinely
  interior.
- **The ±5% step is not an ELM.** An ELM also raises $n_e$; the map steps $T_e$
  at fixed $n_e$. Joint steps give comparable $\varepsilon_{\rm plateau}$ but a
  time-averaged bound that can fall **4×**. **A joint-step map has not been
  run** and an examiner who knows ELM physics will ask for one
  (`findings_10` §8.3).

### What Claim E still needs

1. **The joint $(T_e,n_e)$ ELM step map.** Same cost as the existing map.
2. **The density line plot** — $\varepsilon$ vs $n_e$, one line per $T_e$.
   Non-negotiable: if it does not look like a ridge, the word "ridge" must not
   be used (`plan_to_submission.md` T1.3).
3. A single decision on the observable ($n_3/n_4$ or A-weighted
   $H\alpha/H\beta$) — they agree to 0.06%, so either is defensible, but say
   which, once.

---

## CLAIM F — The magnitude has a mechanism

> With $n_p = a_p u + c_p$ — the exact two-channel split of the excited
> populations into a ground-fed part and a recombination-fed part — the ratio
> sensitivity to the reservoir is exactly
> $$\frac{d\ln R}{d\ln u} = f_3 - f_4, \qquad f_p = \frac{a_p u}{a_p u + c_p},$$
> so the diagnostic error is largest where the two supply channels compete.

**State: F.1 ✅ · F.2 ✅ · F.3 ▶️ · F.4 ▶️ conditional.** This is the strongest
part of the thesis and it is currently undersold.

### F.1 The split is exact ✅

Fujimoto eq. (4.20) realised in this matrix:
$\mathbf n^{(0)} = -L_{EE}^{-1}\mathbf S_E$,
$\mathbf n^{(1)} = -L_{EE}^{-1}L_{Eg}n_g^{\rm old}$.
Superposition to $3.075\times10^{-14}$ over 784 pairs; the identity
$\int(f_3-f_4)\,d\ln x = \ln[R(x_{\rm new})/R(1)]$ to $3.3\times10^{-16}$ by two
independent quadratures (`thesis_ready.md` A3).

**Honest note that must survive into the text:** the identity check validates
*implementation*, not physics — it is exact by construction. What it establishes
is that the coded $f_3-f_4$ really is $d\ln R/d\ln x$ for the coded family.

### F.2 The functional form ✅ — and the width is not incidental

$f_m(x) = 1/(1+e^{-(x-x_m)})$ with $x_m = \ln(c_m/a_m)$: the two ground-fed
fractions are **the same curve displaced**. Hence
$\max|f_3-f_4| = \tanh(|\Delta|/4)$ with
$\Delta = \ln[(a_3/a_4)/(c_3/c_4)]$, and the corollary
$|d\ln R/d\ln b_1| < 1$ — **a line ratio can never respond to the ground-state
reservoir faster, in relative terms, than the reservoir itself moves.**

ADDENDUM B strengthens this: a general Hill function maps to $\sigma(m(X-X_0))$
with width $1/m$; here **$m = 1$ *because* the excited populations are affine in
$n_g$**. Width 1 ⟺ Hill coefficient 1 ⟺ exact linearity in the reservoir. The
whole 42-state network enters only through the location $x_p$.

**Invariance, verified:** $\Delta$ and the $\tanh$ bound are invariant under
rescaling the $n_g/n_i$ normalisation (exact under factors $10^{10}$ and
$10^{-7}$); $x_3, x_4, f_p$ individually are **not**. $\Delta$ is also
insensitive to $\ell$-weighting — 1.94561 / 1.94474 / 1.94447 / 1.94471 /
1.94574, a **0.07% spread**. Say this next to the 0.212.

**Do not print the $\tanh$ gate as validation.** Fed corrupted input it still
passes (3↔4 swap leaves $|\Delta|$ and $|\ln r|$ unchanged); it is a theorem
given $a,c\ge0$ and can only fail on an arithmetic bug (ADDENDUM B §B.3 item 2).

### F.3 The location has a physical mechanism ▶️ — three independent routes

The best-evidenced result in the thesis (`findings_10` §2):

| test | **prediction, written first** | measured |
|---|---|---|
| ridge moves with shell pair, Griem $n^{-17/2}$ | $(5/4)^{8.5} = 6.7$ | (3,4)→(4,5) ratio **7.2** |
| absolute density, Griem LTE criterion at $n=4$, $T_e=2$ eV | $2.05\times10^{13}$ | ridge at **1.93×10¹³** |
| independent diagnostic, `boundary_descent.csv` | boundary between 3 and 4 at the ridge | $\bar n = 3.08$–3.25, **all $T_e$** |

**The competing hypothesis was refuted:** "both shells become ground-fed because
collisional coupling dominates" is wrong — at high $n_e$ both $f$'s fall toward
*zero* (continuum-fed): at [23,5]-scale conditions $f_3 = 0.071$, $f_4 = 0.011$
at $n_e = 10^{15}$.

**What would have refuted it:** the ridge failing to move with the shell pair,
or moving the wrong way. **Neither occurred.**

### ⚠ Skeleton correction 4 — the density maximum is a POSITION effect

The skeleton's "competition between channels" is right but underspecified, and
the obvious reading of it — that the maximum sits where the *span*
$\tanh(|\Delta|/4)$ peaks — is **falsified** (ADDENDUM B §B.2).

$\Delta$ is nearly **flat** across density: 0.979 at $10^{12}$, 1.945 at
$1.4\times10^{14}$, 1.939 at $10^{15}$; $\tanh(\Delta/4)$ spans only
0.394→0.450, a **14% span**, while the operating point
$\ln(u_{\rm CRE}/u_{\rm peak})$ sweeps from **+1.29 to −3.29**:

| $j$ | $n_e$ | $\Delta$ | $\tanh(\Delta/4)$ | $\ln(u_{\rm old}/u_{\rm peak})$ | $|\bar S|$ | $\varepsilon$ |
|---|---|---|---|---|---|---|
| 0 | 1.00e12 | 0.9655 | 0.2368 | **+1.2877** | 0.1754 | 4.873% |
| 2 | 7.20e12 | 1.6662 | 0.3940 | **+0.3842** | 0.3884 | 11.053% |
| 3 | 1.93e13 | 1.8857 | 0.4394 | **−0.2487** | 0.4261 | 12.166% |
| 5 | 1.39e14 | 1.9403 | 0.4503 | −1.7695 | 0.2288 | 6.361% |
| 7 | 1.00e15 | 1.9368 | 0.4496 | −3.2910 | 0.0668 | 1.831% |

> **The maximum sits where $\ln(u_{\rm CRE}/u_{\rm peak})$ changes sign.** The
> $\tanh$ cap is not what locates it.

This is more specific than "sensitivity-dominated" and **must replace that
phrasing in Chapter 5**. It also forbids the inference that the effect vanishes
at high $n_e$ — on this grid it does not, because the model is **open**
($\mathbf S_E$ is an externally imposed source not tied to $n_i$ by Saha), so
the two channels **can never merge by construction**. The grid never reaches
LTE. This was a prediction written in advance, falsified, and the falsification
produced a better result — the model behaviour that ought to be advertised.

### ⚠ Skeleton correction 5 — the temperature trend is 85% a step artifact

At the ridge column ($j=3$, heating):

| $T_e$ | $|\ln x|$ | $|f_3-f_4|$ | $\varepsilon_{\rm plateau}$ |
|---|---|---|---|
| 1.000 | 0.678 | 0.427 | 0.371 |
| 1.600 | 0.446 | 0.463 | 0.232 |
| 2.947 | 0.276 | 0.437 | 0.111 |
| 5.179 | 0.179 | 0.390 | 0.071 |
| 9.541 | 0.124 | 0.353 | 0.044 |

$\varepsilon$ falls **8.4×**; $|\ln x|$ falls **5.5×**; the intrinsic
sensitivity $|f_3-f_4|$ is **flat to ±13% and non-monotonic** (it peaks at
1.6 eV). *"The error decreases monotonically in temperature"* is a statement
about how far a fixed 4.81% $T_e$ step moves the ground state, not about the
diagnostic.

**The physically interesting statement is the opposite of the one written:** at
the ridge density the diagnostic's sensitivity to ground-state lag is
essentially temperature-independent over the whole decade.

### F.4 The ridge *location* is conditional ▶️

- **Proportional to the assumed neutral density.** One decade in $n(1s)$ moves
  the ridge roughly one decade in $n_e$ (`findings_10` §4.4). "The ridge sits at
  $n_e\approx2\times10^{13}$" is a statement about *this model's ionisation
  balance*, not about a divertor.
- **Quote it as a range: $7\times10^{12}$–$5\times10^{13}$ cm⁻³.** The grid
  resolves only 0.43 decades per column; a parabolic fit in $\ln n_e$ puts the
  vertex at $1.65\times10^{13}$, a 15% shift. Two independent agents reached
  this (ADDENDUM B §B.3 item 3).
- **"Roughly fixed density" needs its resolution stated:** the sub-grid peak
  runs $3.66\times10^{13}$ (1 eV) → $1.51\times10^{13}$ (9.5 eV), range 2.56×,
  drifting **downward**, while Griem predicts $\propto\sqrt{T_e}$, i.e. **upward**
  by 3.2×. The drift is inside one grid interval (2.68×) so it cannot be
  resolved. Write *"constant to within one grid interval, a factor 2.68, over
  1–10 eV"* (`CH5_EVIDENCE.md:63-65`).
- **The pair, not "the Balmer diagnostic".** An $H\alpha/H\gamma$ ($n=3/n=5$)
  diagnostic has its worst density at $7.2\times10^{12}$, a factor 2.68 lower.
  Write *"the $H\alpha/H\beta$ ratio is least reliable at…"*, never *"the Balmer
  diagnostic is"*.
- **The two scripts use different point sets** (ADDENDUM B §B.3 item 4):
  `verify_plateau_gridmap.py` excludes $M\le900$; `verify_ridge_mechanism.py`
  applies no such filter, and **54 of 400 points have $M\le900$**, all at
  $j\ge4$. The ridge script's $j=6,7$ statistics include points where the
  plateau state does not physically exist.
- **The cold-row ridge is not robust under trapping** (ADDENDUM A §A.4): below
  1.15 eV the argmax wanders across three columns — a factor 7 in density — as a
  function of an assumed slab thickness. **The two coldest rows cannot support a
  ridge-location claim at all.** At $T_e\ge1.6$ eV it is $j=3$ in every run and
  the values barely move (0.12166 → 0.12163 across a twentyfold slab range).
- **Open, and it gates the location:** the $r_1$ deficit (§A.8). Report
  $\partial(\text{ridge location})/\partial(r_1\text{ scaling})$ explicitly. **If
  a factor-3 $r_1$ change moves the ridge by more than one grid interval, the
  ridge location must be withdrawn.** The *mechanism* (F.3) survives either way.

### What Claim F still needs

1. The $r_1$-scaling sensitivity of the ridge location (Tier 2, and it decides
   whether F.4 survives).
2. Re-run `verify_ridge_mechanism.py` with the $M>900$ mask, or state that its
   $j\ge4$ statistics are unmasked.
3. **Before claiming novelty**, read Fujimoto (2004) Ch. 4 and the Fujimoto
   JPSJ *Kinetics of Ionization–Recombination* series I–IV (1979–1985) directly.
   The logistic form is a **one-line rewriting** of a published formula and is
   the same object as the Michaelis–Menten saturation fraction whose elasticity
   bound is textbook. Write *"not previously stated in this form in the CR
   literature"*, **not** *"genuinely new"*.

---

## CLAIM G — The scope of the result is a measured boundary, not a hedge

*Not in the skeleton. It should be, because it converts the thesis's biggest
apparent weakness into evidence.*

> The restriction to $T_e\ge2$ eV is not a disclaimer. It is the measured
> boundary beyond which the one physical process most likely to invalidate the
> result — Lyman-series radiation trapping — stops mattering.

**State: ▶️ Run** (blocked from ✅ only by the escape-factor geometry question).

| run | all `window_ok` | $T_e\ge2$ eV | $T_e\ge2$ & $n_e\ge10^{14}$ |
|---|---|---|---|
| untrapped (canonical) | 202/680, worst 0.3868 | 45/448, worst **0.1748** | 0/108, worst 0.0717 |
| trapped $D=1$ cm | 176/680, worst 0.2925 | **45/448, worst 0.1748** | 0/108, worst 0.0712 |
| trapped $D=5$ cm | 170/680, worst 0.2550 | **45/448, worst 0.1746** | 0/108, worst 0.0694 |
| trapped $D=20$ cm | 160/678, worst 0.2237 | **45/446, worst 0.1740** | 0/106, worst 0.0627 |

**Above 2 eV the count does not move at all and the worst case moves 0.5% across
a twentyfold range in slab thickness.** $\Theta_P({\rm Ly}\alpha)$ at 2.947 eV
runs 0.9999 → 0.973. Neutral-temperature sensitivity is not load-bearing
(Franck–Condon $T_{\rm at}=3$ eV gives 173/680 vs 170/680).

**Below 2 eV the headline falls by 2.5–3.3×:** 38.7% → **11.6–15.7%**. It stays
above 10%, so the *point* still breaks down, but the number is a factor 2.5–3.3
too large and its size is set by a slab thickness the 0-D model does not contain.

**The mechanism of the fall is not loss of sensitivity.** $|f_3-f_4|$ moves only
6% at $D=1$ cm. What collapses is the *displacement*: trapping lengthens the
$n=2$ lifetime → raises stepwise ionisation → drops the neutral fraction
($n(1s)$ at [0,3] falls $2.99\times10^{14}\to7.26\times10^{13}$) → shortens
$\tau_{\rm QSS}$ 8× → a ground state that turns over faster is less stale →
$|\ln x|$ roughly halves. Consistent with the two-axis attribution: density lives
in $\bar S$, temperature in $\Delta\ln u$, and **trapping acts on $\Delta\ln u$**.

**What Claim G still needs:** the escape-factor geometry decision
(`RESUME_HERE.md` §5.1) — the $T_e\ge2$ eV conclusion is safe either way, the
cold-corner numbers are not.

---

## CLAIM ¬ — The anti-claims, which must be stated explicitly

A thesis whose central quantity was once misnamed must state what it is *not*
claiming, in the body, not in a rebuttal.

| ¬1 | **This is not QSS breakdown.** The closure is exact to $10^{-8}$ on the plateau at the worst grid point. Demonstrating that is *part of the result*. |
| ¬2 | **$M$ does not predict where the diagnostic fails.** Greenland (2001) established the general form; this work quantifies it for one diagnostic. |
| ¬3 | **It is not detachment.** The ridge sits 5.2× below the detached band (1.9× if $j=4$ is used — `grid.py:73` is inconsistent with `verify_eps_gridmap.py` and must be retraced); at citable detached densities the bound never reaches 10%. Live in `chapter3.tex:1349` and `thesis_main.tex:336` and must be deleted from both. |
| ¬4 | **It is not a divertor result.** No molecules, no transport, optically thin where $\tau_{{\rm Ly}\alpha} = 114$ cm⁻¹. **Never quote 38.7% and "divertor" in one sentence.** At [0,4] the model's own equilibrium is **96.6% neutral** — conditions where literature attributes 60–70% of D$\alpha$ to molecular channels the model does not contain (`grep -riE "H2|molecul|MAR|dissociat" src/rates/` returns nothing). |
| ¬5 | **The magnitude is not a number, it is a rate.** 38.7% *per 4.81% step*. |
| ¬6 | **The ridge is not "the Balmer diagnostic's" — it belongs to the (3,4) pair.** |

---

# PART 2 — Chapter map

Seven chapters. For each: **the one question**, **the story beat**, **which
claims it establishes**, and **what it must NOT claim**.

> Write in the order **3 → 5 → 4 → 2 → 6 → 7 → 1**
> (`thesis_architecture.md` PART 5 — that ordering survives this revision).

---

### Chapter 1 — Reading the Light from a Divertor

- **Question:** What does a spectroscopist actually measure, and what does
  turning that measurement into a temperature assume?
- **Beat:** *Setup and complication.* Promise the reversal without spoiling it.
- **Establishes:** the premise of Claim E (that a table indexed on $(T_e,n_e)$
  is what practice uses) and the framing of the central question.
- **Contents:** the divertor and why it matters for ITER · what a spectrometer
  records and what it does not · the inversion chain
  $I_{H\alpha}/I_{H\beta}\to$ table $\to(T_e,n_e)$ · **the assumption hidden in
  the table is equilibrium ionisation balance, not excited-state steadiness** ·
  why the divertor is never in equilibrium · the question, stated plainly.
- **Citations that are not optional:** **Sawada & Fujimoto, Phys. Rev. E 49,
  5565 (1994)** — *"Validity range of the quasi-steady-state solution of coupled
  rate equations"*, the thesis's exact question 32 years earlier, and currently
  cited nowhere in the repo; **Greenland (2001)**; **Verhaegh et al., PPCF 61,
  125018 (2019)** (neutral fraction already treated as a free parameter).
  The open `\todo` on Sawada–Fujimoto is **the single largest unresolved
  publication risk in Chapter 1.**
- **Must NOT claim:** that the thesis tests whether excited states keep up (it
  *starts* there and refutes it) · any ITER divertor parameter range without a
  source — the current draft asserts $T_e = 1$–5 eV, $n_e = 10^{13}$–$10^{15}$
  with none, and **three mutually inconsistent "ITER divertor" boxes are live**
  in the code (`qss_analysis.py:362`, `plot_results.py:89`,
  `verify_eps_gridmap.py:58`; only the last carries a citation).

---

### Chapter 2 — The Atoms

- **Question:** What happens to a hydrogen atom in a plasma, and how well do we
  know the rates?
- **Beat:** *The obvious suspect.* Build the instrument that will test it.
- **Establishes:** **Claim A** (A.1, A.2, A.5, and the $\ell$-mixing argument).
- **Contents:** the 43-state space and why $\ell$ is resolved to $n=8$ · cross
  section → Maxwellian-averaged rate coefficient · CCC excitation, and why
  $\Delta n=0$ transitions were excluded on the data provider's instruction ·
  ionisation · radiative and three-body recombination · spontaneous emission ·
  **proton-impact $\ell$-mixing and the structural argument for it** ($2s\to1s$
  is E1-forbidden, $\Delta n=0$ electron impact excluded, so proton $\ell$-mixing
  carries **99.92%** of the 2S loss rate and $n=2$ is unphysical without it) ·
  the Anderson benchmark and the $n=5$ story · **the two corrections found by
  audit** ($\ell$-mixing $F(U_m)$, traced to Badnell 2021 Eq. 9, bounded at
  <0.85%; the eigenvalue filter, 19 of 400 points).
- **Must NOT claim:** NIST as the source of the $A$-values (they are exact
  hydrogenic, +0.058% above NIST) · that the model contains molecular channels ·
  that the bundled block has been convergence-tested (it has not).

---

### Chapter 3 — Two Clocks

- **Question:** If the rates are known, why is solving the problem hard — and
  what shortcut does everyone take?
- **Beat:** **The reversal.** The conceptual core; the chapter that earns the
  thesis.
- **Establishes:** **Claims B, C, D, F.1, F.2.**
- **Contents:** the master equation as a population balance (the ChemE
  reaction-network analogy is exact) · $\dot{\mathbf n} = L\mathbf n+\mathbf b$
  and why recombination is a **feed stream**, not a matrix element ·
  conservation as an assembly check · **the two clocks**, with the eigenvalue
  structure · **QSS as Bodenstein applied to the whole excited manifold** ·
  adiabatic elimination and the Schur complement · **the two reference states
  and why they are not the same** (`findings_09` §2) · the two-channel
  decomposition, the logistic form, $x_m = \ln(c_m/a_m)$, the $\tanh$ bound and
  the elasticity corollary · **the opacity chain, stated *here*, not only in
  Ch. 6** (ADDENDUM B §B.4): trapping ⇒ $A = A(n_g)$ ⇒ linearity in $n_g$ broken
  ⇒ Hill coefficient $\ne1$ ⇒ width $\ne1$ ⇒ bound becomes $\tanh(m\Delta/4)$
  with $m$ unknown. **The framework is two-channel *by construction*; a
  molecular channel would be a third and would destroy the functional form, not
  perturb it.**
- **Also state here:** the fixed-ion-reservoir assumption is **bounded** — total
  bound population per unit $n_i$ at the benchmark is $8.925\times10^{-4}$, so
  freezing $n_i$ costs at most **0.09%**. Quote that bound.
- **Must NOT claim:** "QSS breaks down" in any form · the detachment sentence
  (`chapter3.tex:1349` — **delete**) · the ADAS SCD96/ACD96 comparison promised
  at `chapter3.tex:1377-1387`, which does not exist · $M$ or $\tau_{\rm QSS}$
  without naming the operator (pre- or post-step).

---

### Chapter 4 — Does the Model Work?

- **Question:** Why should anyone believe a number this model produces?
- **Beat:** *Credibility, including the errors we found ourselves.*
- **Establishes:** **Claim A** (A.3, A.4, A.6–A.9) and the validation ladder of
  PART 4 below.
- **Contents:** detailed balance · coronal and Saha limits · timescale hierarchy
  (Gate E) · **the external benchmark against Fujimoto Table 4.1 and App. 4B** ·
  Anderson RMPS and the honest $n=5$ account · **§4.x Errors found and
  corrected** — the $\ell$-mixing $F(U_m)$ error, the eigenvalue filter, the
  misnamed metric · **Gate D reported as failing, with its diagnosis** ·
  **the two attacks that were survived and quantified** (`findings_10` §3.1
  closure, §3.2 the lower bound), each with the falsifier that did not occur.
- **Must NOT claim:** that Gate D passes, or omit it · that the closure test
  settles transport (it does not; it does not even close the electron balance) ·
  that the $\tanh$ check is validation.

---

### Chapter 5 — What the Model Says

- **Question:** How wrong is the table, where, and why there?
- **Beat:** *The result.* The reversal delivered and mapped.
- **Establishes:** **Claims E, F.3, F.4, G, ¬2.**
- **Contents, in this order:** (i) **the QSS closure is exact to $10^{-8}$ on
  the plateau — state this first, it frames everything**; (ii) the error map,
  headlined on **$d\varepsilon/d\ln T_e$** with the per-step value underneath at
  a stated step; (iii) the mechanism, using the **exact** form
  $\varepsilon = |e^{\bar S\,\Delta\ln u}-1|$ — **not** the linearised
  $|f_3-f_4|\cdot|\ln x|$ as the headline (`CH5_EVIDENCE.md` §2 and
  `chapter5_C5D.tex:141` both say so; `thesis_ready.md` A9 disagrees with them
  and A9 is the one at ▶️); (iv) **the two-axis attribution** — density lives in
  $\bar S$ (varies 7.11×), temperature is 76% in $\Delta\ln u$; (v) **the ridge,
  as a position effect**: the maximum sits where
  $\ln(u_{\rm CRE}/u_{\rm peak})$ changes sign, with the Griem-boundary
  verification (7.2 vs 6.7) as its mechanism; (vi) why the error is not averaged
  away over an ELM, with the honest note that the averaging is nearly a no-op on
  the counted set; (vii) **why timescale separation does not predict it**, with
  the sign quoted and not the magnitude.
- **Figures (none currently exist):** $\varepsilon$ map with the window mask ·
  **the $\varepsilon$-vs-$n_e$ line plot, non-negotiable** · $M$ vs
  $\varepsilon$ scatter · trapping sensitivity at the worst point. Do not reuse
  anything in `figures/` except the four `fig3_*`.
- **Must NOT claim:** 38.7% bare, or anywhere near the word "divertor" · a ridge
  location for $T_e<1.6$ eV · "at every temperature" (false at 3 of 49 rows) ·
  "roughly fixed density" without its resolution · the amplification maximum
  (1271×, unstable where $\varepsilon_{\rm step}\to0$) · that the linearisation
  understates one-sidedly (it does not — the endpoint form understates in only
  219 of 392 heat steps and **over**states at the benchmark, ratio 0.9459) ·
  the `[MECHANISM NOT ESTABLISHED]` bracket at `chapter5_C5D.tex:270` must be
  resolved or converted to an explicit "empirical" statement before an examiner
  sees it.

---

### Chapter 6 — What This Model Cannot Say

- **Question:** Where would you not trust this?
- **Beat:** *Bound the claim before an examiner does.*
- **Establishes:** the approximation register of PART 5 below; **Claim G**;
  **¬4**.
- **Contents:** optical thickness — **quantified, not hedged**: $\tau_{{\rm Ly}\alpha}$
  per cm of 114 / 15.0 / 0.32 / 0.036 / 0.0023 at 1.00–2.95 eV, escape factor
  $2.7\times10^{-5}$ over 5 cm at the worst point, independently reproducing
  §3.5.3's $3\times10^{-5}$ — followed by ADDENDUM A showing the $T_e\ge2$ eV
  result is invariant · **transport**, with the number that matters (6–10 µs
  transit against 233 ms; ≈26 µs renewal suffices to drop [0,4] below 10%) ·
  **molecules**, with the 96.6%-neutral figure and the literature's 60–70%
  D$\alpha$ attribution · $n_{\max} = 15$ with the terminal-shell excess
  **measured** ($r_1(15)$ high by 4.9–6.2×) · open system / fixed ion reservoir,
  bounded at 0.09% for the population but **not** for $\tau_{\rm QSS}$
  (factor 41 at the cold corner) · Maxwellian electrons · $T_i = T_e$ in
  $\ell$-mixing · the low-$T_e$ grid edge, and why 1 eV is where the model must
  stop (a hardcoded convention in nine files, not a data limit — CCC runs to
  ~960 eV; the *defensible* reason is the absence of molecular channels) · the
  escape-factor geometry question, stated openly.
- **Must NOT claim:** that any of these are small · that the divertor framing
  survives them.

---

### Chapter 7 — What It Means

- **Question:** What should a modeller or a diagnostician do differently?
- **Beat:** *Hand over something usable.*
- **Establishes:** the transferable criterion; the honest novelty statement.
- **Contents:** the criterion in a form applicable to any CR matrix — compute
  $a_p$, $c_p$, $\Delta$, and $\ln(u/u_{\rm peak})$; the bound
  $|d\ln R/d\ln b_1|<1$ tells you the worst possible response · where in
  operating space to distrust a $(T_e,n_e)$ inversion · what a time-dependent
  inversion would require · what would settle the open questions (the $r_1$
  deficit, opacity geometry, molecules, transport) · non-normality and
  Mori–Zwanzig as continuation, **explicitly cut from this thesis**.
- **Must NOT claim:** novelty for the logistic form beyond *"not previously
  stated in this form in the CR literature"* · that the work is a divertor
  result · that Mori–Zwanzig results are current (April, pre-correction,
  self-contradictory summary, and $\tau_K$ lives in the one block the robustness
  argument does not protect).

---

# PART 3 — The four-step chain, per major result

Every result must have all four: **physical picture → mathematics → prediction
(stated in advance, with a falsifier) → numerical test.**
**A result missing its prediction step was never falsifiable.**

| # | Result | Picture | Maths | **Prediction + falsifier** | Test | Verdict |
|---|---|---|---|---|---|---|
| R1 | Two-timescale structure | ✔ two clocks | ✔ eigen-decomposition | ✔ Fujimoto App. 4B predicts $10^{-7}$/$10^{-4}$ s at a grid point | ✔ 3.4e-8 / 1.77e-4 | **complete** |
| R2 | **QSS closure is exact** | ✔ Bodenstein | ✔ Schur complement; M-matrix positivity | ✔ if the closure were the culprit, the residual would be $O(\varepsilon)$; it is $10^{-8}$ | ✔ both errors along one trajectory | **complete — the model chain** |
| R3 | Partial-equilibrium plateau | ✔ stale reservoir | ✔ one linear solve | ✔ analytic PE must match LSODA; window deficit predicted 0.98406 | ✔ 6 digits; deficit 0.98527 | **complete** |
| R4 | Two-channel split | ✔ fed from above and below | ✔ Fujimoto 4.20 realised | ✔ superposition must be exact; a 3↔4 swap must be caught | ✔ $3.1\times10^{-14}$; swap caught | **complete** |
| R5 | $d\ln R/d\ln u = f_3-f_4$, logistic, $\tanh$ bound | ✔ | ✔ closed form | ⚠ the $\tanh$ gate is **not** severe — it passes on corrupted input | ✔ but via superposition + reduced-vs-full $R$ | **complete, with the gate relabelled** |
| R6 | **The magnitude ($\varepsilon$ map, "38.7%")** | ✔ | ✔ exact solves | **✘ NO PREDICTION WAS EVER STATED** | ✔ reproduces bit-for-bit | **INCOMPLETE — see below** |
| R7 | Ridge mechanism (Griem boundary) | ✔ boundary level descends | ✔ $n^{-17/2}$ | ✔✔ **6.7 predicted before measurement**; refuters named | ✔ 7.2 measured; both refuters absent | **exemplary — the model result** |
| R8 | Ridge is a position effect | ✔ operating point sweeps | ✔ $\Delta$ flat, $\ln(u/u_{\rm peak})$ sign change | ✔ predicted $f_3-f_4\to0$ at LTE — **FALSIFIED, productively** | ✔ $\Delta$ flat 0.98→1.94 | **exemplary — a falsified prediction that improved the claim** |
| R9 | $M$ does not predict failure | ✔ | ✔ partial correlation | ✔ if $M$ predicted, control for $(T_e,n_e)$ would leave the correlation | ✔ flips sign; Arrhenius matches | **complete** |
| R10 | ELM persistence / the bound | ✔ plateau then decay | ✔ $\varepsilon_{\rm plateau}\frac{\tau_{\rm QSS}}{\tau_d}(1-e^{-\tau_d/\tau_{\rm QSS}})$ | ⚠ predicted "conservative"; **W3 refuted that**, then §3.2 showed near-exactness | ✔ true/lower 0.9999–1.0066 | **complete but the claim changed twice — state the history** |
| R11 | Trapping invariance above 2 eV | ✔ | ✔ self-consistent $\Theta_P$ | ✔ prior claim: trapping makes breakdown *worse* — **FALSIFIED** | ✔ 45/448 unchanged over 20× slab | **complete** |
| R12 | Open/closed boundary robustness | ✔ | ✔ exactly conserving closure | ✔ a factor $\sim\tau_{\rm QSS}/\tau_d\approx2300$ would refute A11 | ✔ factor 15; 202→200 | **complete** |
| R13 | Observable robustness | ✔ rigid-body $\ell$ | ✔ A-weighted sums | ✔ the 4F dipole-dark lever should break it | ✔ $f(4S)=0.0608$ vs $f(4F)=0.0606$ | **complete** |

## The one broken chain, and it is the headline

> **R6 — the magnitude — is the only major result with no prediction step, and
> it is the number planned for the abstract.**

Nothing was written down in advance saying how large the error *should* be, or
what size would have refuted the picture. The consequence is exactly what the
verification found: the quantity turned out to be **linear in an arbitrary grid
step**, so "38.7%" is not a prediction of anything — it is a measurement of how
far a 4.81% temperature step moves the reservoir.

**The repair is architectural, not computational.** Replace the headline with
the quantity that *does* have a prediction:

$$\text{headline} = \frac{d\varepsilon}{d\ln T_e}\Big|_{\rm ridge}, \qquad
\text{predicted} = |f_3-f_4|\cdot\left|\frac{d\ln n_g}{d\ln T_e}\right|_{\rm CRE}$$

Both factors are independently measurable ($|f_3-f_4|$ from the split; the CRE
gain from the ionisation balance, tabulated in `findings_09` §4.1 as 12.7 / 7.8
/ 4.6 / 3.1 across the temperature decades), so the product is a **prediction**
and the map is its **test**. That converts the thesis's weakest chain into one
of its strongest, at the cost of a derivative and a table.

**Two secondary chains that also need their prediction written down:**

- The **temperature dependence of $\varepsilon$**. Currently reported as an
  observation ("height falls with $T_e$"). It is 85% the CRE gain
  $d\ln n_g/d\ln T_e$, which is predictable. Predict it, then show the residual
  is the flat $|f_3-f_4|$.
- The **10% threshold count**. 348/202/61 at 5/10/20% is a sensitivity, not a
  prediction. Either state in advance what count would falsify the picture, or
  stop leading with the count and lead with the worst case, which *is* robust.

---

# PART 4 — The validation ladder

**The thesis must not mix "the model is correct" with "the model predicts
something interesting."** These are Chapter 4 and Chapter 5 respectively, and
they must be physically separated in the document.

**VALIDATION** = the model reproduces things already known to be true.
**DISCOVERY** = the model says something not previously known.
A rung that is really a discovery masquerading as a check (the $\tanh$ gate) is
worse than no rung.

## The ten rungs, ordered from cheapest/most fundamental to most demanding

| # | Rung | Status | Establishing artifact | Measured number |
|---|---|---|---|---|
| 1 | **Atomic-data provenance** | ✅ **PASS — strongest rung** | `src/rates/*.py` headers, `data/**/*meta.csv`, `data/raw/adas/PROVENANCE.md` | Source tallies exact: $K_{\rm exc}$ 546 CCC + 36 CCC_n9 + 36 CCC_n10 + 201 V&S = **819**; $K_{\rm ion}$ 36 + 1 + 6; RR 43× Johnson1972 |
| 2 | **Units / dimensions** | ⚠ **PARTIAL — no executable gate anywhere** | prose only; `verify_fujimoto_table41.py:281` | `S_grid` label **fixed** by `ffe1768` (B8 is stale); Rydberg **8 values / 40 literals**; `CHI_H` in four **Saha–Boltzmann exponents**, 0.0534% high; no `pint`, no constants module |
| 3 | **Detailed balance** | ✅ **PASS at two levels** | Gate A `validate_gates.py:90-122`; **raw-data check `src/parsers/qc_ccc.py:87-168`** | Gate A max err **<0.01%** (819×3 pairs, but true *by construction*); raw CCC ratio **mean 0.9995, std 0.0032, 98.7% within 1%** |
| 4 | **$A$-sum / oscillator strength** | ⚠ **PASS on the $f$ round-trip; WEAK against NIST** | `verify_lyman_trapping.py:169-215`; `pre_assembly_check.py:316-337` | $f$ from stored $A$ = **0.4162 / 0.0791 / 0.0290 / 0.0139**; $\sigma_0$ vs independent module **0.059%** (tol 1%, raises) |
| 5 | **Particle conservation** | ⚠ **PASS, sampled and unpersisted** | `assemble_cr_matrix.py:269-282` "Check D" | Max relative column-sum error **2.59×10⁻¹¹** — over **20 of 400** points, and recorded in no `validation/` file |
| 6 | **State-space convergence / truncation** | ✘ **NEVER DONE** | `verify_bundling_psm20.py` exists; `outputs/bundling/` does not | **No number.** C6 ($n_{\max}$ = 12/20) never run and explicitly deferred |
| 7 | **Solver tolerance / conditioning** | ⚠ **PARTIAL; one traceability FAIL** | `verify_fujimoto_table41.py:329-341`; `verify_plateau_slowmode.py:316-330` | cond($L_{EE}$) **7.2e3–1.6e5** at 4 point-variants; rtol 1e-9 vs 1e-12 → **1.20e-8**; `expm` vs `expm_multiply` → **1.13e-10** |
| 8 | **Limiting cases** | ✅ **PASS** | Gates B, C; `verify_divertor_map.py`; `verify_lyman_trapping.py:492-507` | Gate B **50/50**, worst convergence 0.0034; Gate C **50/50** monotone & below-Saha; untrapped rebuild **0.000000e+00** |
| 9 | **Comparison with published coefficients** | ✘ **MIXED — Gate D FAILS 0/400** | `validation/gate_summary.txt`; `validation/fujimoto_table41/` | $\eta\in[\mathbf{5.73},\ \mathbf{8346}]$, **0/400 within a factor 2**; Fujimoto 4.1(b) $r_0$ ratio **0.88–1.01** at $p\ge2$; $r_1$ deficit **8.3×** at $p=3$ open |
| 10 | **Independent reproduction** | ⚠ **PASS, with one ✅ that must be retracted** | backlog §2026-09-09; `make_ch5_figures.py:378-458` | A1/A8/A11/A12 **bit-for-bit / byte-for-byte**; A12 19/400 differ, 19/19 with $\tau_{\rm relax}^{\rm new} = \tau_{\rm QSS}^{\rm old}$ exactly |

### Ladder verdict

**Genuinely passed and severe:** rungs 1, 3, 8, and the reproduction half of 10.
**Cheap to close:** 2 (a constants module and one gate), 5 (widen Check D to all
400 and stamp the residual), 7 (produce $\kappa(L_{FF})$ or delete the sentence).
**Unclimbed:** 6.
**Honest failures that must be reported, never repaired:** 9.

### Three items the architecture must treat as load-bearing

1. **Rung 6 is a genuine hole with a defence question already written against
   it** (`master_plan_v2.md:316`). Ch. 6 must either carry the run or carry the
   scope limitation explicitly.
2. **`chapter3.tex:719-721`'s $\kappa(L_{FF})$ range has no producing script.**
   It is written LaTeX asserting a grid-wide number nothing computed.
3. **A5's ✅ is not supported by a run.** The May report was never re-run and
   the project's own derivation file says so. Demote or re-run.

### What was searched for and does not exist

- Rung 6, any output: `outputs/bundling/`, all 37 `validation/` entries, a
  repo-wide `find -name "*bundling*"`, and the script's `git log`. Nothing but
  the script.
- Rung 7, $\kappa$ of the full $L$: `np.linalg.cond` has **two hits**, neither on
  $L$.
- Rung 4, sum rules: **no Thomas–Reiche–Kuhn check**, no lifetime assertion, no
  code path reading `wiese_fuhr.pdf`.
- Rung 9, PEC: **no `adf15`/PEC file under `data/`** — so the H$\alpha$/H$\beta$
  = 4.1 vs "ADAS PEC 2.8–4.5" comparison in `report_adas.md` cannot be cited.
- Rung 2, dimensional gate: no `pint`/`unyt`; `astropy` declared in
  `requirements.in:26`, zero hits in `src/`.

## Where the ladder is currently mixed with discovery — separate these

| Currently presented as validation | Actually |
|---|---|
| **Gate A, "detailed balance to <0.01%"** | **True by construction** — `K_deexc` is *derived from* `K_exc`, so it can only catch a wiring error. The physics check is `qc_ccc.py` on Bray's **raw** cross sections (mean 0.9995, std 0.0032), and it is currently invisible. **Swap which one Chapter 4 leads with.** |
| **Gate C, "Saha limit"** | Its `passed` flag is `monotone and below_saha` **only**; the `approaching` column is computed and then **excluded**. Measured `approach_frac` = **0.0123–0.0155** — populations reach ~1.5% of Saha at $n_e = 10^{15}$, so **LTE is never actually approached**. That is correct physics for an optically thin open system, but the gate does not test the limit it is named for. Say what it tests. |
| **`radiative_rates.py:222-238` checking $A$ against "NIST" anchors** | The anchors are the **Hoang-Binh hydrogenic values**, so the check structurally cannot detect the +0.058% offset from the true Wiese–Fuhr set. It validates self-consistency and is labelled as an external check. |
| "$\tanh$ bound honoured at all 248 points" | **A theorem given $a,c\ge0$.** Cannot fail except on an arithmetic bug; passes on corrupted input. Relabel as a wiring check (the docstring already does; the printed line does not) |
| "superposition holds to $3\times10^{-14}$" | **Exact by construction** — validates implementation, not physics. Say so, then note it *is* severe because it catches a 3↔4 swap |
| "the identity $\int(f_3-f_4)d\ln x = \ln[R/R_0]$ holds to $3\times10^{-16}$" | Same category |
| Isolation gate `if iso_gr.min() < 10.0` against a measured minimum of 24.33 | **A gate 2.4× slacker than the quantity it guards, with no source for 10.0** |
| `preflight.py` printing "All checks passed" | **Two of six scripts are never checked** — `SCRIPTS` names files that do not exist on disk, and the miss is downgraded to a non-blocking `NOTE`. Per `CLAUDE.md` rule 2 this must raise |
| `audit_writers.py` as single-writer evidence | **Four live CSV writers are invisible to it**, including the producer of `plateau_gridmap.csv` — the source of A8/A9/A11. **Do not cite this script** |

## Gates whose *design* is sound and should be advertised

- The Ly-$\alpha$ $\sigma_0$ cross-check that **failed at 1156% on first run**
  and caught a $4\pi$ CGS/SI error. A gate that has actually fired is worth more
  in a defence than ten that never have.
- The untrapped-rebuild identity ($\max|{\rm rebuilt}-{\rm canonical}| = 0.000000$).
- **`verify_fujimoto_table41.py`'s deliberately-broken variant**: with proton
  $\ell$-mixing removed it returns $r_0(2) = 104.5$ and the script flags it
  itself as *"unphysical, a level cannot exceed Saha equilibrium"*. A check that
  fires on a known-bad input is the definition of severe.
- **`make_ch5_figures.py:378-458`** re-computes from the matrix and compares
  against the CSV at rtol $10^{-9}$ **on every run**, raising on mismatch. It is
  the best-engineered script in the repo — reads the step size from the CSV
  header rather than re-declaring it, derives indices in log space, raises rather
  than asserts. **Use it as the template for everything else.**
- **The semigroup propagator check, withdrawn by the author as tautological**
  (it returns 0.000e+00 by construction). Withdrawing your own check is
  evidence of judgement; report it.
- **`verify_timescales.py:207-213`**, a recorded negative result correcting an
  earlier version that misread column sums as conservation.
- All residual gates **raise**, none warn; no `try/except` swallows anything in
  the four numeric verification scripts; zero occurrences of `np.nan_to_num`,
  tolerance-widening or outlier-dropping in the nine audited files.
- **`assert` is not a guard.** `make_ch3_figures.py:160` and
  `verify_plateau_bridge.py:519` protect the benchmark index with `assert`,
  which `python -O` deletes. Zero impact today; adopt `raise` everywhere.

---

# PART 5 — Every approximation, four ways

For each: **what is approximated · why it might be justified · where it is
expected to fail · how it was tested.**

---

### 1. The quasi-steady-state closure — $\dot{\mathbf n}_E = 0$, **not** $\dot n_g = 0$

- **What:** the excited manifold is algebraically slaved to the reservoir:
  $\mathbf n_E = -L_{EE}^{-1}(L_{Eg}n_g + \mathbf S_E)$.
- **Why justified:** $M = \tau_{\rm QSS}/\tau_{\rm relax}\gg1$ everywhere;
  $L_{EE}$ Metzler and Hurwitz with a spectral gap of $3.7\times10^7$ between
  $\lambda_0$ and $\lambda_1$.
- **Where expected to fail:** where an intermediate mode appears between the two
  scales, or where $M$ falls toward unity ($M_{\min} = 86.8$ at [49,7]). Also
  under strong trapping, where $\tau_{\rm relax}$ rises 27× at the cold corner.
- **How tested:** measured directly along the trajectory — closure residual
  $8.66\times10^{-6}$ at the benchmark and $6.73\times10^{-9}$ at the worst
  point, against a reported error of $10^{-2}$–$10^{-1}$.
  **It does not fail anywhere tested. That is the thesis.**
- **The distinction is the thesis:** $\dot n_g = 0$ is a *separate and much
  stronger* assumption, is what a $(T_e,n_e)$ table imposes, and is the one that
  fails.

### 2. Equilibrium ionisation balance (the CRE assumption inside the lookup table)

- **What:** that $n_{1s}/(n_e n_{\rm ion})$ is a function of $(T_e,n_e)$ alone.
- **Why it might be justified:** in a steady plasma it is exact; it is what
  every ADAS-style inversion table is built on.
- **Where expected to fail:** whenever the operator changes faster than
  $\tau_{\rm QSS}$ — ELMs, detachment transients, any $\mu$s-scale event.
- **How tested:** this is **the object of the thesis**, not a side assumption.
  Measured across 680 pairs; the cost is the whole result.

### 3. Optically thin (no radiation trapping)

- **What:** photons escape; $A$ coefficients are the vacuum values.
- **Why:** $\Theta_P({\rm Ly}\alpha) = 0.9999$–0.973 at $T_e = 2.947$ eV over
  1–20 cm.
- **Where expected to fail:** $T_e\lesssim1.6$ eV, where $\tau_{{\rm Ly}\alpha}$
  reaches 114 per cm and the escape factor $2.7\times10^{-5}$ over 5 cm — the
  effective Lyman decay rates in `L_grid.npy` are wrong by 4–5 orders of
  magnitude there.
- **How tested:** ADDENDUM A, self-consistently, all 14 Lyman channels, swept
  over slab thickness. **Above 2 eV: count unchanged, worst case moves 0.5%
  across a twentyfold slab range. Below 2 eV: the headline falls 2.5–3.3×.**
  Residual risk: the escape-factor **geometry** (sphere vs slab, ≈2.1×) is
  unresolved and affects only the cold-corner numbers.

### 4. Open system — the ion is a fixed reservoir, no 44th state

- **What:** $\mathbf S_E$ is an externally imposed source; ion density does not
  respond.
- **Why:** total bound population per unit $n_i$ at the benchmark is
  $8.925\times10^{-4}$, so freezing $n_i$ costs at most **0.09%** on populations.
- **Where expected to fail:** on $\tau_{\rm QSS}$, not on populations — the open
  boundary is what makes $\tau_{\rm QSS}$ reach 67 s in the cold corner. It also
  means the two channels **can never merge** (no Saha tie), so the grid never
  reaches LTE — a structural, not numerical, consequence.
- **How tested:** exact particle-conserving 44-state closure. Grid-wide factor
  min 1.00, median 1.00, **max 41.4 — and unity everywhere $\tau_{\rm QSS}$
  approaches $\tau_d$.** ELM recount 202→200, worst 0.38682→0.38563.
  Falsifier named ($\approx2300$) and absent. **Caveat: it does not close the
  electron balance and adds no transport.**

### 5. Atomic hydrogen only — no molecular channels

- **What:** no D$_2$, D$_2^+$, MAR, dissociative excitation.
- **Why it might be justified:** above ~2–3 eV molecular densities are small.
- **Where expected to fail:** exactly where the headline lives — at [0,4] the
  model's own equilibrium is **96.6% neutral**, and the literature attributes
  60–70% of D$\alpha$ and 10–20% of D$\gamma$ to molecular channels at
  detachment onset.
- **How tested:** **NOT TESTED.** `grep -riE "H2|molecul|MAR|dissociat" src/rates/`
  returns nothing. Cannot be settled inside this model.
  **Structural consequence (ADDENDUM B §B.4): a molecular channel is a *third*
  channel — it does not perturb the two-channel split, it destroys its
  functional form.** Chapter 6 must say the framework is two-channel *by
  construction*.

### 6. $n_{\max} = 15$, $n\ge9$ bundled, no $\ell$-mixing in the bundle

- **What:** the state space is truncated and partly bundled.
- **Why:** high-$n$ states hold negligible population and are collisionally
  dominated.
- **Where expected to fail:** the terminal shell — $r_1(15)$ measured **4.9–6.2×
  high** against published values, while $n=10$ agrees to 4–20%. Also anywhere
  the observable is `max_p`-dominated: the old A7 metric attained its max at
  $n=15$ at 346 of 400 points.
- **How tested:** truncation moves the observable **0.04% per top shell**;
  recomputing A7 over resolved states only moved the grid mean 0.517→0.515.
  **`verify_bundling_psm20.py` has never been run** — the bundling confound
  itself is untested.

### 7. $\ell$-resolved to $n=8$; proton-impact $\ell$-mixing with $T_i = T_e$

- **What:** $\ell$ is resolved where it matters; ion temperature equals electron
  temperature in the mixing rate.
- **Why:** proton $\ell$-mixing carries **99.92%** of the 2S loss rate;
  $2s\to1s$ is E1-forbidden and $\Delta n=0$ electron impact is excluded, so
  $n=2$ is unphysical without it.
- **Where expected to fail:** at low density where mixing weakens and radiative
  decay competes — measured as the 0.3% departure from statistical at
  $n_e = 10^{12}$; and wherever $T_i\ne T_e$ (divertor conditions, routinely).
- **How tested:** the $\ell$-distribution is measured, not assumed — 4F fraction
  0.4361–0.4375 vs statistical 0.4375; $\ell$-populations move as a rigid body;
  $\Delta$ varies 0.07% across five $\ell$-weightings. The $F(U_m)$ error was
  found, traced to Badnell 2021 Eq. 9, and bounded at **<0.85%** on
  $\tau_{\rm relax}$.

### 8. Maxwellian electrons

- **What:** rate coefficients are Maxwellian averages of cross sections.
- **Why:** standard; the collisional timescale is short compared with the
  transients considered.
- **Where expected to fail:** in a detached divertor with strong gradients and
  non-thermal tails, which is precisely the regime invoked.
- **How tested:** **NOT TESTED.** State as a scope statement in §1.5 and
  Chapter 6.

### 9. Zero-dimensional, uniform plasma, no transport

- **What:** a single fluid element with no spatial coupling and no recycling.
- **Why it might be justified:** for the *fast* excited-state physics, which is
  local and complete in nanoseconds.
- **Where expected to fail:** for the *slow* ground-state physics, which is the
  whole result. A 1–3 eV D atom crosses 10 cm in **6–10 µs** against the model's
  **233 ms**; an effective renewal time of ≈26 µs already drops the worst point
  below the 10% threshold. **The ridge location is proportional to the assumed
  neutral density, and in a divertor that density is set by recycling.**
- **How tested:** **NOT TESTED and untestable in 0-D.** Quantified as a
  conditional: *if* the ground-state neutral density is frozen for the duration
  of the event, the error is as computed. Settling it needs a two-region or
  SOLPS-coupled calculation in which $n(1s)$ is set by recycling influx.
  **This, not closure, is the real threat to the cold-corner magnitude.**

### 10. The step idealisation — instantaneous $T_e$ change at fixed $n_e$

- **What:** the operator jumps; the reservoir does not.
- **Why justified:** any finite ramp with $\tau_{\rm relax}\ll t_{\rm ramp}\ll
  \tau_{\rm QSS}$ reaches the same plateau. **The step idealisation is safe —
  but not for the reason `ramp_vs_step.csv` suggests.** That file's
  ${\rm De} = \tau_{\rm relax}/\tau_d\approx10^{-5}$ is the suppression of the
  *fast* step error; the plateau is governed by
  ${\rm De}_{\rm slow} = \tau_{\rm QSS}/\tau_d = 2332$ at [0,4].
  **Never quote `ramp_vs_step.csv` for or against the magnitude.**
- **Where expected to fail:** for real ELMs, which raise $n_e$ as well as $T_e$.
  Joint steps give comparable $\varepsilon_{\rm plateau}$ but a time-averaged
  bound that can fall **4×**.
- **How tested:** partially. **The joint $(T_e,n_e)$ step map has not been run.**

### 11. Linearisation of $\varepsilon$ in $|\ln x|$

- **What:** $\varepsilon_{\rm plateau}\approx|f_3-f_4|\cdot|\ln x|$.
- **Why:** it is the first term of the exact
  $\varepsilon = |e^{\bar S\Delta\ln u}-1|$.
- **Where expected to fail:** outside $|\ln x|\in[0.124, 0.682]$, the measured
  one-interval window. At $|\ln x|\approx5.7$ the linear estimate misses by
  **46×**. The four-interval numbers (58.5%, 81.1%, 190.4%) are exact solves but
  sit **3.8× beyond** the validated ceiling.
- **How tested:** ratio measured/predicted over 680 points — median 1.031
  (heating), 0.985 (cooling), range 0.766–1.389.
  **Read the ratio, not the correlation** (+0.989 is near-tautological).
  ⚠ **The "linearisation understates" claim is false as one-sided:** for the
  endpoint form actually computed it understates in only 219 of 392 heat steps
  and **over**states at the benchmark (0.9459). **Name which linearisation is
  meant.** Use the exact form as the headline.

### 12. $\Delta n = 0$ electron-impact transitions excluded

- **What:** a whole class of collisional transitions is omitted.
- **Why:** on the data provider's explicit instruction (the CCC data are not
  reliable there).
- **Where expected to fail:** it is why proton $\ell$-mixing must carry the 2S
  loss; if the excluded rates were comparable, $n=2$ would be wrong.
- **How tested:** structurally, via the 99.92% argument. Not tested numerically.
  State it as a stated choice with its source, per `writing_specification.md` N5.

### 13. The ELM time-average model (plateau, then exponential decay on $\tau_{\rm QSS}$)

- **What:** $\bar\varepsilon = \varepsilon_{\rm plateau}\frac{\tau_{\rm QSS}}{\tau_d}(1-e^{-\tau_d/\tau_{\rm QSS}})$.
- **Why:** the spectrum has no intermediate mode, so nothing can decay faster
  than $\tau_{\rm QSS}$.
- **Where expected to fail:** it did — for the *other* observable
  $\varepsilon_{\rm res}$, whose 1/e time is 2.58 s against
  $\tau_{\rm QSS} = 67.2$ s (`derivation_07:301-304`). And W3 showed it
  overstates the true average at 330 of 680 pairs for cooling steps.
- **How tested:** direct 43-state eigen-propagation at seven points including
  the worst: true/lower = **0.9999–1.0066**. Say both things — it is not a
  guaranteed bound, and it is empirically accurate to 0.7% for **this**
  observable.

### 14. Escape-factor geometry (only inside ADDENDUM A)

- **What:** ADAS214 eq. 3.14.14 applied as if it were a slab.
- **Why it might be justified:** it is the standard ADAS expression.
- **Where expected to fail:** it *is* the isotropic/sphere-centre case (ADAS
  g1); a true slab (g2) is **≈2.1× smaller** at large $\tau_c$.
- **How tested:** identified by direct reading of the primary source. **Not yet
  resolved.** Affects only the $T_e<2$ eV numbers; the $T_e\ge2$ eV conclusion
  is safe because $\Theta_P\approx1$ there. Everything else in the module
  verified correct against the primary source.

### 15. Non-relativistic, infinite-nuclear-mass hydrogenic $A$-values

- **What:** the stored Lyman $A$'s and $f$'s are exact hydrogenic
  ($f = 0.4162 = 2^{13}/3^9$).
- **Why:** self-consistent, analytically exact for the idealised atom.
- **Where expected to fail:** they sit **+0.058% above** the NIST/Wiese–Fuhr
  recommended values. Immaterial numerically.
- **How tested:** reproduced to 1–4 parts in $10^5$ by inverting the repo's own
  $A$'s. **Consequence for the text: do not cite NIST as their source.**

---

# PART 6 — What must be settled before the rewrite starts

Ordered by cost of getting it wrong, not by effort.

1. **Notation for the two references.** $\varepsilon^{\rm QSS}$ vs
   $\varepsilon^{\rm CRE}$, or better names. Patch `chapter4.tex`
   §`sec:qss_ratio` first; everything downstream inherits from that one false
   sentence.
2. **The pre/post-step operator convention**, applied to every $\tau$ and $M$.
3. **The observable**, once: $n_3/n_4$ or A-weighted $H\alpha/H\beta$.
4. **The headline quantity of Chapter 5**: $d\varepsilon/d\ln T_e$, with the
   per-step value as illustration. This is the PART 3 repair and it is the only
   architectural change that turns a broken chain into a complete one.
5. **The `eps_step` name collision** — ten definitions, two of them mutually
   inconsistent hardcoded fits with a 9.6% spread at $T_e = 3$ eV.
6. **The benchmark point's name.** Grid [23,5] is *"the point nearest
   $T_e = 3$ eV, $n_e = 10^{14}$"* — an illustrative anchor with no citation.
   **Never call it an ITER reference.** (`ffe1768` already retired that naming;
   `thesis_ready.md` A1 still uses it.)
7. **The title.** The registered title names an approximation this work shows to
   be *exact*. It must change, and that is a supervisor conversation.
   `thesis_architecture.md` PART 2's candidates survive this revision; prefer
   one that names **ionisation balance**, not QSS.
8. **Four deletions**, textual and non-negotiable: the detachment sentence
   (`chapter3.tex:1349`, `thesis_main.tex:336`), the ADAS SCD96/ACD96 promise
   (`chapter3.tex:1377-1387`), the `[MECHANISM NOT ESTABLISHED]` bracket
   (`chapter5_C5D.tex:270`), and the **unsupported $\kappa(L_{FF})$ range**
   (`chapter3.tex:719-721`) — unless a script is written to produce it.
9. **Three register repairs**, in `thesis_ready.md`: **B8 is stale** (the
   `S_grid` label was fixed by `ffe1768`); **A5 must be demoted** from ✅ (the
   May run was never re-run and `derivation_04:375-381` says so); and the
   "**at least six**" `eps_step` definitions in B6 is **ten**.
10. **Rename one of the two "Gate D"s** — the ADAS external gate and the
   assembler's conservation Check D currently share a name in a repo where
   two-names-one-quantity has already caused three retractions.
11. **Correct `CLAUDE.md`'s known-issue table**: it names `qss_analysis.py` for
   `eigs[eigs < -1.0]`, which now reads `eigs < 0.0`. The literal is live at
   `src/rates/solve_cr.py:269` and `src/rates/check_mz.py:10`, imported by
   nothing. **Fix the table, not the code** — per the standing rule, and because
   the filter may be load-bearing elsewhere.

**Final grep before submission**, against the PDF: `25 ns` · `M = 611` ·
`−46%` · `1.18 µs` · `ne^-1.00` · `38.7%` · `detachment` · `ITER reference` ·
`QSS breakdown`.
