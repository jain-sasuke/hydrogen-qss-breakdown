# Round 4 review — adjudication

Written 16 September 2026. Four read-only audits (math, CR physics, evidence,
feasibility) against the current tree. **Nothing in the repository was modified.**

The reviewer's build is **commit `65b05da`, 10 Sep 2026 23:50, 175 pages**.
Fourteen commits land after it; the current build measures **193 pages**. Every
"already closed" below is anchored to a git diff, not to a status file.

**Ground rule applied throughout:** the reviewer is not an authority. Each claim
was re-derived or re-measured. Two of his findings are wrong, three are already
fixed, and one of them — item 8 — turns out to break a claim the thesis makes.

---

## 1. Verdict on all fifteen items

| # | His claim | Verdict | Action |
|---|---|---|---|
| 1 | Numerical implementation PASS | **Agreed** | none |
| 2 | Shell-ratio vs line-ratio closed | **Agreed** | none |
| 3 | ADAS corrected; factor-two criterion weak | **PARTLY WRONG** — the distribution *is* reported at ch4:1238. But his substantive point lands harder: the box count is an interpolation artifact | fix the reader, then requote |
| 4 | `u = ACD/SCD` untested; highest priority | **PARTLY** — identity valid and exact; his *level* test is information-free; the *derivative* test is real and appears to pass at 1–2 % | run G comparison |
| 5 | n_max convergence of u, τ_slow, ε_plateau | **PARTLY LIVE** — n_max 18/20 **impossible**; Table 4.4 row 1707 has no artifact and is scope-mislabelled | rescope the row |
| 6 | Bundling proves less than the text implies | **CORRECT, and worse** — margin is 100, not 1538 | requote the margin |
| 7 | Terminal-shell mechanism physically wrong | **CORRECT on the error**, over-agnostic on the replacement | rewrite two sentences |
| 8 | Fujimoto not like-for-like; ℓ-closure unproven | **CORRECT — and the test now refutes the thesis's explanation** | **withdraw the explanation** |
| 9 | Fujimoto status self-contradictory | **PARTLY** — 2 of 3 legs closed in `71691bf`; title/TOC collision survives | retitle §6.7 |
| 10 | §4.13 obsolete | **F DESERVED** — 3 of 5 byte-for-byte unchanged | rebuild the table |
| 11 | "plateau lower bound" must not be a theorem | **ALREADY CLOSED** — he read pre-16-Sep text | none |
| 12 | Inversion work is verification, not validation | **FAIR, and the thesis already says so** | none |
| 13 | Provenance not submission-grade | **CORRECT, and worse than he knew** | see §4 |
| 14 | ACD benchmark is 9.3 % low, not 6 % | **CORRECT — at two sites** | fix both |
| 15 | Run Test A and Test B | A: cheap, and passing. B: **not possible as specified** | see §5 |

---

## 2. The finding that changes a result — item 8

The thesis explains the Fujimoto factor-8 `r_1` deficit as an ℓ-closure
difference, and demotes the row to "tests the ℓ closure rather than this model"
(ch4:1717, ch4:1148-1153, ch6:1046-1049).

**That explanation is now refuted by its own test.** Building the statistical-ℓ
bundle exactly as Fujimoto closes it, `L_bundle = P L R` with
`P[k,i] = 1` for i in shell k and `R[i,k] = g_i/g_k` (`P@R = I` exactly):

| variant | p=2 | p=3 | p=4 | p=5 | p=10 | p=15 |
|---|---|---|---|---|---|---|
| Fujimoto Table 4.1(b) | 1.790e-4 | 5.720e-5 | 1.660e-5 | 5.080e-6 | 9.610e-8 | 9.340e-9 |
| PRODUCTION (ℓ-resolved) | 1.659e-5 | 6.897e-6 | 3.778e-6 | 1.624e-6 | 9.226e-8 | 4.563e-8 |
| NO-PROTON-ℓ-MIXING | 1.959e-3 | 2.310e-5 | 5.495e-6 | 2.011e-6 | 1.062e-7 | 5.291e-8 |
| **STATISTICAL-ℓ BUNDLE** | **1.515e-5** | **6.926e-6** | **3.798e-6** | **1.630e-6** | **9.253e-8** | **4.576e-8** |

**Imposing Fujimoto's own closure moves `r_1(3)` by 0.4 % and `r_1(2)` by 8.7 %.
The factor 8.3 at p=3 becomes 8.2. At p=2 it gets slightly worse, 10.8 → 11.8.**

Two structural facts underlie this:

1. `max |P B R| = 8.494e-07` against a matrix scale of `1.003e+11`, where B is
   the explicit ℓ-mixing block. **The statistical-ℓ bundle annihilates the
   ℓ-mixing operator identically** — it is intra-shell and conserves shell
   population. So statistical-ℓ is the limit in which ℓ-mixing *cannot matter*,
   while ℓ-mixing-off is the limit in which it is *absent*. The reviewer's claim
   that these are near-opposite limits is exactly right.
2. The production model is **already at the statistical-ℓ limit for p ≥ 3**
   (steady-state departures: n=3 → 1.049/1.017/0.980; n=8 → within 1.6 %). It
   always was the like-for-like comparison.

Therefore the "bracket" argument at ch4:1124-1125 — that the two variants
straddle the tabulated value "almost symmetrically" — is a **false bracket**.
The statistical-ℓ answer sits on top of the production value, not between them.

**Corrected status: the `r_1` deficit at p = 2–5 is unexplained.** Factor 8–12
at lg n_e = 18, factor 1.8–2.0 at lg n_e = 21. State-space closure now joins
truncation (1.2 %) and Lyman trapping as eliminated non-causes.

**What survives.** The other four items in the ch4:1075-1147 list are untouched,
in particular item 3: reproducing the tabulated coronal asymptote demands
`K_exc(1s→n=2) = 1.44e-7 cm^3/s` at 11.03 eV, fifteen times the accepted value.
That is independent of the state space and remains the strongest argument that
the *target* is unverified. So ch4:1072-1073's "open comparison whose target is
unverified" is fine. What must go is item 5 and every sentence downstream that
says the disagreement *measures the ℓ closure*.

**The headline is untouched.** `f_3 − f_4` is a difference of two responses
inside one ℓ-resolved operator and never passes through a bundled coefficient.
ch4:1151-1153 is right about that, and right independently of why r_1 disagrees.

**Note:** `verify_fujimoto_table41.py`'s own docstring, under the heading
"TWO RUNS — AND NEITHER IS A STRICT FUJIMOTO BENCHMARK", already states the
reviewer's point and names the exact missing test. The thesis prose did not
follow its own script.

---

## 3. The bug that has been making the thesis look worse than it is

`src/validation/diagnose_gate_d.py:114-123` reads ADAS with `np.interp` on the
**raw linear coefficient in linear Te**, over a table that carries a `log10_K`
column it ignores, with only 7 Te nodes between 1 and 10 eV
(1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0). SCD rises by a factor **8.52** across the
single 1.5→2.0 eV interval. Linear interpolation of a convex function of that
curvature is biased high:

| Te | linear read / log-log read |
|---|---|
| 1.10 eV | **4.511** |
| 1.20 eV | 3.674 |
| 1.40 eV | 1.582 |
| 2.947 eV (benchmark) | 1.047 |

Additionally the density band `(0.5, 2.0)` admits **two** ADAS density rows at
the benchmark (1e14 and 2e14), so `np.interp` receives duplicated Te abscissae
and the benchmark n_e = 1.389e14 is never actually interpolated to.

Re-reading the same tables correctly:

| | as implemented | bicubic log-log |
|---|---|---|
| η_SCD within factor 2 | **323/400** | **400/400** |
| η_SCD range | 0.130 – 0.931 | 0.700 – 0.876 |
| η_ACD benchmark | 0.907 | 0.971 |
| median, 1–1.5 eV | 0.302 | 0.730 |

The qualitative truncation story survives (max η < 1, r = +0.42). The quoted
numbers do not. **Every error runs in the thesis's favour.**

Caveat: ADAS's canonical read is a spline in log10 space; the bicubic
approximates it. The definitive arbiter is `xxdata_11` on the raw
`scd96_h.dat` / `acd96_h.dat`, not run here.

---

## 4. Provenance — item 13, worse than he knew

`src/validation/audit_writers.py` **cannot fire on the dominant write idiom.**
Fault injection: two throwaway modules, each declaring "Report only. Writes
nothing.", each writing the same path via `with (out / "shared.csv").open("w")`:

```
fault_a.py resolved: [] | unresolved: [] | claims read-only: True
fault_b.py resolved: [] | unresolved: [] | claims read-only: True
```

Two scripts colliding on one output, both lying, completely invisible. Cause:
the mode test at line 111 reads `n.args[1]`, but for `p.open("w")` the mode is
`n.args[0]`. The liar-detector at line 165 is `if claims and (res or unres)` —
both lists empty, so it cannot fire either. Its verdict
"none — every read-only claim checks out" is a **wiring check, not a gate**.

The thesis says four live CSV writers are invisible to it (ch4:1668). An AST
count gives **38 call sites in 27 scripts**.

The three-path collision is live today: `qss_analysis.py:411-413` and
`validate_gates.py:462-464` both `np.save` `M_grid.npy`, `tau_QSS_grid.npy`,
`tau_relax_grid.npy`, both to a **relative** `OUT_DIR = 'validation'`. Exactly
six downstream readers. The arrays on disk carry mtime 2026-08-23 11:17:01,
matching `validate_gates.py`'s other outputs — nothing in the files records it.

To its credit `verify_inversion_error.py`'s gate **is** real: run with
`--frac 0.08` it raises `RuntimeError: window_ok disagrees at ('heat', 24, 7)`.

---

## 5. The two demanded tests

**Test A (ADAS reservoir) — valid, cheap, and it appears to pass.**

The identity holds exactly for this model. From the column-sum structure
`Σ_q L_{q,p} = −n_e K_ion(p)` (verified to 3.6e-11 over 400 points):

    u_CRE = n_g/n_ion = ACD_model / SCD_model        verified to 4.1e-13

The thesis's defence that ADAS is closed-system and this model open (ch4:1277,
ch2:1270, ch6:897) **does not block it** — the open/closed distinction moves the
approach to the fixed point, not the fixed point. A 44-state closed operator
gives the same u to 4.3e-6, eigensolver-limited. The identity is **already
implemented** in `verify_open_reservoir.py` as `alpha_eff/S_eff`; those two
quantities *are* n_e·SCD and n_e·ACD.

His *level* test is a repackaging: `u_model/u_ADAS ≡ η_ACD/η_SCD` to 1.6e-12.
Ten minutes, zero information.

His *derivative* test is the real result. `G ≡ dln u/dln Te` is Chapter 5's
central coefficient, and `G = dln ACD/dln Te − dln SCD/dln Te`, so ADAS supplies
an external `G_ADAS`. ch2:1273 confirms no ADAS data enters the rate matrix.

| ADAS read | G_model/G_ADAS min / med / max |
|---|---|
| as currently implemented | 0.294 / 1.092 / 2.239 |
| bilinear log-log | 0.855 / 1.004 / 1.181 |
| **bicubic log-log** | **0.989 / 1.008 / 1.023** |

Sign agrees at **392/392**. That is an external check at the 1–2 % level.

**Test B (n_max convergence) — not possible as specified.**

No atomic data exists above n = 15 anywhere in the repo:
`radiative_rates.py` `N_BUNDLED_MAX = 15`; `compute_K_VS.py`
`bund_high = list(range(11,16))`; `assemble_K_exc.py` `{n:(n-9) for n in
range(9,16)}`; `assemble_cr_matrix.py:137` `L = np.zeros((43,43))`. Raw CCC
quantum data stops at n=10; n=11–15 are already Vriens & Smeets semiclassical.

n_max = 18 and 20 require new rate generation across five scripts plus auditing
~32 downstream consumers of the fixed 43-state layout. `REMAINING.md:189` and
`HANDOFF.md:234` already say so: *"needs the rate pipeline rerun at n_max = 12
and 20, not a validation script."*

Downward (10, 12, 14) is reachable but has **no working script** — the only
historical attempt is the invalidated one that produced the spurious 22–37 %
jump. The numbers in Table 4.4 row 1707 trace to a markdown addendum only.

---

## 6. Things he could not see, that are worse than what he found

| site | defect |
|---|---|
| `chapter3.tex:1638` | *"no agreement with ADAS is claimed anywhere in this thesis"* — a universal quantifier the thesis falsifies 260 pages later at ch4:1271. Unchanged since his build. |
| `chapter5.tex:763` vs `:773` | Ten lines apart in one paragraph: "No script in the repository computes them and no artifact under `validation/` stores them" vs "reproduced to four significant figures by `verify_reservoir_gain.py`, whose output is stamped in `validation/reservoir_gain/`". Both cannot be current. |
| `chapter4.tex:1072` | "for five reasons", followed by six `\item`s. Correct in his build; a sixth was added without updating the count. |
| `chapter5.tex:1147` | "a cancellation to 6.8 per cent" is a **median of ratios** (6.76 %), printed beside three medians whose **ratio** is **8.50 %**. Moves 25 % with the definition; should not be quoted to two digits there. |
| Table 4.4 row 1707 | Graded **severe** — "a plausible wrong version of this model fails it" — with no producing script and no stamped artifact. A check that cannot be re-run cannot fail anything. Also scope-mislabelled: the table says "State-space convergence in n_max", the prose (ch4:682) says only `f_3 − f_4` was tested. |
| `chapter4.tex:813`, `432`, `454` | Markdown-only provenance, not disclosed as such, while ch5:761-773 and ch4:674-677 *do* disclose theirs. The Declaration's carve-out is honoured inconsistently. |
| HANDOFF / REMAINING | Current build measures **193 pages**. HANDOFF.md:21 says 192; REMAINING.md:3 says 190. None is script-produced. |
| `CLAUDE.md` known issues | Says `qss_analysis.py` carries `eigs[eigs < -1.0]`. Line 137 now reads `eigs < 0.0`, and `tau_QSS_grid` max is 67.23 s. The filter was fixed; the table never caught up. |

No script in the repository reads `thesis_tex/*.tex` and checks its numbers or
status words against `validation/`. **This entire class of defect has zero
automated coverage**, which is why §4.13 drifted.

---

## 7. Corrections to the reviewer, for the reply

1. **Item 3 is half wrong.** The η distribution *is* reported alongside the box
   count at ch4:1238-1242 (range 0.130–0.931, median 0.696; above 2 eV
   263/280, median 0.767). His "the meaningful information is the distribution,
   not the box" describes something the text already does.
2. **Item 11 is obsolete.** ch5:1664-1685 already demotes the bound, in his own
   terms: *"An inequality with a known counterexample is not an inequality."*
   Grep for `theorem` returns nine hits, none applying to `eq:elm_bound`.
3. **Item 9 is two-thirds closed.** His quoted string "and has not explained"
   no longer exists anywhere in `thesis_tex/`; it was repaired in `71691bf`,
   11 Sep 01:37. What survives is the word "fails" in the §6.7 title standing
   nine TOC lines from "The source resolves it".
4. **Item 7 is right about the error, over-agnostic about the fix.** The net
   sign *is* deducible once the channels are separated: in the ground-fed
   channel truncation is exactly equivalent to deleting upward collisional
   escape (×1.45 per shell); in the recombination-fed channel the removed loss
   and removed gain cancel to 0.2–0.3 % by Saha detailed balance (r_0 ×0.998).
   The correct mechanism is blocked **collisional** escape — radiative decay at
   n=15 is 8.16e4 against 1.9e11 collisional, five orders down.
5. **Item 14 is right, at two sites** (ch4:1286 and ch4:1350), and ch4:1273
   gives a third figure (10 %) for the same quantity.

---

## 8. Priority

**Text-only, no new physics — do these first.**

1. Withdraw the ℓ-closure explanation (§2 above). ch4:1118-1127, ch4:1148-1153,
   ch4:1717, ch6:1038-1049. Replace with "an open external disagreement with
   three demonstrated non-causes".
2. Rebuild Table 4.4 and §4.13 from the current claim state. 7 stale items.
3. Fix both 6 % sites; pick one figure for the ACD benchmark.
4. Rewrite the two terminal-shell sentences (ch4:708-710, ch6:1100-1103).
5. Requote the bundling margin as 100, not 1538; drop "three orders of
   magnitude" at ch6:1198.
6. ch3:1631-1638, ch5:763/773, ch4:1072, ch5:1147.
7. Rescope Table 4.4 row 1707 to `f_3 − f_4`, or drop its "severe" grade until
   it has a script.

**New computation, in value order.**

8. Fix the ADAS reader, requote §4.10. Half a day. Improves every number.
9. Run G_model vs G_ADAS. An afternoon. Buys an external check at the level the
   thesis actually claims at, which is what item 4 was really asking for.
10. n_max = 10/12/14 with a correct self-consistent truncation, if time allows.
    18/20 are out of reach before the defence; say so in scope.

**Do not attempt** a rate-pipeline extension above n = 15 before 15 October.

---

*Every number above was recomputed from the canonical arrays. The scratch
computations in §2, §3 and §5 are agent work and have not been reproduced on
the author's machine; nothing here has entered the thesis and nothing should
until it is re-run independently. That is the ✅ bar and this document does not
clear it.*
