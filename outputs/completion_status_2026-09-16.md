# Thesis completion status and what will remain incomplete

16 September 2026. Build: **195 pages, 0 LaTeX errors, 0 undefined references,
0 undefined citations, 0 BibTeX warnings, 0 em dashes in live prose.**
Defence 15 October 2026.

---

## 1. Done this session

| item | what changed | verification |
|---|---|---|
| Statistical-l bundle test | Re-derived and re-run independently. Chapter 4's l-closure explanation of the Fujimoto factor-8 is **refuted**: imposing Fujimoto's own closure moves `r_1(3)` by 0.42 %. | `max\|P@R - I\| = 0.0`; production `r_1(3)` reproduces the thesis's 6.897e-6; severity check moves `r_1(2)` by 130x on a non-statistical operator, so the test has power |
| Appendix A, *State Ordering* | Written from empty stub: 43-state table, bundling assumption, the two hydrogen ionisation energies, provenance | Sum g = 2480 = 2*sum n^2; `I_p * p^2` constant at 13.6058 eV; column sums to 1.8e-12. Builds clean |
| C3.1, C3.2 | Deleted two stale `[KEY REQUIRED]` comments | both keys verified present in `references.bib` |
| C4.1 | "for five reasons" -> "for six reasons" | enumerate has six `\item` at 1076/1084/1105/1111/1118/1129 |
| C5.3 | Deleted "No script in the repository computes them and no artifact under validation/ stores them" | `verify_reservoir_gain.py` (14 KB) and `validation/reservoir_gain/reservoir_gain.csv` (443 KB) both exist; the sentence was contradicted eight lines later in its own paragraph |
| C6.3 | Deleted a stale `\todo` | answered at `chapter6.tex:843` with `\cite[Sec.~3.2.2]{Pitts2019}`, sixteen lines above it |

Markers remaining: **3 `\todo` in chapter6, 1 in thesis_main** (was 5), 0 `[KEY REQUIRED]`
(was 2), 1 `[UNVERIFIED]`, 1 `[MECHANISM NOT ESTABLISHED]`, 1 unmarked aside that prints.

---

## 2. WILL REMAIN INCOMPLETE — cannot be closed before the defence

These are physics limits, not writing limits. Each should become an explicit
scope statement rather than a marker.

| # | item | why it cannot close | disposition |
|---|---|---|---|
| 1 | **Fujimoto `r_1` deficit at p = 2-5** | As of today the explanation is withdrawn. l-closure joins truncation (1.2 %) and Lyman trapping as *eliminated* causes. Factor 8-12 at lg n_e = 18. | State as an open external disagreement with three demonstrated non-causes. The other four arguments that the *target* is unverified survive, in particular that matching the tabulated coronal asymptote demands `K_exc(1s->2)` fifteen times the accepted value. |
| 2 | **n_max = 18, 20 convergence** (`chapter5.tex:1446`) | **No atomic data above n = 15 exists anywhere in the repository.** `radiative_rates.py N_BUNDLED_MAX = 15`; `compute_K_VS.py bund_high = range(11,16)`; raw CCC stops at n = 10. Needs new rate generation across five scripts plus ~32 downstream consumers of the fixed 43-state layout. | Convert `[UNVERIFIED]` to scope. Downward (10/12/14) is reachable but has no working script. |
| 3 | **Two-region / SOLPS-coupled ground density** (`chapter6.tex:628`) | Requires an edge transport code supplying the recycling influx. It is the single conditional under which every magnitude in Chapter 5 stands. | Already named impossible in-scope; convert `\todo` to scope prose. Carried as further work at `chapter7.tex:543`. |
| 4 | **Molecular channels inside L** (`chapter6.tex:724`, `chapter7.tex:504`) | Requires an H2/H2+/H- state space and rate set that does not exist here. | Keep the dilution bound, correct its numbers (the recomputation gives 6-9, not 3-5, and the diluted sensitivity changes sign near phi_3 ~ 0.75), state it as an estimate. |
| 5 | **Non-Maxwellian electrons** (`chapter6.tex:1214`) | The rate integrals do not accept an EEDF. | Scope statement. |

---

## 3. REACHABLE but not yet done

| # | item | cost | note |
|---|---|---|---|
| 6 | Appendix B, *Atomic Data Sources* | half a day | Entirely unwritten, prints as a title-only page. Must complement Chapter 2 (81 KB on the same subject), not retell it: a provenance table is the right form. |
| 7 | Appendix C, *Numerical Methods and Convergence* | half a day | Entirely unwritten. Material exists in `validation/operator_conditioning/`, `validation/scaling_tests/`, section 4.6. **Caveat:** the n_max convergence numbers it would carry have no producing script (item 11 below). |
| 8 | Requote ~13 numbers from artifacts already on disk | a day | Six stamped artifacts exist and are cited *nowhere* in the thesis: `lyman_optical_depth/`, `closed_nuclei/`, `open_reservoir/`, `molecular_channel/`, `ion_closure/`, `ng_scaling/`. They settle several numbers the text currently gets wrong. |
| 9 | Two-parameter (Te, ne) inversion Jacobian | a day | Not new physics: needs a second Balmer ratio on the same grid. |
| 10 | Rebuild section 4.13 and Table 4.4 from the current claim state | half a day | 7 of 24 checked items stale; the summary contradicts section 4.10 four hundred lines earlier. |
| 11 | n_max = 10/12/14 with a correct self-consistent truncation | a day | The only historical attempt produced the documented spurious 22-37 % jump. Needed to give Table 4.4 row 1707 a producing script; it is currently graded *severe* with no artifact. |
| 12 | Nomenclature table overflow | an hour | `Overfull \hbox 182.70pt` and `\vbox 165.43pt`: the table runs off the right edge *and* the bottom. Worst single typesetting defect. |
| 13 | Four em dashes in generated captions | 10 min | Fix in `src/analysis/make_ch5_figures.py`, then rerun. The `.tex` is generated. |

---

## 4. Numbers the thesis currently states that its own artifacts contradict

Each has a stamped artifact on disk. None is cited in the text.

| site | thesis says | artifact says |
|---|---|---|
| `chapter5.tex:2341` | Lyman-alpha depth 114 / 0.036 / 0.0023 | Reproduces **only** with the wrong Doppler width `sqrt(kT/m)`. Correct hydrogen values 80.294 / 0.025468 / 0.0016317. The cited script never emits an optical depth at all. |
| `chapter5.tex:2346`, `chapter6.tex:175`, `:199` | effective A wrong by "four to five orders of magnitude" | 1.8 to 3.1 orders |
| `chapter6.tex:178` | benchmark escape 0.0023 | that is `[23,4]` with the wrong width; benchmark `[23,5]` is 0.0039 |
| `chapter6.tex:286` | escape factor 0.9999 at D = 1 cm | 0.99862 |
| `chapter5.tex:2261` | quasineutrality correction +13.9 | **both numbers are right.** +13.9 is the fixed-`ne` inference and reproduces to 0.35 %; +1.05 is the self-consistent closed-parcel solve. They answer different questions. Needs relabelling and a second column, not renumbering |
| `chapter5.tex:2278` | correction never exceeds 5.4e-3 | that is heating-only; both directions gives 7.982e-3 and a 2.4 % plateau change. The artifact records `P1: REFUTED` |
| `chapter6.tex:593` | "upper estimates" of the error | `err/eps_plateau > 1` at **165 of 180** rows. "Upper" is not what tau_esc < tau_slow implies |
| `chapter6.tex:1193` | bundling margin 1538 | **RETRACTED 16 Sep: the thesis is right.** 1538 is stamped in `outputs/bundling/bundling_report.txt`, reproducible, from a script whose silent-fallback bug is already fixed. The "margin is 100" figure appeared only in this document and `round4_adjudication_2026-09-16.md`, with no producing script anywhere in the repo. It was a scratch-session assertion quoted as a measurement. The shell-equilibration-eigenvalue comparison may still be worth building, but it does not exist and must not be cited until it does |
| `chapter4.tex:1286`, `:1350` | ACD agrees "to within 6 %" at the benchmark | 0.907, i.e. **9.3 %**. 6 % is the median. `chapter4.tex:1273` gives a third figure, 10 % |
| `chapter5.tex:1147` | "cancellation to 6.8 %" | **the number is right**: 6.76 % is the stamped median-of-ratios. The defect is wording only, since the sentence implies it equals 0.091/1.07 = 8.5 %. Corrected 16 Sep |

---

## 5. Risks

1. **Nothing under `validation/` is in git.** Not one artifact, across ~40
   directories. `appendixA.tex` and this file are untracked too. The remote is
   public and the `.npy` data is gitignored, so git is not a backup of the
   verification record. Two artifact directories were accidentally regenerated
   at 02:43 today; determinism was verified afterwards (two runs differ only in
   the `generated` timestamp) so no numbers were lost, but the 11 September
   provenance stamps are gone and were unrecoverable.
2. **`audit_writers.py` cannot fire** on `Path.open("w")`, the dominant write
   idiom: 38 call sites in 27 scripts are invisible to it. Its clean verdict is
   a wiring check.
3. **The `eigs < -1.0` filter still exists** in live code at `solve_cr.py:269`
   and `check_mz.py:10`; `_timescales` is called from three places in
   `solve_cr.py`. CLAUDE.md's known-issues entry points at `qss_analysis.py`,
   where it was fixed on 23 August.
4. **`preflight.py`** looks for `verify_ch3_groupB.py` and
   `verify_grid_coverage.py`; the files are `verifych3_gb.py` and `grid.py`, so
   the suite's self-check reports two false negatives.

---

## 6. Recommended order

1. The ten requotes in section 4 above. Every number is on disk; this is the
   highest ratio of credibility recovered to time spent, and it removes ten
   places where an examiner checking an artifact finds a different number.
2. Rebuild section 4.13 / Table 4.4 (item 10).
3. Appendix B and C (items 6, 7).
4. Nomenclature overflow and the generated em dashes (items 12, 13).
5. Convert the five impossible items into scope statements.
6. Only if time remains: items 9 and 11.

The Fujimoto rewrite is deliberately not in this list. It changes a claimed
result, and under this project's own rule it needs reproducing on the author's
machine before the text moves.


---

## 7. Corrections to this document, 16 September 2026

Three claims in section 4 above were wrong when first written and are corrected
in place rather than deleted:

1. **The bundling margin.** This document asserted the margin is 100 rather than
   the thesis's 1538. That figure has no producing script and no stamped
   artifact; it originated in a scratch session and was repeated here as though
   measured. The thesis's 1538 is stamped and reproducible. Retracted.
2. **The quasineutrality correction.** Both +13.9 and +1.05 are correct under
   their own definitions. This document framed it as a factor-13 error.
3. **The cancellation percentage.** 6.8 % is correct. The defect was the
   surrounding wording, not the number.

The pattern in all three: a number computed in a scratch session, or an
agent's summary of one, was carried into a document and acquired the standing of
a measurement. That is the same provenance failure the thesis itself is being
audited for.
