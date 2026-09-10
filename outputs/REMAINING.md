# What is left

**Compiled 10 September 2026** by scanning the chapters, not from memory. Every
"still live" below was confirmed present in the current text.

Thesis state: seven chapters, 150 pages, clean build, zero em dashes, 62
verified references, ten of the ten review gates run.

---

## A. Wrong statements still in the thesis — 17, all confirmed live

These are the highest priority. Each was found by an audit, and **none has been
applied to the text.** An examiner can catch any of them.

### Chapter 3 — 7

| where | says | should say |
|---|---|---|
| `:222-225` | an error in one off-diagonal, a transposed index, or a missing back-reaction "all break this immediately" | **None of the three do.** All were injected and the residual did not move. Only an A/γ inconsistency is caught. The chapter's own line 286 supplies the standard |
| `:1026` | the split and the tanh bound "hold for a trapped-line rate matrix exactly as they do for an optically thin one" | **False, and Chapter 6 says the opposite about the same operator.** Trapping makes A depend on n(1s), which breaks the affine structure and the unit width |
| `:190-193` | three-body supplies "under 10%" of the feed into any level at 10¹² | **30 of 36 levels exceed 10%** at 1.00 and 2.95 eV |
| `:444-446` | v₁ is "distributed across the excited manifold" | **PR(v₁) = 2.64**, a vector on two or three components. Also name the norm: v₁'s ground weight is 1.0000 raw and 1.9×10⁻⁴ population-scaled |
| `:817` | M_eff floor 77.6 | **77.3.** 77.6 multiplies two extrema from different grid points |
| `:1538` | "nine orders of magnitude" | **4.8.** Line 632 says five for the same two numbers |
| `:276, :721` | "eight orders of magnitude spanned by the entries" | **~12 decades** at the benchmark, 11.9 to 15.0 grid-wide |

### Chapter 2 — 3

| where | says | should say |
|---|---|---|
| `:924` | the frozen Debye density varies F "by about 30%" | **×3.7 for n=2 and ×50 for n=8.** Keep the result: f₃−f₄ moves ≤0.42%, so ℓ-mixing is saturated |
| `:940` | 2s "cannot radiate" | Two-photon at **8.229 s⁻¹**. Line 192 concedes it. Saying it *strengthens* the argument by supplying the n_e-independent floor |
| `:554-555` | α_RR "peaks near Te ~ I_n" | **Monotonically decreasing** over the whole 1–10 eV grid for every level |

### Chapter 6 — 5

| where | says | should say |
|---|---|---|
| `sec:molecules` ×2 | "no bound can be constructed" | One can, from references cited in the same paragraph: **factor 3 to 5** on cold-corner ε_plateau |
| `:143` | σ₀ = 7.74×10⁻¹⁴ cm² | **5.47374×10⁻¹⁴.** High by exactly √2 (Doppler width built from √(kT/m)) |
| `:150-166` | τ = 114 per cm, and 7900 over 5 cm | **80.4 per cm.** 114×5 = 570, so the two are inconsistent by 13.9×, and neither appears in the cited artifact |
| `sec:transport` | free-streaming transit "6 to 10 µs" | **72 µs** with CX trapping the neutrals, against a 26 µs threshold. **This correction is in the thesis's favour** |

### Chapter 1 — 2

| where | says | should say |
|---|---|---|
| `:441-443` | a divertor "passes from the first into the second as it detaches" | Chapter 6 withdraws this and Chapter 3 refuses it. Chapter 1 still asserts it |
| `:124` | `\cite{Stangeby2023, Stangeby2023}` | The same key twice. Part A (Nucl. Fusion 63, 016016) has a `% VERIFIED` comment but **no entry** |

---

## B. The transport finding is not in the thesis at all

Gate 9's headline, verified independently: **not one of the 45 breakdown points
above 2 eV satisfies the closed-parcel assumption used to compute it** (median
τ_esc/τ_QSS = 0.0104 at 20 cm), while 26.5% of the other 635 window_ok pairs do.
The selection is structural: a point enters the census because τ_QSS is long,
and a long τ_QSS is exactly when neutral transport dominates.

Chapter 6 tested transport only at the cold corner, where it survived by 2.8×.
That was the wrong place to test it.

**The partition to write:** unconditional are the two-channel split, the
logistic, tanh(|Δ|/4), |d ln R/d ln b₁| < 1, the S̄ map and the exactness of the
QSS closure. Conditional on the closed parcel are G, ε_plateau, the census and
every ELM magnitude.

---

## C. Markers: 29

| kind | count | where |
|---|---|---|
| `\todo` | 15 | ch1 ×3, ch2 ×1, ch4 ×4, ch6 ×7 |
| `UNVERIFIED` | 9 | ch4 ×1, ch5 ×6, ch7 ×2 |
| `SOURCE REQUIRED` | 3 | |
| `MECHANISM NOT ESTABLISHED` | 1 | why \|G\| falls with temperature |
| `\needcite` | 1 | ch4, Janev ionisation coefficient |

Several are correct behaviour and should survive to submission. The ones that
should not: the Janev citation, and Chapter 5's remaining UNVERIFIED items that
now have artifacts.

---

## D. Three decisions only you can make

1. **PE versus PEA.** Chapter 5 coins "partial equilibrium". In chemical
   kinetics *partial equilibrium approximation* means eliminating a fast
   reaction **extent**, not a fast **species**. Your examiners are chemical
   engineers. Recommendation: keep the term, distinguish it in one sentence at
   first use.
2. **The molecular bound.** Chapter 6 says none can be constructed;
   `findings_10` D.4 constructs one. One of the two must change.
3. **The duplicate Stangeby citation.** Add Part A's entry, or delete the
   duplicate key.

---

## E. Not written

- **The abstract.** `thesis_main.tex:123-132` has the six-sentence skeleton, and
  its sentence 5 still says "at least 39% at 1 eV", which the pivot retires.
- **Front matter** beyond the title block.
- **Chapter 5 does not yet include the six new figures.** `fig5_5_reversal`,
  `fig5_6_trajectory`, `fig5_7_structural_maps` and `fig6_1_scope` exist with
  captions in `figures/story_captions.tex` and are not `\includegraphics`'d
  anywhere.

---

## F. Computation not done

1. **Gate D.** Fails at 400 of 400 points; the ACD half is unimplemented
   (`acd_adas` is loaded and never used). Validation ladder rung 9.
2. **`verify_bundling_psm20.py`** has never been run, and it synthesises grids
   silently when files are missing, violating CLAUDE.md rule 2. **Fix the
   fallback before running it.**
3. **The joint (Te, n_e) ELM step map.** Costs the same as the existing one and
   closes the "±5% is not an ELM" objection.
4. **The eight existing Chapter 3 and 5 figures fail a colour-blind check.**
   `#2e7d32` and `#c0392b` separate by ΔE 4.2 under deuteranopia, below the ΔE 6
   floor. The six new figures use Okabe-Ito and pass.
5. **`eigs[eigs < -1.0]`** survives at `solve_cr.py:269` and `check_mz.py:10`.
   Imported by nothing, which is luck rather than design.

---

## G. Two library trips, neither doable remotely

1. **Sawada & Fujimoto (1994), Phys. Rev. E 49, 5565, the body.** Its abstract
   says "the overall response of excited level populations to ionization and
   recombination rates was also examined". If that section treats a *changing*
   ground-state balance, `chapter7.tex:316-317` ("neither contains the other")
   is at risk, and so is part of Claim 6. **The single largest publication
   risk.**
2. **Fujimoto, *Plasma Spectroscopy* (2004), Chapter 4.** Not in any scanned
   corpus reachable from here. If a closed form for the ratio of two population
   coefficients appears there, the remaining novelty weakens further.

---

## H. What is closed, so nobody reopens it

The Fujimoto benchmark (bundled-n, transcription exact, ℓ-closure is the whole
explanation). The 2s partition (worst-case separation 86.5). Detailed balance
(7.31×10⁻⁹ and 0.999997). State-space convergence, ladder rung 6. The tanh
novelty question as mathematics (Yule 1912). Backlog C1, C2, C6. B8. The
condition-number provenance. All 261 em dashes.
