# What is left

Updated 11 September 2026 (evening), after the Round 2 session. **190 pages, no
LaTeX errors, no undefined references or citations, zero em dashes. Defence
15 October 2026.** Everything from the Round 2 session is uncommitted.

## A0. Round 2 review, closed 11 September 2026

All ten demands plus the ramp point have a stamped artifact and updated text;
see `HANDOFF.md` §5b and backlog K5. Two loose ends remain from that work:

1. `validation/partial_correlation/partial_correlation_sweep.csv` has no
   provenance header (writer at `make_ch5_figures.py` ~line 882). Add the
   header and regenerate, which reruns the figure script.
2. ~~`make_ch5_figures.py` results lock~~ **Removed 11 Sep (evening)** on the
   author's instruction: the sign of the quadratic partial is now printed and
   the figure is drawn either way; chapter 5's sentence updated.
3. Chapter 4's 4f-fraction framing (lines ~764 to 770) now carries a
   qualifying sentence; the paragraph could be shortened.
4. The overshoot scan across the grid (456 of 784 pairs) lives in the ramp
   script's docstring from the skeptic pass, not in a stamped artifact; the
   thesis quotes only the benchmark value, which is stamped.


Every claim below is stated with the artifact that would settle it. Anything
marked CLOSED is closed in the thesis text, not only in this file.

---

## A. Closed this session

| item | how it closed |
|---|---|
| Six built figures not in any chapter | inserted with their generated captions; `figures/story_captions.tex` and `fig5_captions.tex` now load from the preamble |
| Transport selection effect not in the thesis | `verify_transport_selection.py`; Ch6 §transport_selection and §transport_partition |
| Ch7's Sawada and Fujimoto passage | rewritten from the source: they hold n(1) constant, write Eq. (4) and do not solve it, Eq. (23) bounds the fast states only |
| Abstract and front matter unwritten | certificate, declaration, abstract, acknowledgements, nomenclature |
| τ_QSS rename | τ_slow, 102 sites plus the figure scripts. Ch3 carries the reason |
| Gate D fails 400/400, ACD half unimplemented | `diagnose_gate_d.py`. Cause demonstrated; ACD built and agrees at 400/400 |
| `verify_bundling_psm20.py` silent fallback, never run | fallback removed, two more defects found, script run, result in Ch6 |
| Joint (Te, ne) ELM step map not run | `verify_joint_step_map.py`, 4968 cases; Ch5 §joint_step |
| The 202-versus-201 census count | 202, as 105 heating + 97 cooling, counted from `divertor_map.csv` |
| The crest vertex had no producing script | `verify_crest_subgrid.py`; 1.65e13 is the median of 49 rows, spread a factor 2.6 |
| Whether the eps_step zero locus exists at other densities | 6 of 8 density columns, 5.18 to 8.69 eV |
| The reservoir-gain aggregations had no script | `reservoir_gain_summary.csv`, one row per filter |
| Ch7 claimed Ch6 and Ch7 contradict on the molecular bound | Ch6 constructs the bound; Ch6's own contradicting sentence fixed |
| Ch3's promise to test the two recombination datasets separately | withdrawn where it was made; Ch4 states the gap is real |
| Ch2's F(U_m) density spread | 3.7 and 50 were different l channels; now stated with scope, and the n=8 spread runs 50 to 687 at 1 eV |

Three findings came out of that work that were not on the list:

1. **The eps_step zero locus is in 6 of 8 density columns.** Where it sits, the
   diagnostic is blind to temperature in steady state, which is a worse failure
   for a practitioner than the transient one this thesis maps. Reported, not
   pursued.
2. **The joint-step correction goes the wrong way in the worst case.** The
   benchmark spot check showed the ELM-averaged bound falling by a factor 4 with
   a density rise. Over the defended scope the count barely moves, 10.0 to 11.3
   percent, and the worst case *rises* 26 percent, 0.1748 to 0.2206. The
   fixed-density map understates rather than overstates.
3. **ACD agrees with ADAS at every point**, median 0.943 and 0.907 at the
   benchmark. That is the strongest external check in the thesis and it was
   available all along, hidden behind the half of Gate D that was never written.

---

## A2. Round 1 mathematical review, closed 11 September 2026

A reviewer audited sections 3.5, 5.4 and 5.5 and returned FAIL in current form
while confirming the mathematical core survives. Independently checked against
the text and the stamped artifacts; most of it landed. No computed number
changed anywhere in the repair.

| item | outcome |
|---|---|
| Sign inconsistency in the gain equation | **Real, diagnosis wrong.** Root cause: u- and u+ were never defined. Orientation now pinned to the artifact, `Delta ln u = ln(u+/u-)` |
| Derivative versus finite difference; "step-independent" heading | Already fixed by the author before the review arrived |
| `d eps / d ln Te` ill-posed at zero | **Correct.** `|e^z - 1|` has a corner; replaced by the ratio limit plus a smooth signed form |
| tanh bound "does not depend on the atomic data" | **Correct.** Split into the universal `|f3-f4| < 1` and the sharp `tanh(|Delta|/4)`, whose value moves with `Delta` |
| "No refinement of the rate coefficients can make that worse" | **Withdrawn.** Does not follow from sitting at 97% of the cap |
| ionising versus ground-fed | **Correct and worse than stated.** Chapter 3 contradicted itself 92 lines apart; two further instances fixed |
| switching point "a property of the network" | **Correct.** It is `u_m = c_m/a_m`; the `b_1` form carries `Z_1 n_e` |
| sum-rule integral limits | **Correct.** Now `-inf` to `+inf`, with a note that chapter 5's is a finite-interval object |
| inversion relation | **Correct.** Now carries its fixed-`n_e` assumption plus the 2x2 Jacobian form |
| generalise to the A-weighted emissivity | **Done.** New section sec:emissivity_generalisation, new script and artifact |

The emissivity generalisation closed a live overclaim: chapter 3 previously
stated the amplification bound for "a line ratio" while proving it only for a
shell ratio. It now proves it for any non-negative linear functional of the
fast-state vector. Verified at all 400 grid points: all four emissivity
coefficients strictly positive; `max|F_a - F_b|` reproduces `tanh(|Delta_j|/4)`
by direct maximisation to 9e-16; and `Delta_j` is invariant to the photon-versus
-energy weighting to 7e-16, since a constant per line cancels. `Delta_j` runs
0.673 to 2.196 against the shell 0.711 to 2.199, median disagreement 0.134% and
worst 5.4% at the low-density corner, which is where 4f matters most.

One repair caught an error of my own before it landed. The first draft of the
sign-convention paragraph said cooling reverses both factors so the product
stays positive. The data says otherwise: over all 2288 rows both `Sbar` and `G`
are negative in **both** directions, so the product is positive for heating and
negative for cooling, and `eps` is positive in both only because the equation
takes a modulus.

## B. Open, and needs a decision only the author can make

1. **PE versus a chemical-kinetics name.** The reviewer objects that partial
   equilibrium, partial LTE and partial Saha equilibrium already have meanings
   in plasma spectroscopy, and suggests frozen-reservoir QSS or conditional QSS.
   The τ_slow rename removed the worse of the two collisions; this one is still
   open. Cost of changing: one macro plus prose, smaller than the τ_slow change.

2. **Stangeby Part A entry.** Whether to cite it, and where.

3. **Chapter 4's register.** The reviewer calls it a forensic software report:
   filenames, line numbers, git history and repository archaeology in the main
   narrative. The scientific consequences of the bugs must stay. Whether the
   provenance detail moves to an appendix is an authorial choice about what kind
   of thesis this is.

---

## C. Open, and needs work

1. ~~**The reviewer's first and most serious demand is unmet.**~~ **CLOSED
   11 Sep 2026, and it forced the reframing the reviewer predicted.**
   Lomanowski et al. 2015 (Nucl. Fusion 55 123028) read in full. It infers Te in
   JET-ILW from the D 9->2 / 5->2 intensity ratio using ADAS PECs, and it does
   **not** eliminate n_1s/n_i: the defining figure plots the ratio against Te
   *and* the neutral fraction, drawn at an assumed n_0/n_e = 0.50. Bracketing
   the neutral fraction over 0.1 to 0.8 turns one measured ratio into
   Te = 1.8 to 2.5 eV, quoted as an upper limit. The closure they adopt is that
   the ratio is "mainly driven by recombination", following Lumma 1997 and
   McCracken 1998, available because "the contribution due to excitation is
   negligible for n_upper > 5 for Te < 2 eV".
   **Consequence, now written into Chapter 1 as sec:worked_inversion and into
   the abstract:** no procedure read for this thesis eliminates the ratio by
   imposing ionisation balance. The careful practice is the cancellation route.
   The thesis is therefore not a demonstration that divertor Balmer diagnostics
   fail; it is the cost of the closure that must be adopted where the
   cancellation is unavailable, which is precisely at n=3 and n=4. That is the
   narrower claim the reviewer said was defensible, and it is now the claim made.
   A bonus: the non-uniqueness of the inversion is stated in that paper and is
   found independently here.

2. **The observable is n_3/n_4, not a ratio of line integrals.** The reviewer's
   structural point: reformulate the derivation on the A-weighted emissivity
   ratio, which the affine decomposition should survive almost unchanged because
   line emissivities are linear functionals of the state vector. Chapter 4
   reports the shell-versus-line discrepancy as small but does not reformulate.

3. **Colour-blind check.** Eight existing Ch3 and Ch5 figures use `#2e7d32`
   against `#c0392b`, ΔE 4.2 under deuteranopia. The six new figures were built
   after this was known; the eight older ones were not.

4. **The appendices are empty stubs.** State ordering, atomic data sources,
   numerical methods and convergence: three `\chapter` headings with no body.

5. ~~**Fujimoto, *Plasma Spectroscopy* (2004), Chapter 4.**~~ **CLOSED
   11 Sep 2026.** Chapter 4 and Appendices 4A/4B read. Four things settled:
   (a) Table 4.1's density rows run lg n_e = 12 to 24, which is a plasma only in
   m^-3, confirming the reading Chapter 4 of this thesis already argued from
   Griem; (b) the columns are p with no l label, confirming the bundled-shell
   reading; (c) the chapter contains **no tanh, no logistic and no bound** on the
   difference of two population coefficients, which closes the largest remaining
   novelty risk; (d) Appendix 4B's figures are captioned as quoted from Sawada
   and Fujimoto (1994), so the book and the paper are one calculation, not two.
   The App. 4B timescale comparison was re-derived and reproduces the recorded
   0.34x and 1.77x exactly.

6. **The crest position's sensitivity to n_max.** Needs the rate pipeline rerun
   at n_max = 12 and 20, not a validation script.

---

## D. Remaining markers, 6

Closed 11 Sep 2026, on top of section A above:
- the fault-injection grid point, **recovered** rather than replaced. Sweeping
  all 400 points and matching on the three recorded max|dL| values identifies
  [23,5], the benchmark, to 1.63 percent, which is the printed table's rounding.
  `verify_fault_injection.py` rebuilds L through the pipeline's own assembly,
  checks the rebuild reproduces L_grid exactly, and reproduces every residual.
- the controlled-correlation sweep. The unspecified "eight combinations" are
  replaced by a named 4 scopes x 4 bases sweep stamped in
  `validation/partial_correlation/`. The partial is **negative in 12 of 12**
  combinations under quadratic control or richer, range -0.272 to -0.769.
- three source-required markers, all closed by producing scripts.

## D-old. Remaining markers, 20

11 `\todo`, 3 `[UNVERIFIED]`, 3 `[SOURCE REQUIRED]`, 1 `[MECHANISM NOT
ESTABLISHED]`, 1 `\needcite` in the text and 1 in the macro definition.

Most of the `\todo`s are citations the author must supply: the ITER divertor
heat-flux figure, the lower end of the density range, the ITER neutral-pressure
range, the separatrix band. Those cannot be closed from inside the repository.

The rest name work that is genuinely outstanding: the SOLPS-coupled calculation,
the molecular matrix, the n=15 ground-fed provenance, and the fault-injection
grid point.
