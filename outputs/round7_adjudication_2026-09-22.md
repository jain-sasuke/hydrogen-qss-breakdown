# Round 7 (adversarial thesis-wide kill test): adjudication, 22 September 2026

Adjudicated with the review-audit skill against the live tree. Four read-only
subagents re-derived each claim from the current text and the stamped
artifacts before the reviewer's reasoning was read; favourable claims were
audited at the same standard, which is where the most valuable finding came
from. Park 1972 was obtained by the author during the session and read in
full, so the literature claim is settled from the primary source rather than
from an abstract.

## Verdict table

| # | Claim (restated) | Class | Verdict | Check that settled it |
|---|---|---|---|---|
| 1 | Chapter 3 introduces the observable as a line-integrated brightness, then says the geometry factors "cancel" and replaces it with the local ratio n3/n4, without a uniformity condition | algebraic, regime | **Correct** (severity moderate, not MAJOR) | The chord ratio is the n4-weighted average of the local ratio; it equals the local ratio only where that ratio is constant over the emitting region. Instrument calibration and the common geometric scale do cancel; the n_ion prefactor cancels pointwise; the spatial profile does not. Chapter 3 conflated the third with the first two at lines 942 and 1207. Verbatim defect confirmed. Severity is moderate because no number in the thesis is a chord quantity: every reported figure is a point property of L(Te, ne). |
| 2 | The Fujimoto r1 benchmark disagrees at exactly the two shells the observable uses, so "the atomic data are traceable and benchmarked where the observable lives" is too strong | attribution, evidence | **Partially correct** | Numbers exact: r1(3) 8.29x low, r1(4) 4.39x low (`validation/fujimoto_table41/`); consequence at the 1e12 scenario, cap +28.6 % median, sensitivity +29.5 % median, range −58 to +524 % (`validation/fujimoto_r1_consequence/`). The reviewer's "median ~30 %" is accurate. Overstated in one respect: that is one of two scenarios, and at the highest density the deficit is nearly common and the cap moves only −3.8 %. The status is already stated as open at length in chapter 4 and chapter 6, so the claim is not new; what was genuinely too strong is the one sentence conflating benchmarked INPUT rates with a benchmarked DERIVED coefficient. Corrected. |
| 3 | The percentages are conditional closed-parcel results, and the abstract sentence is too broad | scope | **Wrong as to the abstract, correct as to the physics** | The abstract already carries both, in bold: "What follows is the cost of that closure where it is adopted, not a demonstration that divertor spectroscopy fails", and closes "The magnitudes are conditional on it, and are reported as such." The 0 of 45 closed-parcel failure is confirmed from `validation/transport_selection/`, and chapters 5, 6 and 7 each state the conditionality. No edit needed; the reviewer read an older abstract. |
| 4 | Park 1972 proposed an Halpha/Hbeta temperature diagnostic for nonequilibrium hydrogen and is absent | literature | **Correct, and understated** | Verified via Crossref and then read in full from `refs/`. The omission is real and larger than the reviewer knew: Park does not merely have the idea, he assumes quasi-steady state within the excited levels, writes the same two-coefficient decomposition, and carries the normalised ground-state population as the third argument. Cited and conceded. |
| 5 | Three timescale arrays have two writers, so provenance fails and 9982 is unattributable | provenance | **Partially correct** (defect real, consequence refuted) | Both writers confirmed at `qss_analysis.py:411-413` and `validate_gates.py:462-464`, same three paths, no producer field. But the arrays feed no thesis number: their six readers all write figures and files that appear nowhere in `thesis_tex/`. 9982 comes from `verify_timescales.py` via `validation/spectrum_testpoint.csv` and `validation/timescale_verification.csv`, and recomputes bitwise from the canonical L_grid. A latent hazard, correctly disclosed, not a contaminated number. |
| 6 | The console criterion is not self-contained without an independent temperature | scope | **Correct, already applied** | Chapter 7 states it: "This is a forward criterion: it requires an estimate of the transient from outside the spectrum, since the ratio itself carries none." |

### Favourable claims, audited at the same standard

| Claim the reviewer cleared | Verdict | Check |
|---|---|---|
| Non-normality considered, spectral timescale compared against a norm, observable verified by full propagation | **Earned** | Computed explicitly in chapter 3 |
| Finite-step identity uses step-averaged S and G and recovers the direct plateau | **Earned on arithmetic, proves less than claimed** | 0.063613 against 0.063612, four figures; the thesis itself says the comparison "can fail only through arithmetic error" |
| Eightfold high-n ionisation and 3BR stress test moves eps_plateau by 0.7 % median, 3.9 % max | **Earned exactly** | `validation/high_n_ionisation_scaling/` at s = 1/8 |
| The ramp objection is closed and the step approximation changes the plateau negligibly in the worst-error region | **UNEARNED, and it exposed a real overstatement in the thesis** | See below |

## The finding that came from auditing a clearance

Chapter 6 said the step idealisation is "safe by one to two orders of magnitude
where a plateau is reported, not by three." Over the 45 breakdown pairs the
Damkohler ratio runs from 1.95 to 288 with median 11.2, all stamped in
`validation/divertor_map/`. Evaluating the chapter's own lag law
De(1 - exp(-1/De)) at those three values gives 0.78, 0.96 and 0.998 of the step
plateau. So at the least favourable pair the ramp reaches only 78 per cent, a
suppression of 22 per cent, which is the size of the errors the chapter is
reporting. "One to two orders of magnitude" describes the median, not the
bottom of the range. Corrected in place, with the arithmetic disclosed as
arithmetic on stamped ratios rather than a separate run.

## Park 1972, settled from the primary source

Park, JQSRT 12, 323 to 370 (1972), received 26 May 1971, Ames Research Center.
Read in full from `refs/0022-4073_2872_2990050-7.pdf`.

What he has, and the thesis now concedes: the Halpha/Hbeta ratio as an
electron-temperature indicator for a plasma out of Saha equilibrium; assumption
(d), "a quasi-steady-state distribution prevails within hydrogen excited
states", with its timescale-separation justification; the same two-coefficient
form, rho(p) = r0(p) + r1(p) rho(1); the normalised ground-state population
rho(1), which is the departure coefficient b1 of this thesis, as the third
argument; and a tabulated conversion from apparent excitation temperature to
true electron temperature.

Where the boundary falls, in Park's own words. He computes in the two limits he
calls terminal nonequilibrium, decaying where r0 dominates and rising where
r1 rho(1) dominates, and he names them terminal "because it represents the
condition in which any further departure from equilibrium does not affect the
emission ratio of hydrogen lines". That definition is exactly the vanishing of
the sensitivity this thesis measures. Chapter 3 already proves the same thing
from the other side: the sensitivity "vanishes in two limits, and only there",
when both shells are fed in the same proportion, and is largest in between.

So Park's tables are valid precisely where this thesis's sensitivity is zero,
and this thesis measures the interval between his two limits, where it is not.
Checked numerically on the model's own CRE populations: across all 400 grid
points the ground-fed fractions run f3 from 0.055 to 0.943 and f4 from 0.009 to
0.796, the difference never falls below 0.046, and it exceeds 0.1 at 343 of the
400 points. Zero points are Park-like. The benchmark f3 = 0.260 and f4 = 0.048
reproduce the values the thesis already prints.

This makes the concession large and the boundary sharp, and it is a better
position than before the paper was read: Park documents the far extreme,
Lomanowski documents the near one, and the unmapped middle is what is measured
here.

## What is worth the author's time

**Applied this round.** The two chapter 3 cancellation sentences; a new chapter
6 subsection, "One chord, one point: the line-of-sight integration"; the
chapter 6 step-idealisation margin; the chapter 4 input-versus-derived
benchmark distinction with the open r1 disagreement named; the Park
bibliography entry, a chapter 1 prior-art paragraph and a chapter 7 novelty
boundary.

**Author's decisions, unchanged.** The two-writer repair (real, latent, feeds
no thesis number); naming the independent party in chapter 4.

**Ignore.** Claim 3's abstract edit, which describes an older abstract; the
"FAIL" chapter verdicts, which rest on defects that were already corrected in
rounds 5 and 6.

## What the review missed

**The abstract is about 734 words. The IIT Kanpur guide limits an M.Tech
abstract to 300 words, about one page.** That is a formal submission
requirement, it is missed by a factor of two and a half, and no review in this
series has raised it. Cutting it is an authorial decision about what survives;
the conditionality sentences should outrank the atomic-calibration caveat the
reviewer wanted added, which is why nothing was added to the abstract this
round.

## Source calibration

Six substantive claims: one correct and understated (Park), one correct
(chapter 3 cancellation), two partially correct (Fujimoto wording, provenance),
one wrong as stated (abstract conditionality), one already applied (console
criterion). No fabrications, and the two verified numbers it quoted were exact.
Characteristic failure mode, consistent with earlier rounds: it reads a build
several rounds old, so a third of its findings are already closed, and it
inflates severity, filing scope sentences and a latent provenance hazard as
MAJOR against a thesis whose numbers none of them touch. Its strengths are
real: it found the one uncited paper that genuinely bears on the novelty
boundary, and its instinct about the ramp margin, though offered as a
clearance, pointed at a sentence that was overstated. Read its literature and
arithmetic claims closely; re-derive its severities and check its target build
before acting.
