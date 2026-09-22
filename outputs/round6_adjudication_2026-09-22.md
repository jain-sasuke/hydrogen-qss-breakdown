# Round 6 review (chapters 1 and 7): adjudication, 22 September 2026

Adjudicated with the review-audit skill against the live tree (branch
`backup/verification-session-2026-09-10`, HEAD `3b613c1` plus this session's
edits to chapters 2, 4, 5, 6). Three read-only subagents re-derived every claim
from the current text and the stamped artifacts before the reviewer's reasoning
was read: chapter 1 (evidence-auditor and cr-physicist roles), chapter 7 §§7.1
to 7.3 (math-auditor and evidence-auditor), chapter 7 §§7.4 to 7.5
(evidence-auditor and cr-physicist). The reviewer read a 213-page build; the
tree builds at 207 pages and Round 5 (K18, 21 Sep) was applied after that
build, so the first check on every claim was whether the quoted text still
exists.

## Verdict table

| # | Claim (restated) | Class | Verdict | Check that settled it |
|---|---|---|---|---|
| 1 | §1.3: the chord ratio is the n4-weighted average of n3/n4 and equals the local ratio only for a uniform emitting region; the text claims the path length "divides out" | algebraic, regime | Partially correct | Eq. 1.5 (ch1:333-338) is written with chord integrals and is exact; scale factors do cancel; what the text omits is the profile of n3/n4. No number in the thesis uses a chord (0-D scope stated at ch1:913, ch2:1617, ch6:406). A one-sentence scope gap, not MAJOR. |
| 2 | Opening, Fig. 1.1 and §7.1 invert one ratio for (Te, ne) while §1.4 and §1.8 ask for Te at known density | notation | Correct but minor | Fig. 1.1 caption and abstract were fixed in K11 (story_captions:166-171 "at known density"); one residual at ch1:23-25 "reads off an electron temperature and an electron density"; §1.9 ch1:958-960 still promises a (Te, ne) inversion that ch7:292 says was not carried through. |
| 3 | "closure has to be adopted" ignores the other dispositions of u | literature | Correct, already applied | Gone; ch1:560-562 now "wherever it is not carried as an unknown, the closure is what remains"; residual: u supplied from a transport model is neither. |
| 4 | "only two honest ways" should be three (eliminate, specify, retain) | logic | Correct, already applied | ch1:499-500 already reads "three honest ways", routes one, two, three. |
| 5 | Chapter 1 asserts the ELM is a step change; it is a timescale motivation for an idealised perturbation | regime | Correct, and understated | ch1:687-689 and 877-878 present. From `validation/window_membership_exposure/` (K15): 237 of the 448 defended pairs have τ_slow below 100 µs, and the 45 census members all have τ_slow above 195 µs, so over much of the defended range the reservoir is faster than a 250 µs ELM rise and the step's plateau is never reached; chapter 5 knows this (ch5:167-176), chapter 1 does not say it. ch1:731 also gives "milliseconds" against the 250 µs cited at ch1:652. |
| 6 | §1.7 concedes the right prior art | literature | Correct | Six concessions each present with a resolving bib key (SawadaFujimoto1994, Bates1962, FujimotoMcWhirter1990, Fujimoto2004, Verhaegh2019, Greenland2001a/b, Verhaegh2021, Wijkamp2023). |
| 7 | "That is the whole of the gap" (Sawada/Fujimoto) overstates | attribution | Correct, and understated | Fujimoto ch. 4 (refs/, pypdf): App. 4B treats n(1) changing in time as a validity condition and defers its dynamics to App. 5B; ch4:1041 quotes that appendix's 1e-4 s ground-state time. So ch7:434-436 "the frozen reservoir is how the standard textbook treatment is posed" is contradicted by the source the thesis cites. |
| 8 | Greenland "settled ... twenty-five years ago" too final | literature, style | Correct but minor | Both Greenland papers are 2001 (25 years); "settled" overreaches because Greenland's criteria are per-system checks, and the thesis's own result (M does not rank the error) is such a check. |
| 9 | Negative novelty claim unscoped; tanh bound over-emphasised | scope | Correct | ch7:329-330 absolute ("has not previously been stated"); ch7:355-356 already search-scoped. Line count of §7.3: bound and logistic 74 lines, reservoir-vs-QSS 58, decomposition 8, Greenland 10, factorisation 7; crest mechanism 0; Σ = P + SG 0 (it lives only in §7.2:257-293). |
| 10 | The strongest contributions (error separation, factorisation, crest mechanism, P + SG amplification) are under-emphasised in §7.3 | not a physics claim (emphasis) | Correct as a coverage finding | Same line count. Applied as the F14 rewording of the closing paragraph, not as a restructuring: page budget is the author's. |
| 11a | "correct to eight decimal places" | precision | Correct | trajectory_census.txt:19 eps_track 8.66e-6 at the benchmark, 4.4e-5 grid-wide (ch5:290); K15 removed the same phrase from ch5 §5.1. |
| 11b | The eigenvalue-gap argument is weaker than the direct trajectory comparison | logic | Correct but minor | ch4 §4.8.2 states it as a theorem with hypothesis S ≠ 0 and two excluded points ([48,7], [49,7]); ch7:50-53 drops both; ch7:45-49 already leads with the direct integration. |
| 11c | "for the whole of τ_slow it is false" | wording | Correct but minor | Single exponential (ch5:1674): at t = τ_slow the departure is 37 % of its start. |
| 12 | §7.2: S̄ < 0 and G < 0 everywhere yet "their product is positive for heating and negative for cooling" is algebraically wrong; S̄ is not the ordinary mean of f3 − f4 | sign algebra | Correct (sentence wrong as written; no number changes) | reservoir_gain.csv, 1144 heating and 1144 cooling rows, no exception: S̄ < 0, G < 0, S̄G > 0 in both; S̄GΔln Te > 0 heating, < 0 cooling; operator_slope_decomposition.csv 448 pairs: f3 > f4 448, P̄ > 0 448, natural-sign SḠ < 0 448, identities to 5e-15. ch5:639-646 already states it correctly; ch7 dropped the Δln Te factor. |
| 13 | The console procedure is not self-contained (needs the true ΔTe) | scope | Partially correct | ch7:80-86 and 137-139 already say the ratio carries no record of the transient and the reader supplies Δln Te; only the label "forward criterion" is missing. |
| 14 | G called a secant but the procedure says "differentiate" | consistency | Correct | ch7:113-114 vs 135-136; ch5:690 secant. |
| 15 | "1 % costs the Balmer inversion 3.5 % in the line ratio" conflates ratio and inference | interpretation | Correct | max |S̄G| over the Te ≥ 2 eV heating window-passing rows = 3.5353 (over both directions 3.580 at cool [15,3]); the inferred-Te error needs the slope, median amplification 8.3 (K8). |
| 16 | "decays as a single exponential" too exact | precision | Correct but minor | Estimate/true 0.979 to 1.022 (100 µs), 0.967 to 1.038 (506 µs), 0 of 448 beyond 5 % (trajectory_census). |
| 17 | 166 + 238 ≠ 448; the 44 fold-crossing pairs are missing | arithmetic | Correct | K8: 448 = 166 off-table + 44 fold-crossed + 238 clean; "44" absent from ch7. |
| 18 | "understates the worst case" is unqualified | scope | Correct | joint_step_map.csv: +1 index in density is a factor 2.68; ch5:2118-2121 carries the qualifier, ch7:224 dropped it. |
| 19 | "where the Balmer inversion is ordinarily applied" needs model qualification | scope | Correct | ch1:586-591 already says practice uses n ≥ 5 in detached conditions. |
| 20 | Final novelty claim: "two-parameter table", "arbitrary operating point", "properties of the plasma rather than the disturbance" | overclaim | Correct | reservoir_gain.csv at [23,5], k = 1, 2, 4: heating S̄ −0.2288 / −0.2457 / −0.2782 (21.6 %), G 5.5 %, cooling S̄ 29.6 %; K8's inversion is at known density. |
| 21 | "a ceiling no atomic dataset can raise" ambiguous | scope | Correct but minor, already applied | ch7:471 now "whose form no atomic dataset changes"; both bounds stated at §7.2:183-189. |
| 22a | §7.4: "fast transport means the mechanism does not operate" | physics | Correct, already applied | Gone; ch7:499-502 states the reviewer's own synthesis. |
| 22b | "Chapter 5 magnitudes are upper estimates" | scope | Correct, already applied | Gone; ch7:503-505 "conditional ... neither bounds nor estimates". |
| 22c | "structural S results untouched by transport" | physics | Partially correct | ch7:505-507 right in intent, does not say which results are structural; ch6 §6.2.5 lists the S̄ map and ridge as conditional; ng_scaling moves the ridge 1.70e13 to 2.30e12 under ×0.1 in n_g. |
| 22d | "Te ≥ 2 eV is the only self-consistent region" | scope | Correct | ch5 §5.10.3: quasineutrality contour 1.41 to 1.50 eV, Lyman contour 1.01 to 2.36 eV; a working boundary. |
| 22e | "It is not detachment" | scope | Correct, already applied | Gone; ch7:520 "The crest is not a detachment criterion". |
| 23 | Molecular paragraph called a "bound" | naming | Partially correct | Body says "sensitivity scenario"; one residual "bound" at ch7:542-543 and the fixed-u(t) caveat of ch6 §6.3 missing. |
| 24a | "A density grid fine enough to name an ELM" overpromises | rhetoric | Correct but minor | transport_selection: 0 of 45 defended breakdown pairs closed; a finer grid refines a closed-parcel census. |
| 24b | The ADAS reservoir check u = ACD/SCD vs u_CRE, Δln u, G is missing and should come first | evidence | Wrong (stale) | Stamped 18 Sep (K13, `validation/reservoir_vs_adas/`): u_model/u_ADAS median 1.18, Δln u per step median difference 5.8 %, ACD 400/400; in ch4 §4.10. Residual: chapter 7 never cites it. |
| 24c | No stamped n_max convergence of u_CRE, G, τ_slow, ε_plateau | evidence | Partially correct | K14 converges u_CRE (−0.13 to +0.03 %), Δ, τ_slow (−1.6 to −0.3 %) and Δln u (≤ 2.3e-5 between n_max 14 and 15); G, S̄ and ε_plateau have no direct scan. |
| 25 | §7.5 molecular paragraph is right (independent H2 reservoir gives R(u, v)) | mathematics | Correct | Numeric check with random a, c, m: at fixed v the bound tanh(|Δ(v)|/4) holds exactly with a v-dependent Δ; ch7:639-641 is more pessimistic than the algebra and than ch6 §6.3. |
| 26 | The reviewer's own novelty statement | scope | Partially correct | Everything in it is already in §7.1 and §7.3 except "strongly amplifying ... near a fold", which the thesis does not establish: the K8 numbers (47.7 % median, amplification 8.3) exclude the 44 fold-crossed pairs; the fold is Σ = 0. |

Counts: 35 sub-claims; correct in some degree 33 (of which already applied before this round 8, understated 3, minor 8, partially correct 7), wrong 1 (24b, stale), not a physics claim 1 (10). Nothing fabricated. The headline "FAIL" rested on items 1, 2, 12 and 22; of these 22 was already applied, 1 is a scope sentence, 2 a residual sentence, and 12 is real.

## What is worth the author's time

**Act now (applied this session).** 12 (the sign sentence is wrong as written); 17 (the 44 missing pairs); 7 (the Fujimoto attribution is contradicted by the cited source); 5 (chapter 1 asserts the step as physical fact where chapter 5 has measured that the reservoir is faster than the ELM rise at 237 of 448 pairs); 1 and 2 (the two scope sentences of the measurement model).

**Cheap, applied.** 3, 8, 9, 11a to 11c, 13 to 16, 18 to 20, 22c, 22d, 23, 24a, 24c, 25: each one sentence, every number from a stamped artifact.

**Defer (author's decision).** 10, the re-weighting of §7.3 toward error separation, the crest mechanism and P + SG: real, but it is a restructuring under a page budget the author holds; the closing paragraph was reworded instead.

**Ignore.** 24b (already done), 22a, 22b, 22e, 4, 21 (already applied); the section grades (not physics).

## What the review missed

Chapter 1: ch1:713-722 rests the slow clock on the 13.6 eV barrier where at 1e14 to 1e15 the ground state empties stepwise through n = 2 at 10.2 eV. Chapter 7: ch7:512-515 still quotes the fixed-density inference (68 of 392 above 10 %, 36 above 100 %, 5.4e-3) without the self-consistent values K18 put in ch5 §5.10.1 (67 and 7). Both applied.

## Source calibration

This review: 33 of 35 claims correct in some degree, 1 wrong because stale, 0 fabricated. Characteristic mode: reading a build two rounds old, so a quarter of the findings were already applied, and severity inflation on framing items (1 and 2 filed MAJOR). Where it was strong: the sign algebra of 12, the arithmetic of 17, the attribution of 7 and the ELM-step point of 5, all real and two of them understated. How to use the next one: act on its algebra and arithmetic directly; check its "missing calculation" claims against the backlog first; re-derive its severity labels.
