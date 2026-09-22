# Master correction register (Rounds 0 to 10 consolidated): adjudication, 22 September 2026

Adjudicated with the review-audit skill against the live tree (branch
`backup/verification-session-2026-09-10`, HEAD `3b613c1` plus this session's
uncommitted edits, which include the Round 6 application, K23). Three
read-only subagents (evidence-auditor and cr-physicist roles) checked every
item against the CURRENT text and the stamped artifacts, never against the
register's description. The register was written from a build two rounds old
and says so ("deduplicated ... across Rounds 0 to 10"); the first question on
every item was therefore whether the defect still exists.

## MUST items

| ID | Item (restated) | Verdict | Check that settled it |
|---|---|---|---|
| M1 | one sign convention for S, S̄, G, Δln u everywhere | Already applied (K23) | Nomenclature main:307, ch5:616-645, ch7:114-123 all agree: S̄ < 0, G < 0, S̄G > 0, S̄GΔln Te signed by direction, Δln u = ln(u⁺/u⁻); abstract carries no sign sentence |
| M2 | RR asymptotics reversed | Already applied (K12) | ch2:591-604 against `validation/rr_limits/`: slopes −0.50 and −1.36, monotone, no maximum |
| M3 | trace 0.32902966 / 0.24075077 | Already applied (K15) | ch5:1902-1912 states the +0.6 eV four-interval provenance and the operating points |
| M4 | §4.13 "not been run" | Already applied (K13, K20) | zero hits in §4.13 |
| M5 | "upper estimate" for transport | Already applied (K11, K18) | zero hits in ch6, ch7, abstract; ch6:589, ch7:522 state the conditional reading |
| M6 | S "untouched by transport" | Already applied (K18, K23) | every hit refers to the atomic function; the sampled S̄, G, ridge are stated conditional |
| M7 | molecular generalisation preserves logistic form | Not a defect | ch6:740-747, ch7:665-676 restrict survival to a fixed molecular ratio |
| M8 | molecular "bound" → sensitivity scenario; fractions attributed | Already applied (K18, K23) | ch6:688-700: φ3 from Dα, φ4 chosen, Dγ belongs to n = 5 |
| M9 | local vs line-of-sight | Already applied (K23) | ch1:347-350 chord average stated; ch7 scope 0-D |
| M10 | "Te at known ne" everywhere | Partially; residual applied today | ch7:36 opening question requoted to known density |
| M11 | 448 = 166 + 44 + 238 | Partially; residual applied today | abstract main:238 and ch5 summary now carry the 44 |
| M12 | "closure has to be adopted", "two honest ways" | Already applied (K11, K23) | ch1:505 three ways; ch1:566 "neither carried as an unknown nor supplied from outside" |
| M13 | universal |S| < 1 vs tanh cap | Already applied (K11); ch5 summary residual applied today | ch5:2626 now names the ceiling and the dataset-specific cap separately |
| M14 | Te ≥ 2 eV as adopted working domain | Partially; two captions applied today | `story_captions.tex:128`, `fig5_captions.tex:99` and their generator templates requoted |
| M15 | "It is not detachment" | Already applied (K11, K18) | zero live hits; "not a detachment criterion" at ch5:1557, ch7:551 |
| M16 | joint stress test qualified | Already applied (K15, K23); one fragment fixed today | ch7:233 dangling "The" removed |
| M17 | high-density "nothing breaks down" | Already applied, verified | 7.2 % reproduces from `divertor_map.csv` (108 pairs, max 0.0717) |
| M18 | timescale separation "does not predict any of it" | Not a defect | ch7:256-261, ch5:2285 "M supports the reduction ... cannot rank the error" |
| M19 | "eight significant figures" | ch5 applied (K15); ch4 residual stamped today | `verify_saha_balance.py` → `validation/saha_balance/` (see K24) |
| M20 | 392/338, 784/680 denominators | Already applied (K15, K17) | no sentence calls all candidates analysed |
| M21 | editorial markers | Already applied (K15) | only the `\todo` macro definition and one comment remain |
| M22 | ramp as operator-ramp sensitivity test | Superseded (K21) | the true L[Te(t)] ramp is now computed and quoted |
| M23 | conservation as fixed-density inference | Body applied (K18); Fig 5.8 caption applied today | caption and template requoted with the self-consistent 1.98 to 2.05, 67 and 7 |
| M24 | "nothing above to cascade out to" | Already applied (K13) | ch4:703, ch6:1075 give the corrected mechanism |
| M25 | Fujimoto r1 unresolved | Already applied (K11, K13); ch7 §7.4 sentence added today | ch4:1191, ch6:1320; ch7 now names the open factor 8.3 |
| M26 | substitutions not "uncertainty" | Already applied (K12) | residual "systematic" hits are legitimate uses |
| M27 | Declaration | Already applied (K13) | main:164-167, no retraction |
| M28 | two-writer .npy arrays | Needs author (repair) | text honest at ch4:1730-1750; both scripts still write |
| M29 | audit coverage | Already applied (K13) | ch4:1736-1746 states the defect and the 45 call sites |
| M30 | n_max claim reproducible | Already applied (K14) | ch4:658-668, appC:121-141 cite the stamped scan |
| M31 | "independent party" | Partially, needs author | method stated at ch4:1632; the party unnamed; if AI-assisted, say so |
| M32 | rewrite chapter 7 from settled claims | Done by adjudicated patching (K11 to K23), not from scratch | every stale claim the register lists is absent; the author may still restructure |
| M33 to M35 | forward criterion, secant, 3.5 % ratio | Already applied (K23) | ch7:143, 136, 175 |
| M36 | five summaries synchronised | Partially; six residuals applied today | ch5 summary tanh wording and conditional sentence, "3.5 at the worst heating pair", ch6 "108 pairs", ch7 §7.4 Fujimoto sentence; the 45/43 census split and the 100/506 µs ordering are consistent and left |

Counts: 36 MUST items; 27 already applied before this register arrived (in K11 to K23), 6 with residual sentences applied today (M10, M11, M13, M14, M16, M23, M36 overlapping), 1 superseded by new work (M22), 1 stamped today (M19), 2 needing the author (M28 a repair, M31 a disclosure), 2 not defects (M7, M18). Nothing fabricated.

## SHOULD items

| ID | Verdict | Evidence |
|---|---|---|
| S1 | Already done (K12), in ch2 §2.3 rather than Appendix B | ch2:426-466 closes 3117/3115 and 2190/1740 |
| S2 | Open, author's decision | no `validation/MANIFEST.csv`; `outputs/claim_evidence_table.md` is the nearest |
| S3 | Open, author's decision | a clean regeneration from an empty output directory |
| S4 | Already done (K13, K20) | ch6:1172 total-loss comparison |
| S5 | Already done (K12) | ch2:992-1013 |
| S6 | Already done (K12) | ch2:314 tail bound 3e-41 |
| S7 | Mistaken as a defect | ch2:839-850 states the policy with the sizes computed; the 13.6058 versus 13.605693 split is 0.0008 % |
| S8 | Already done (K13) | appC:75-80 |
| S9 | Already done (K15) | zero hits |
| S10 | Already done (K15) | ch5:1415-1424 |
| S11 | Author's call | §5.9 is physics-led; 49 of 216 lines carry statistics |
| S12, S13 | Already done (K23) | |
| S14 | Open, author's decision | schematic |
| S15 | Applied today | `\hypersetup{pdftitle, pdfauthor}` |
| S16 | Already done | no `\todo` left in the acknowledgements |
| S17 | Applied today, see the cleanup section | |

## PAPER-LATER items

P1 (ADAS reservoir) is done and in chapter 4 (K13); P2 (n_max of u_CRE, τ_slow) is done (K14), with G, S̄, ε_plateau covered indirectly (appendix C, K23); P3 (physical Te(t) ramp) is done (K21). P4 to P11 remain future work as the register says. The register's ordering "P1 first" is therefore moot: it was done before the register was written.

## What is worth the author's time

**Act now (applied).** The six M36 synchronisation sentences, the three captions with their templates, the Saha stamp, the PDF metadata.

**Author's decisions.** M28 (choose one writer per timescale array), M31 (name the independent party or state that it was an AI-assisted re-implementation), S2, S3, S14, and whether to restructure chapter 7 from scratch (M32) rather than keep the adjudicated patches.

**Ignore.** M7, M18, S7 (not defects); P1 to P3 as "missing" (done); every item marked already applied.

## Writing practice: what the sources say, and what was changed

The reviewer's Round 11 cleanup list names phrasing, not physics. The author asked what thesis-writing guidance actually says. A physics department's thesis guide (Lund, Mathematical Physics, "How to write a thesis") asks for "a dry, correct and objective account" and warns that "a common mistake is thinking of the thesis as a simple laboratory report, where you are tempted to list all your trials in chronological order"; a university writing centre's register guide (Coventry, "Style and Register") says directly addressing the reader "is overly friendly and chatty, and should therefore be avoided"; an examiner-facing guide (Thomson, "Don't give your thesis examiner a bad first impression") names generic or stodgy headings and a cluttered abstract as first-impression failures; the examiner study behind "Advice for writing a thesis (based on what examiners do)" asks for headings that identify what follows and a coherent whole. The IIT Kanpur guide is a format document and defers style to standard manuals.

Inventory before cleanup (live prose, comment lines excluded): "An earlier draft" 52, "earlier version" 7, `findings_10` 23, `ADDENDUM` 12, `thesis_ready` 3, `pivot_decision` 1, "the reader" 9, "It is worth" 16, "obvious" 15, "honest" 11, "attack" 12, "A chemical engineer" 6, "suspect" 5; headings "The suspect was innocent", "The question, and the suspect", "Two attacks, and what they were expected to do", "The open boundary: an attack that was expected to be fatal", nine subsections titled "The question" and four "The picture". Cleanup by science-editor subagents per chapter under a hard rule: every number, caveat, refuter and provenance pointer survives (a numeric-token guard checks each file), draft history is deleted or recast impersonally, working-note citations become "(project working note, no producing script; Table 4.4)" so the provenance disclosure stays, headings become descriptive. Results per chapter are in K25.
