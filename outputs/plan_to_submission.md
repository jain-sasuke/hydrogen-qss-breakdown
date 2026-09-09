# Plan to Submission

**Written 10 September 2026.** Defence ~21 September per
`outputs/handover/HANDOVER.md:4` — **11 days**. The date in `CLAUDE.md`
(1 September) is stale and should be corrected. **The submission deadline is
still unconfirmed and is the one input that can invalidate this whole plan.**

---

## 0. The blocking question — settle today, before anything else

`master_plan_v2.md:398` has flagged this since 23 August and it is still open:

> Confirm the submission deadline. If examiners need the thesis a week out…

If examiners require the document **one week before** the defence, the real
deadline is **14 September — four days** — and the plan below must be cut to
its Tier 1 only. Ask the supervisor today. Everything downstream is scheduled
against the answer.

**Second question, same email:** is a 7-chapter structure required, or will
5 chapters with merged material be accepted? The spine in
`thesis_architecture.md` PART 4 is ambitious for the time remaining.

---

## 1. Where the thesis actually is

| Chapter | File | State | Work left |
|---|---|---|---|
| 1 — Reading the Light from a Divertor | — | **does not exist** | write from literature |
| 2 — The Atoms | — | **does not exist** | write; describes an existing, working pipeline |
| 3 — Two Clocks | `chapter3.tex` | **complete**, 1610 lines | 4 targeted edits (§3 below) |
| 4 — Does the Model Work? | — | **does not exist** | write; gates exist, Gate D fails |
| 5 — What the Model Says | `chapter5.tex` + `chapter5_C5D.tex` | 4 stubs + 327-line side draft | write and integrate; **no figures exist** |
| 6 — What This Model Cannot Say | — | **does not exist** | largely already drafted, see §2 |
| 7 — What It Means | — | **does not exist** | short |
| front matter / abstract | `thesis_main.tex` | preamble + macros only, **no `\include`** | wire up, write abstract last |

**The binding constraint is Chapter 5, not the physics.** The physics is done
and, after this week, better verified than most M.Tech work. The document is
what is missing.

---

## 2. The accelerator nobody has counted

`findings_10_four_agent_review.md` is not only criticism. Large parts of it are
**near-final chapter text** and should be lifted, not rewritten:

| findings_10 section | goes to | what it gives you |
|---|---|---|
| §3.1 closure test (202→200, 38.68→38.56%) | **Ch 4** | a validation gate with a stated falsifier that did not occur |
| §3.2 lower-bound test (true/lower 0.9999–1.0066) | **Ch 4** | second gate, direct 43-state integration |
| §2 mechanism, three routes (Griem 7.2 vs 6.7) | **Ch 5** | the strongest single result in the thesis |
| §4.1 optical depth + ADDENDUM | **Ch 6** | quantified, not hand-waved |
| §4.2 transport, §4.3 molecules, §4.4 ridge ∝ n(1s) | **Ch 6** | the whole chapter, already written |
| §9 publication assessment | **Ch 7** | what the work does and does not claim |
| §12 defense one-liners | defence prep | five hostile questions, pre-answered |

**Chapter 6 is close to free.** Chapter 4 is half-written. Budget accordingly.

---

## 3. Tier 1 — must happen, in this order

### T1.1 Fix the register before writing from it (half a day, do first)

Writing Chapter 5 from `thesis_ready.md` PART A as it stands means writing
numbers that are falsified and rewriting them later. Cheapest possible ordering
is to fix the register first.

1. Demote **A8, A9, A10, A11** from ✅ to ▶️ and strike the W1–W6 sentences.
2. **A1:** τ_QSS floor → 75.4 ns unrestricted, or state the M > 900 scope on all
   six bounds. (`chapter3.tex` Eq. `M_range` is already right.)
3. **A10:** add column j = 4, or state the migration below 1.15 eV. Reconcile
   with A11, which sits at j = 4.
4. **A11:** restate inside §3.5.3 — *45 of 448 above 2 eV, worst 17.5%, zero
   breakdown at citable divertor densities where the worst is 7.2%* — and note
   it is invariant under Lyman trapping across a twentyfold slab range.
5. **Detachment:** delete the claim from `thesis_ready.md:294`,
   `chapter3.tex:1349`, `thesis_main.tex:336`. `chapter5_C5D.tex:284` is right.

### T1.2 Chapter 5 (3 days — the long pole)

Fold `chapter5_C5D.tex` into `chapter5.tex` and clear the four stubs. Order:
map → mechanism → ridge → divertor magnitudes → what it does not show.

Use the **exact** form `ε = |e^{S̄ Δln u} − 1|`. Do **not** use the linearised
`|f₃−f₄|·|ln x|` as the headline — `CH5_EVIDENCE.md` §2 and
`chapter5_C5D.tex:141` both say so, and `thesis_ready.md` A9 disagrees with
them. Lead the chapter on the **attribution** (density → sensitivity 7.11×,
temperature → displacement 76%), because the identity itself tests nothing.

Remove the `[MECHANISM NOT ESTABLISHED]` bracket at `chapter5_C5D.tex:270`
either by deriving ∂ln n_gs/∂ln Te at CRE or by saying in the text that the
temperature dependence is empirical. Do not let the bracket reach an examiner.

### T1.3 Chapter 5 figures (1 day)

There is currently **no Chapter 5 figure**. Minimum viable set, produced by a
script under `src/analysis/` following `make_ch3_figures.py`'s guard pattern:

1. ε_plateau map over (Te, ne), heating, with the window mask shown.
2. **The density line plot** — ε vs ne, one line per Te. This one is
   non-negotiable: a referee will ask for it, and if it does not look like a
   ridge, the word "ridge" must not be used. Present it beside the map.
3. M vs ε scatter, showing they are uncorrelated once (Te, ne) is controlled.
4. The trapping sensitivity: ε_plateau vs slab thickness at the worst point.

**Do not reuse anything already in `figures/`** except the four `fig3_*`. The
other 108 files predate the corrected matrix and 24 have no producing script.

### T1.4 Chapter 4 (1.5 days)

Gates A–E plus the two survived attacks from findings_10 §3.

**Gate D:** do not try to make it pass. It fails at 100% of points with a
systematic factor 9–65 and `acd_adas` is loaded but never used. In the time
available, **report the failure and the diagnosis** — SCD is the ionizing
coefficient while `SCD_model` sums over the full steady state, so ionisation
out of recombination-fed Rydberg states belongs under ACD. A failing external
gate, honestly diagnosed, is a stronger chapter than a silent omission.

**And rewrite `chapter3.tex:1377-1387`**, which currently promises a comparison
against ADAS SCD96/ACD96 that does not exist. A promise in the text with no
result behind it is the single easiest thing for an examiner to catch.

### T1.5 Chapters 1, 2 (2 days)

Ch 1 is literature: divertor Balmer diagnostics, the inversion practice, why
equilibrium ionisation balance is assumed. **Cite Sawada & Fujimoto (1994)** —
same system, same abrupt-step question, currently cited nowhere in the repo —
and **Greenland (2001)**, who stated that CR validity criteria are unrelated to
equilibrium timescales. Also Verhaegh et al. (2019) for the neutral fraction as
a free parameter in current practice. Not citing these is the fastest route to
a hostile viva.

Ch 2 describes a pipeline that already works: 43 states, ℓ-resolved to n = 8,
bundled 9–15, the data sources, and the two corrections found by audit
(ℓ-mixing F(U_m), the eigenvalue filter). Both belong here as evidence of
rigour, per `CLAUDE.md`'s "what good work looks like".

### T1.6 Chapters 6, 7, abstract, wire-up (1.5 days)

Ch 6 from findings_10 §4 and the ADDENDUM. Ch 7 from §9. Abstract last,
following the six-sentence structure already commented into
`thesis_main.tex:123-132` — but note sentence 5 there still says "at least 39%
at 1 eV", which T1.1 supersedes.

Add the `\include` lines to `thesis_main.tex`. **Compile early and often** —
the file has never been compiled with real chapters and 1610 lines of
Chapter 3 will surface macro collisions.

---

## 4. Tier 2 — do only if Tier 1 finishes early

- **`verify_bundling_psm20.py`** — written, never run, one command. Closes the
  truncation confound that `chapter5_C5D.tex §sec:ridge_alternatives` admits is
  untested. Highest value per minute of anything on this list.
- **Joint (Te, ne) ELM step map** — costs the same as the existing map and
  closes the "±5% is not an ELM" objection.
- **The r₁ deficit vs Fujimoto Table 4.1(b)** (factor 8.3 at p=3). Report
  ∂(ridge location)/∂(r₁ scaling). If a factor-3 change moves the ridge more
  than one grid interval, the ridge *location* must be withdrawn — the
  *mechanism* survives regardless.
- Resolve the ε_step name collision across all ten definitions (§5.11).

## 5. Tier 3 — explicitly cut, and say so in Ch 6

- Mori–Zwanzig. April, pre-correction, self-contradictory summary, and τ_K
  lives in the one block the robustness argument does not protect.
- Any attempt to implement transport or molecular channels.
- Any regeneration of the 108 stale figures.
- The paper. `findings_10` §9 has the target (J. Phys. B / JQSRT) and the
  contingency (the r₁ benchmark). It is post-submission work.

---

## 6. Schedule, against a 21 September defence

| Day | Date | Work |
|---|---|---|
| 0 | **Wed 10 Sep** | Email supervisor: submission deadline + chapter count. Then T1.1 register fixes |
| 1–3 | Thu 11 – Sat 13 | **Chapter 5** |
| 4 | Sun 14 | **Chapter 5 figures** |
| 5–6 | Mon 15 – Tue 16 | **Chapter 4** incl. Gate D honest report + the `chapter3.tex:1377` rewrite |
| 7–8 | Wed 17 – Thu 18 | **Chapters 1 and 2** |
| 9 | Fri 19 | **Chapters 6, 7**, abstract, `\include` wire-up, full compile |
| 10 | Sat 20 | Read end to end for contradictions. Defence slides from findings_10 §12 |
| 11 | **Sun 21** | Defence |

**There is no buffer.** If the submission deadline is earlier than the defence,
Chapters 1 and 2 compress to one day each and Tier 2 is abandoned entirely.

---

## 7. Risk register

| Risk | Likelihood | What it costs | Mitigation |
|---|---|---|---|
| Submission deadline is earlier than assumed | **high** | the whole schedule | Ask today. Nothing else on day 0 matters as much |
| `thesis_main.tex` has never compiled with real chapters | high | half a day of macro debugging | Wire in `chapter3.tex` and compile **today**, not on day 9 |
| Writing Ch 5 from the unfixed register | high | full rewrite of the results chapter | T1.1 first — this is why it is ordered first |
| Examiner asks for the density line plot | **certain** | the word "ridge" | Produce it in T1.3 and look at it honestly |
| Examiner has read Sawada & Fujimoto (1994) | moderate | credibility | Cite it in Ch 1 |
| Examiner asks about molecules at 1 eV | **high** | the divertor framing | Ch 6 already answers this; never pair 38.7% with "divertor" |
| A stale number survives into the text | moderate | a retraction at the viva | Grep the final PDF for 25 ns, 611, −46%, 1.18 µs, 38.7%, "detachment" |

---

## 8. Standing rules for the writing period

1. **Nothing enters a chapter that is not ▶️ or ✅ in the register**, and after
   T1.1 that excludes the old A8–A11 phrasings.
2. **Every number carries its scope.** Both live contradictions found this week
   — the τ_QSS floor and the Ly-α sensitivity — are regime-restricted numbers
   quoted globally. That is this project's characteristic failure mode.
3. **Report failures, do not repair them.** Gate D fails; say so. The r₁ deficit
   is open; say so. `CLAUDE.md` is explicit and examiners reward it.
4. **Corrections in place, never silent.** The ℓ-mixing error, the eigenvalue
   filter, the −46% Hα artifact and the renaming in `findings_09` are the
   thesis's strongest evidence of rigour. Give them a section, not a footnote.
5. **Back up daily.** `~/Desktop/nmcr_backups/` — and note that
   `data/processed/**/*.npy` is gitignored, so git alone does not protect
   `L_grid.npy`.
