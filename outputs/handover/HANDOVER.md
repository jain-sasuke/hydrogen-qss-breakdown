# Handover — continuing in a new chat

**Written 23 August 2026.** Read this first, then the files listed below.
Defence ~21 September. Submission deadline still unconfirmed.

---

## PART 1 — What to upload

### Must upload (the new chat is useless without these)

| file | why |
|---|---|
| `outputs/thesis_architecture.md` | the story, the title decision, the chapter spine, the voice rules |
| `outputs/notation_and_definitions.md` | every symbol fixed once; the two reference states |
| `outputs/findings_09_central_quantity_misnamed.md` | **the most important document.** Why the central claim was renamed |
| `outputs/thesis_ready.md` | what is verified and what is not, A1–A12 and B1–B8 |
| `outputs/master_plan_v2.md` | schedule, cut list, standing decisions |
| `outputs/operating_protocol.md` | roles, the task cycle, the rules derived from twelve chat-model errors |
| `chapter3.tex` | the only chapter written |
| `thesis_main.tex` | preamble and the macro set everything depends on |
| `src/analysis/make_ch3_figures.py` | the figure script, after nine guards |

### Upload if the work touches them

`outputs/derivation_07_transient_diagnostic_error.md` and
`derivation_08_fujimoto_benchmark.md` — needed for Chapter 5 and for the
Chapter 4 external gate. `outputs/derivation_01`–`05` if Chapter 3 or 4 is
being extended. `outputs/verification_ledger.md` before quoting any number.

### Do NOT upload

The old `chapter4.tex` from project knowledge — it holds Chapters 2–4 of the
*previous* draft, whose framing is superseded and whose numbers are stale
(M = 611, 15.3 µs, 25.0 ns). It will contaminate the new chat.

Also skip: `session_status_and_forward_plan.md`, `master_plan.md` (14 Jul),
`CHAPTER5_CORRECTIONS_REPORT.md`, and the 11 KB backlog copy. All superseded.

---

## PART 2 — The prompt

```
This is an M.Tech thesis in Chemical Engineering (IIT Kanpur, defence
~21 September 2026) on time-dependent collisional-radiative modelling of
hydrogen for divertor Balmer-ratio diagnostics.

READ FIRST, IN THIS ORDER:
  1. findings_09_central_quantity_misnamed.md  -- what the work actually
     measures, and why the previous framing was wrong
  2. thesis_architecture.md                    -- the story and chapter spine
  3. notation_and_definitions.md               -- every symbol, fixed
  4. thesis_ready.md                           -- what is verified (A1-A12)
     and what is not (B1-B8)

THE RESULT, IN ONE PARAGRAPH. A divertor spectroscopist inverts the
H-alpha/H-beta ratio against a table indexed on (Te, ne). We built a 43-state
l-resolved time-dependent CR model to test the approximation everyone worries
about -- that excited states equilibrate instantly. They do: the QSS closure
is exact to 8 significant figures on the plateau. That is not the problem.
The problem is that a (Te, ne) table also assumes equilibrium ionisation
balance, and the ground state does not equilibrate: the error reaches tens of
per cent and persists for tau_QSS, longer than an ELM. The mechanism is that
n=3 and n=4 are fed in different proportion from below (excitation) and above
(recombination), giving dlnR/dln b_1 = f_3 - f_4, which vanishes in both
supply limits and peaks where they compete -- which is detachment.

HOW I WORK.
  - One quantity per session. Physics -> hand derivation -> worked example ->
    brutal physics test -> code -> written note.
  - I derive; your job is to ask and wait, not to hand me answers. Stuck
    means give me the next step, not the result.
  - Ground truth is physics > math > code > documents. Documents are
    consistency checks, never authorities -- including the ones above.
  - Predict before computing. Write the expected answer and the observation
    that would refute it, first.
  - Provenance on every number: which script, which file, which grid index,
    which matrix SHA. No number without it.
  - Be skeptical and push back on math, physics, code and architecture.
    Agreement is not useful to me; finding the error is.

WHAT IS DONE. Chapter 3 is written (35 pages, compiles, four figures placed
and captioned). All numbers in it re-verified against the current matrix.

WHAT IS NEXT. [state one: Chapter 5 / the bibliography / Chapter 4 / figures]

CRITICAL WARNINGS.
  - Do not reintroduce "QSS breakdown" framing. The QSS closure is EXACT in
    the tested regime; what fails is the equilibrium-ionisation-balance
    assumption. A title or claim built on QSS breakdown will be the first
    thing an examiner attacks.
  - Six numbers are formally withdrawn (findings_09 W1-W6). Do not quote
    "M >= 86.8 everywhere", "confined to Te <= 2.947 eV", "202 of 680 grid
    points", "conservative lower bound", or the corr(log M, log eps) = +0.76
    headline.
  - The benchmark point is grid [23,5], Te = 2.947 eV, ne = 1.389e14. It is
    an illustrative anchor, NOT a published ITER operating point, and it has
    no citation. Never call it an ITER reference.
  - The ion density is a FIXED RESERVOIR. It appears in the source term and
    is never evolved. Every statement about the "slow subsystem" is about the
    ground state alone.
```

---

## PART 3 — State of the work

### Chapter 3 — written

35 pages, compiles clean, zero overfull boxes. §3.1–3.7 complete. Four figures
placed with captions; figure numbers follow first reference, so the file names
and figure numbers differ (recorded in the script docstring).

Every number in the chapter has been re-verified against the current matrix.
Corrections found that way: the conservation residual (2.59e-11 → 3.61e-11),
$\kappa(L_{FF})$ (an unsupported "< 1.6e5" → the measured 1.48e3–1.74e5), the
$b_1$ range (a four-point probe's 130–1.5e5 → the grid's 76–2.7e5), and the
$\lambda_0$ relation, which was quoted as a constant 2.286 and is in fact a
stepwise-ionisation enhancement ranging 1.16–29.5.

### Open in Chapter 3

- **Bibliography.** Four keys cited, none resolving: `Summers2006`,
  `Fujimoto2004`, `Anderson2000`, `Osterbrock1989`. Every project PDF is a
  scanned image with no text layer, so the entries must come from DOIs.
- **Second verification pass** on $\eta$ (1.16–29.5), $b_1$ (76–2.7e5) and the
  $f_m$ values. All from a single script run.
- **Figures rendered once**, with layout code that has since changed twice.
- **Chapter 1 needs a citation** for the ITER divertor operating range. None
  exists anywhere in the project, and the range is asserted on five figures.

### The eigenvalue check that was never really run

Both the 22 August grid check and the first figure script tested
`max|Im| / max|lambda|` — global against global. The spectrum spans nine
orders of magnitude and $\lambda_0$ is the smallest, so a complex slow mode
could be hidden entirely behind a fast one. Verified by construction: a mode
that is 2.5% complex passes a $10^{-10}$ threshold under the global metric.

The figure script now uses a per-eigenvalue test. **Until it runs, the claim
that the eigenvalues are real grid-wide is not established** — and Table 3.1
and the $\tau_k = 1/|\lambda_k|$ treatment depend on it.

### Chapters 1, 2, 4, 5, 6, 7 — skeletons only

Section headings with `% QUESTION:` comments in `thesis_main.tex`. Content
sources are listed per chapter in `thesis_architecture.md` Part 4.

---

## PART 4 — What was learned the hard way

Worth carrying into the new chat, because each cost a working session.

**A number quoted from a partial probe will be wrong.** Three separate values
in Chapter 3 came from four-point checks and all three moved when computed
grid-wide. Compute inside the script that uses the number.

**A check whose scope is smaller than the claim proves nothing.** The
eigenvalue-realness check is the sharpest case: the metric was well defined,
the code ran, the answer was zero, and it did not test what it was cited for.

**Two files describing the same quantity, neither stamped, is how three
retractions happened.** `qss_analysis_summary.txt` and
`timescale_verification.csv` disagreed by nine orders of magnitude with
nothing to distinguish them. Every output file now carries script, date and
matrix SHA.

**Guards belong where regeneration happens.** A check in a script nobody
re-runs is worthless. The figure script now verifies the eigenvalue structure,
the channel residuals, the superposition identity and the positivity that
§3.5.4 asserts — because that script is what gets re-run when the matrix
changes.

**External review found things internal review did not**, repeatedly: the
missing $\partial R/\partial b_1$ derivation, the CRE-vs-transient mislabel,
the eigenvalue masking, the ADAS overclaim. Keep feeding raw output to a
second model cold, with no framing.

**And the figure loop ran too long.** Four rounds of code review before anyone
looked at a rendered figure. Generate, look, then review.
