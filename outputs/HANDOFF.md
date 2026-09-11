# Handoff: non_markovian_cr thesis

Written 11 September 2026. Everything a fresh session needs to continue without
re-deriving the context. Read this first, then `CLAUDE.md`, then
`outputs/REMAINING.md`.

---

## 1. The project

M.Tech thesis, Chemical Engineering, IIT Kanpur. **Defence 1 September 2026.**
Time-dependent collisional-radiative (CR) modelling of hydrogen, quantifying what
a Balmer-ratio divertor diagnostic costs when it assumes ionisation balance has
settled.

Repo `/Users/phi/Desktop/non_markovian_cr`, branch
`backup/verification-session-2026-09-10`, remote
`github.com/jain-sasuke/NonMarkovianCR` which is **PUBLIC**.

**Current state: 187 pages, 0 LaTeX errors, 0 undefined references, 0 undefined
citations, BibTeX clean, 0 em dashes, 66 bib entries.**
Markers: 4 `\todo`, 1 `[UNVERIFIED]`, 0 `[SOURCE REQUIRED]`,
1 `[MECHANISM NOT ESTABLISHED]`. Down from 29 at the start of this work.

---

## 2. Environment, and the traps in it

**Python is not the default one.** Use `/opt/anaconda3/envs/cr/bin/python`.
The system `python3` has no numpy/scipy/matplotlib. Nothing in `src/` will run
otherwise.

**Build the thesis with:**
```
latexmk -pdf -cd -interaction=nonstopmode thesis_tex/thesis_main.tex
```
The `-cd` matters: chapter files are `\input` by relative name. Check
`grep -c '^! ' thesis_tex/thesis_main.log` and
`grep -ci 'warning--\|error message' thesis_tex/thesis_main.blg`.

**The shell is zsh.** Unquoted `--include=*.py` and `set -- $var` fail. Quote
globs. `timeout` does not exist. Use heredoc + `git commit -F -` for commit
messages, never `-m` with quotes.

**`refs/` is gitignored** and must stay that way. It holds downloaded papers
under publisher copyright; the remote is public. `.gitignore` has a `!*.pdf`
un-ignore earlier in the file that `refs/` overrides.

**PDFs can be read directly.** `pypdf` is installed in the `cr` env. Papers in
`refs/` were read this way, including a 42-page accepted manuscript.

**Web search budget is exhausted** (200/200). `WebFetch` still works. Crossref
(`api.crossref.org`) and Semantic Scholar (`api.semanticscholar.org`) answer;
IOP and ScienceDirect are bot-walled; OSTI `servlets/purl/<id>` serves PDFs.

---

## 3. The argument, and the numbers that carry it

A Balmer ratio is inverted for `(Te, ne)` against a table built at
collisional-radiative equilibrium. Indexing on two axes requires the
neutral-to-ion ratio `u = n_1s/n_ion` to be a function of those two axes, which
holds only once ionisation balance has settled. A divertor during an ELM has not.

**The reversal.** The approximation everyone doubts holds; the one nobody
examines fails. The excited-state QSS closure is accurate to `8.66e-6` at the
benchmark and `6.73e-9` at the worst diagnostic-error point. What does not keep
up is the ground state.

**The exact factorisation.** `eps_plateau = |exp(Sbar · G · Dln Te) − 1|`, with
`Sbar` the mean sensitivity (belongs to the observable) and `G = Dln u/Dln Te`
the reservoir gain (belongs to the plasma). Both are **secants** over the step
taken, not derivatives.

**The bound.** `max|f_3 − f_4| = tanh(|Delta|/4) < 1`, with `Delta` evaluated on
the **post-step** operator.

Key values, all reproducible:

| quantity | value |
|---|---|
| benchmark | `[23,5]`, Te = 2.947 eV, ne = 1.389e14 cm^-3 |
| tau_slow / tau_relax at benchmark | 22.73 us / 2.277 ns, M = 9982 |
| grid | 50 Te (1-10 eV) x 8 ne (1e12-1e15), `dlnTe = ln(10)/49 = 0.0469915` |
| two-channel superposition residual | 3.075e-14 |
| benchmark eps_plateau | 0.063612 |
| census, Te >= 2 eV, 100 us | 45 of 448, worst 17.5% |
| census at ITER-extrapolated 506 us | 24 of 448, worst 15.3% |
| \|Sbar\| range | 0.0649 to 0.4822 |
| \|G\| range | 2.64 to 14.52, negative everywhere |

**Sign convention (settled, do not re-litigate).** `Dln u = ln(u+/u-)`,
post-step over pre-step. Heating: `Dln u < 0`, `Sbar < 0`, `G < 0`, product
positive. Cooling: `Dln u > 0`, `Sbar < 0`, `G < 0`, product negative. `eps` is
positive in both because the equation takes a modulus. Verified on all 2288 rows.

---

## 4. What was done in this work

**Structure.** Six figures inserted with generated captions; abstract,
certificate, declaration, acknowledgements and nomenclature written;
`tau_QSS` renamed `tau_slow` at 102 sites (QSS names the *fast* elimination, so
using it for the slow clock made "QSS holds because tau_QSS is long" a sentence
about two different subsystems).

**Eight new verification scripts**, all in `src/validation/`:
`verify_transport_selection.py`, `diagnose_gate_d.py`, `verify_joint_step_map.py`,
`verify_crest_subgrid.py`, `verify_2s_not_slow.py`,
`verify_cold_corner_pressure.py`, `verify_fault_injection.py`,
`verify_emissivity_generalisation.py`.

**Findings that changed the thesis:**

- **Transport selection effect.** Not one of the 45 defended breakdown pairs
  satisfies `tau_esc > tau_slow`, at 10 or 20 cm, while 27-36% of the other 635
  do. Structural: a pair enters the census because `tau_slow` is long, and a long
  `tau_slow` is exactly when transport sets `n_1s`. Chapter 6 had tested
  transport only at `[0,4]`, below the scope boundary, the one place it survives.
- **Gate D's failure is in the gate.** It mixes both supply channels against a
  table that separates them. Rebuilt on the ground-fed channel: 323 of 400 agree.
  **ACD, which the gate loads and never reads, agrees at 400 of 400**, median
  0.943. Strongest external check in the thesis, available all along.
- **ELM duration.** `tau_drive = 100 us` was unsourced and is the aggressive end.
  Loarte 2003 Sec. 5 gives 506 us for the ITER pedestal. The census nearly halves;
  the worst case barely moves.
- **Lomanowski 2015 reframed the thesis.** No published procedure eliminates
  `n_1s/n_i`; careful practice takes the cancellation route at high n. The thesis
  is now "the cost of the closure where the cancellation is unavailable", not a
  demonstration that Balmer diagnostics fail. This is in the abstract.
- **Emissivity generalisation.** The derivation holds for any non-negative linear
  functional of the fast-state vector, so it covers the real A-weighted
  `Halpha/Hbeta` ratio and not just the shell ratio.

---

## 5. Review history

**Professor review** (earlier): reframing demanded and now done. The remaining
unmet item was "show me the diagnostic that eliminates `n_1s/n_i`", closed by
reading Lomanowski 2015 and reframing.

**Round 1 (mathematical spine).** Verdict FAIL, core survives. All seven repairs
applied in `7e3ef17`. The headline "sign error" was real but misdiagnosed: the
root cause was that `u-` and `u+` were **never defined anywhere**.

**Round 2.** Reviewed by a subagent; I never saw the review text itself. Four
items verified and fixed in `48ac151`: the ELM bound demoted (it printed `>=`
while the thesis's own data has a ratio of 0.9999, a counterexample); the plateau
window sweep cited (it was on disk, uncited, and favourable: `k` sets census
membership, never magnitude); "M carries no information" withdrawn (contradicted
by a table 60 lines below); four caption sites saying "the quasi-steady-state
closure fails", the reverse of the central finding.

**Maths audit of the new work** (`dde2b9c`) found seven defects in same-day
edits, two changing numbers: the grid interval printed as `0.046984` instead of
`0.0469915`, and `|SbarG|` secants presented as the zero-step limit. Also
`eq:eps_plateau_def` defined `R_PE` with `b_1^-` where it must be `u^-`, a factor
of 24 if read literally.

---

## 6. What is left

Read `outputs/REMAINING.md` for the full register. The substantive items:

1. **The emissivity-ratio reformulation is done analytically but not carried
   through the numerics.** `make_ch3_figures.py:252-253` and
   `make_ch5_figures.py:364-368` still use unweighted shell sums. Substituting
   `w·a` and `w·c` there is a two-line change at each site; weights come from
   `Balmer_transient_ratio.load_radiative_weights`. Do **not** use
   `Halpha_sensitivity.py:estimate_halpha_weights_from_L`, which self-documents
   as unsuitable.
2. **Three appendices are empty stubs**: state ordering, atomic data sources,
   numerical methods and convergence.
3. **A5 needs an independent re-run.** Demoted from verified because the May run
   was never repeated.
4. **The crest's sensitivity to n_max** (the one remaining `[UNVERIFIED]`, at
   `chapter5.tex:1242`) needs the rate pipeline rerun at n_max = 12 and 20.
5. **Eight older Ch3/Ch5 figures fail a colour-blind check** (`#2e7d32` against
   `#c0392b`, dE 4.2 under deuteranopia).
6. **Round 2's remaining items**, which I could not verify without the review
   text: a grid-wide bridge test (~680 propagations, genuinely expensive) and the
   finite-exposure observable.
7. Four `\todo`s are author-supplied citations or named-as-impossible work
   (SOLPS-coupled calculation, molecular matrix, the separatrix quantity, the
   n=15 ground-fed provenance).

---

## 7. Working conventions that matter here

- **`CLAUDE.md` rules are real.** Report, do not repair. No hardcoded grids. Fail
  loudly. Provenance on every number. Ground truth is physics > math > code >
  documents. A failing check is information.
- **Every number in the thesis needs a producing script and a stamped artifact
  under `validation/`.** "Recorded in a markdown file on a date" is a provenance
  defect even when the value is right.
- **Check before asserting.** Several times in this work a plausible sentence was
  wrong and the data caught it: the cooling sign case, the "near-step-independent"
  product, the grid interval. Write the check, then the sentence.
- **The user edits files in parallel.** Commit with an explicit file list, never
  `git add -A`, or their work gets swept into your commit message. This happened
  once (`71691bf`).
- **Skills to use:** `scientific-prose-audit` as a *final* pass only, after the
  physics is frozen; `thesis-verification` for establishing numbers;
  `thesis-writing` for producing prose. Subagents: `math-auditor` earns its keep
  and found seven real defects; `Explore` is good for exhaustive inventories.
- **Prose rules:** zero em dashes, no filler transitions, claim strength matched
  to evidence, physical quantities named. The mechanical scan is in the
  `scientific-prose-audit` skill.
