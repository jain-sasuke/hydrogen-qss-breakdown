# Operating Protocol — Verification Sweep to Defense

**Written 22 August 2026**, after a session in which the chat model made twelve
errors and the student caught most of them. The rules below are derived from
those specific failures, not from general principle. Baseline commit `1e810c7`.

---

## PART A — The failure ledger

Every rule in Part C exists because of one of these. Recorded per method 7
(corrections in place, never silent).

### A.1 Root cause 1 — reading numbers from an unverified file

Three retractions, one cause: values were quoted out of
`qss_analysis_summary.txt` without checking them against
`timescale_verification.csv`.

| # | Claim made | Truth |
|---|---|---|
| R1 | "C11/A7 mislabel τ; τ_relax at Te=1, ne=1e12 is 26.5 ns" | 38.879 ns. The note was right |
| R2 | "C7 is obsolete, no point has τ_QSS > 1 s" | 19 points, max 67.2 s. C7 was right |
| R3 | "There is a breakdown corner at M ≈ 1.5" | M = 1.73×10⁹ — the largest on the grid |

The summary file was the *corrupted* one. Both files sat in `validation/` with no
provenance header and no indication which superseded which.

### A.2 Root cause 2 — partial search reported as absence

| # | Claim | Truth |
|---|---|---|
| — | "max M = 1.01197289e8 appears nowhere in `timescale_verification.csv`" | Present at Te=1.048, ne=1.93e13, nine significant figures |
| — | "A7's numbers exist nowhere in the repo" | `verify_ramp_vs_step.py` line 222 `test_full_system` does exactly that integration |
| — | "the 19 affected points are all at ne = 1e12" | They extend to ne = 1.93×10¹³ |
| — | "the code is five weeks ahead of the notes" | Four months *behind* — the MZ work is April |

In each case a subset was searched and the conclusion was stated as if the whole
had been.

### A.3 Root cause 3 — platform assumption

`find -printf`, `ls --time-style`, unquoted `--include=*.py`. Three wasted
round-trips. The machine is macOS/BSD with zsh; GNU coreutils flags do not exist
and zsh does not pass unmatched globs through.

### A.4 Root cause 4 — predicting from documents instead of from mechanism

| Prediction | Outcome |
|---|---|
| "`validate_gates.py` is the unfiltered writer" | Byte-identical filter. Both writers corrupt |
| "`cr_matrix/S_grid.npy` is the ionisation array" | It is the recombination source |

Both predictions came from reading notes. The severe test (Te-dependence,
`grep` for the write) settled each in one command. **A prediction is still
valuable when wrong** — it converts the run into a test. The failure is not
predicting wrongly; it is predicting from the wrong source.

### A.5 What the student caught

R1–R3, the A7 code path, the two-writers problem being worth chasing, and the
insistence on the repo audit before replanning — which is what exposed that four
months of results predate the correction. **The ground-truth hierarchy worked.**

---

## PART B — Roles

| Actor | Produces | Never does |
|---|---|---|
| **`verifier`** (Sonnet, local) | Evidence. Runs scripts, reports with provenance | Repairs, substitutes, adjusts tolerances, "fixes while in there" |
| **Chat (Claude)** | Interpretation, derivation partner, adversary | Produce evidence. Every number computed in chat is arithmetic on pasted data and is suspect |
| **`skeptic`** (Opus, local) | ACCEPT / MODIFY / REJECT | Return "looks good" without listing what it attacked |
| **`referee`** (Opus, local) | Publication judgement | Run before the claim's central parameter is defined |
| **ChatGPT** | Independent read of raw output | See Claude's interpretation first |
| **Student** | Derivations, decisions, arbitration | Accept a number that has no provenance |

**The asymmetry that matters:** the verifier touched your files; it either ran or
it did not. Neither chat model can fake that. Chat and ChatGPT produce
*interpretations*, which are mutually checkable — but their **agreement is weak
evidence** (shared training bias) and their **disagreement is the signal**.

---

## PART C — The task cycle

Every task, without exception, runs these seven steps.

### 1. Scope — one task, one question

If the prompt has two questions in it, split it. The S_grid task and the filter
task were entangled for two exchanges and cost a round-trip.

### 2. Predict — written before the prompt is sent

State the expected answer **and** the observation that would refute it. A number
with no prior expectation can only be rationalised, never falsified.

Predict from **mechanism**, not from notes (A.4). "The filter promotes the ladder
by one, so `tau_relax_grid` is contaminated too" is a mechanism prediction; "the
notes say validate_gates is clean" is not.

### 3. Prompt — written to the verifier, not to a shell

Every verifier prompt states:
- **macOS/BSD, zsh.** No `find -printf`, no `ls --time-style`, `sed -i ''`,
  quote all globs. (A.3)
- **Report only.** No repairs, no substitutions, no tolerance changes.
- **Stop conditions** — what makes it halt rather than work around.
- **Provenance required** — files loaded (full paths), working dir, and the
  SHA-256 of `L_grid.npy` (expect
  `2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`).
- **What would contradict the expectation**, explicitly, with instruction to
  report whether each occurred.
- **Back up before overwriting.** Suffix `_FILTERED_20260721`, never delete.

### 4. Run — verifier only

Chat does not run anything. If chat writes bash, that is a fallback, not the
path.

### 5. Cross-check — raw output, cold, to both models

Paste the verifier's raw output to ChatGPT **before** it has seen any
interpretation. Ask the falsification question, not the summary question:
*"what would this look like if X were false?"*

### 6. Resolve — the student arbitrates

Disagreement between the two models locates real uncertainty. Agreement means
nothing on its own.

### 7. Record — before the next task starts

Update the backlog entry and its graduation state. An unwritten result is one
you will re-derive in three weeks.

---

## PART D — Rules that would have prevented each failure

**D.1 No number without provenance.** Before quoting any value, name the file,
its mtime, the script that wrote it, and whether that script postdates the
current `L_grid`. (A.1)

**D.2 Every output file gets a provenance header** — script, date, source-matrix
SHA — from now on. `qss_analysis_summary.txt` and `timescale_verification.csv`
disagreeing by nine orders of magnitude with nothing to distinguish them is what
caused R1–R3. (A.1)

**D.3 Absence claims require a stated search scope.** "Not in the file" is only
admissible as "grep'd all 400 rows of column 5 with tolerance 1e-6, absent."
Otherwise say "I did not find it in X." (A.2)

**D.4 One writer per output path.** `qss_analysis.py` and `validate_gates.py`
both write `M_grid.npy`, `tau_QSS_grid.npy`, `tau_relax_grid.npy`. Whichever ran
last owns them. **Decision required:** `validate_gates.py` stops writing these
three and reads them instead.

**D.5 Grep before believing a docstring.** `assemble_cr_matrix.py` labels
`S_grid` as `[cm^3 s^-1]`; line 240 passes `n_ion=1.0`, so it is **s⁻¹**.
Verified numerically: `S = ne(α_RR + ne·α_3BR)` to all digits. Read the source,
not the comment.

**D.6 Fix all instances, not the one you found.** The filter was in two files.

**D.7 Back up before regenerating.** The filtered outputs are evidence for the
corrections chapter.

---

## PART E — Task queue

Dependencies are real; the order is not arbitrary.

### Blocking chain (must be serial)

| # | Task | Blocks | Status |
|---|---|---|---|
| **V0** | Commit baseline | everything | ✅ `1e810c7` |
| **V1** | S_grid collision | V3 | ✅ recombination source, s⁻¹, `S = ne(α_RR + ne·α_3BR)` |
| **V4** | Filter characterised | V5, regeneration | ✅ 19 points, ladder shift confirmed, `n_short`=0, `n_cplx`=0 |
| **V4b** | **Apply fix (both files), regenerate `qss_analysis`** | everything downstream | ⬅ **RESUME HERE** |
| **V2** | `verify_ramp_vs_step.py` audit + arg parser | Q8 | ⏳ code path found at line 222; needs the invocation read before running |
| **V3** | Slow eigenvector excited components | Q8 | ⏳ |
| **Q8** | Which τ enters De | the title, Ch. 3, Ch. 5 | ⏳ |
| **Title** | Decision gate | all writing | ⏳ |

### Parallel (no dependency on Q8)

- **V6** figure staleness table
- **V7** `chapter4.tex` number audit
- **Pala conversation** — do not wait for Q8; the status report stands on its own
- **Submission deadline** — still unconfirmed, still blocking the schedule

### Not in the sweep

Mori-Zwanzig, S-criterion, Balmer families, trapping. All April, all
pre-correction. Paper, not thesis.

---

## PART F — Stop conditions

The verifier halts and reports rather than proceeding if:

1. A precondition fails (e.g. `max Re(λ) ≥ 0` before the filter replacement).
2. An input file is missing. **Never substitute a stand-in** — this rule already
   caught one bad path: `verify_ramp_vs_step.py` line 87 stops rather than guess.
3. A script writes files not listed in the expected set.
4. A result contradicts a stated expectation. Report the contradiction; do not
   reconcile it.

Chat escalates to the student rather than deciding when:

- Two files disagree and neither has provenance.
- A fix would touch more than the identified defect.
- A claim's truth depends on a definitional choice not yet made (Q8 is exactly
  this — do not let any downstream result be quoted while it is open).

---

## PART G — Resume point

**Superseded.** The August resume point is closed: the eigenvalue filter was
repaired in `qss_analysis.py` on 23 Aug and in `solve_cr.py` / `check_mz.py` on
16 Sep, `operating_protocol.md` is in the repo, and the filter's effect is
reported in Chapters 2 and 4. For current state read `outputs/HANDOFF.md`, which
is rewritten each session; this file holds only the method.

---

## PART H — The review-adjudication cycle

Four external reviews have now been worked through. The cycle that emerged is
different from Part C's task cycle, because a review is not a task: it is a set
of claims about the thesis, most of which are wrong in some respect.

**H.1 Identify the build the reviewer read.** Every review so far was written
against a stale PDF. Round 4 was written against commit `65b05da`, 175 pages,
fourteen commits behind the tree. Find that commit, diff it against HEAD, and
anchor every "already fixed" to the diff rather than to a status file. Of Round
4's fifteen findings, two were wrong and three were already closed.

**H.2 Adjudicate before acting.** The reviewer is not an authority. Re-derive or
re-measure each claim. In Round 4 the single most valuable finding broke a thesis
claim the reviewer had not realised he was breaking, and two findings that looked
like defects were the thesis being right.

**H.3 Sort the findings into three piles.**
  - *requote* — an artifact exists and disagrees with the text. Cheapest and
    highest value; do these first.
  - *rescope* — the number is right and its set is not stated.
  - *open physics* — cannot be closed by writing. Convert to a scope statement.

**H.4 Never rewrite a claim on an agent's scratch computation.** A number that
exists only in a transcript has the same provenance defect as one that exists
only in a markdown note. If a correction changes a claim, write the stamped
script first. This was done for the Fujimoto bundle result
(`verify_fujimoto_bundle.py`) and it is the standard.

**H.5 Propagate.** A claim usually appears three or four times. The Fujimoto
l-closure explanation appeared in Chapter 4 twice, in Chapter 6 twice, and in
Table 4.4; fixing two of the five left the chapter contradicting itself. Grep for
the claim, not the sentence.

---

## PART I — Failures added since August

Same format as Part A. Four of these five were committed by the assistant in the
16 Sep session.

### I.1 A page count quoted from an unconverged build

`latexmk` can exit 0 with the page numbers not yet settled. "202 pages" was
reported through a whole session and committed, while the same build's `.toc`
placed section C.4 at printed page 210 — in a document measured at 202. Two
further passes give 218.

**Rule:** before quoting a page count, run `latexmk` twice and check the last
`.toc` entry's printed page against `pypdf`'s count. If they disagree, the build
has not converged.

### I.2 A script run with `--help` overwrote its own stamped artifact

Several scripts under `src/validation/` have no `argparse`. An unrecognised flag
is ignored and the script runs its full computation into its own output
directory. `validation/ion_closure/` and `validation/ng_scaling/` were
regenerated this way; both are untracked, so the previous bytes were
unrecoverable. Determinism was verified afterwards — two runs differ only in the
`generated` timestamp — so no numbers were lost, but five days of provenance
stamp were.

**Rule:** read artifacts under `validation/` directly. Do not run a script to
find out what its CLI is; `grep -c add_argument` first. Any new script gets
`--out` before it gets a computation.

### I.3 Code edited without updating the text that describes it

The `eigs[eigs < -1.0]` literal was removed from two files at the author's
instruction. Chapters 2 and 4 both asserted it was still present, so the thesis
became false on a matter of fact, caused by the repair.

**Rule:** grep the thesis for any file path or line number you are about to
change. Code and prose are one artifact here.

### I.4 An instruction was given to an agent and treated as protection

A subagent was told not to overwrite stamped artifacts. It did anyway, by a route
neither of us anticipated (I.2). The instruction was necessary and not
sufficient.

**Rule:** agents that only read get read-only tasks. If a task could write,
scope the writable directory explicitly or run it yourself.

### I.5 A zsh glob with no match aborted the line

`rm -f *.aux *.log *.synctex.gz` — the last pattern matched nothing, zsh failed
the whole command, and every build product was left in place and shipped inside
an archive. This is the trap `HANDOFF.md` §2 already warns about, hit anyway.

**Rule:** `find . -name '*.aux' -delete`, never a bare glob list. Quote every
glob. Verify the archive contents after building it, not before.

---

## PART J — The build and release path

1. Build twice: `latexmk -pdf -cd -interaction=nonstopmode thesis_tex/thesis_main.tex`.
   The `-cd` matters; `\input{../figures/...}` needs it.
2. Check: `grep -c '^! '` on the `.log` for errors, `grep -c undefined` for
   references and citations, `grep -ci 'warning--'` on the `.blg`.
3. Confirm convergence per I.1 before quoting a page count.
4. Commit with an explicit file list, never `git add -A`; the author edits in
   parallel and a parallel workstream was left unstaged on 16 Sep for exactly
   this reason. Message via heredoc and `git commit -F -`.
5. For Overleaf, flatten: `.tex` and `.bib` at the archive root, `figures/`
   beside them, and rewrite `\input{../figures/...}` to `\input{figures/...}`
   because Overleaf compiles from the project root with no `-cd`. Include only
   the figures actually referenced. Test-compile the flattened tree before
   sending it.
