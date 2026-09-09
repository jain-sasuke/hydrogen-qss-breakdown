# Writing Specification

**Written 10 September 2026.** Binding on every chapter. Extends — does not
replace — `thesis_architecture.md` PART 3, which fixes the voice. This document
fixes the **reader**, the **story**, and the **first-principles standard**, and
makes all three testable.

---

## 1. The reader, named precisely

Write for **a final-year B.Tech or first-year M.Tech student in Chemical
Engineering who has never opened a plasma physics book.**

**What you may assume they know**, without explanation:

- Ordinary and partial differential equations; solving $\dot{\bm y} = \bm A\bm y$
  by eigen-decomposition; what an eigenvalue and eigenvector are.
- Matrix algebra: inverse, rank, block structure, similarity.
- Chemical reaction engineering: rate constants, reaction networks, mass-action
  kinetics, the CSTR and PFR, residence time.
- **The steady-state approximation for reactive intermediates** (Bodenstein).
  This is the single most valuable thing they already know, because it *is* the
  quasi-steady-state approximation under another name. Use it relentlessly.
- Thermodynamic equilibrium, detailed balance, the Boltzmann factor.
- Dimensional analysis and order-of-magnitude estimation.
- Basic statistics and curve fitting.

**What you must NOT assume**, and must build from first principles at first use:

- Any atomic structure beyond "an electron orbits a proton". Principal quantum
  number $n$, orbital angular momentum $\ell$, statistical weight $g$, and the
  spectroscopic labels 1S/2P/3D must all be introduced.
- What a spectral line *is*, and why its intensity carries information.
- Einstein $A$ coefficients, oscillator strengths, selection rules.
- Ionisation, recombination (radiative and three-body), and why three-body
  recombination scales as $n_e^2$.
- Saha–Boltzmann equilibrium and the departure coefficient $b_1$.
- What a tokamak divertor is, what an ELM is, and what detachment is.
- Optical thickness, radiation trapping, escape factors.
- Any acronym: CR, QSS, CRE, ADAS, SCD/ACD, LTE, MAR, PSM20, CCC, SOL.
  **Every acronym is expanded at first use in every chapter.**

**The test, applied per section:** hand it to someone with exactly this
background. If they cannot follow it unaided, what is missing is **intuition**,
not detail. Add the picture, not another equation.

---

## 2. The story — one arc, seven chapters

The thesis is not seven reports. It is one argument with a **reversal** at its
centre. Every chapter must know which beat it carries.

> **Setup.** A spectroscopist points a spectrometer at a divertor, measures the
> ratio of two hydrogen lines, and looks the answer up in a table indexed by
> temperature and density. That is how the plasma gets measured. **(Ch 1)**
>
> **Complication.** The table was computed assuming the plasma sits in
> equilibrium ionisation balance. A divertor is violently unsteady — ELMs,
> detachment, transients on microseconds. So: does the table still hold?
> **(Ch 1 → 2)**
>
> **The obvious suspect.** Everyone worries the same thing: the excited states
> cannot possibly keep up with a plasma changing that fast. To test it we build
> a 43-state time-dependent model. **(Ch 2)**
>
> **The reversal — the heart of the thesis.** They keep up. The
> quasi-steady-state closure is exact to one part in $10^8$, even at the worst
> point on the grid. *The thing everyone worries about is not the problem.*
> **(Ch 3)**
>
> **The real culprit.** What does not keep up is the **ground state**, which
> holds essentially all the atoms and turns over ten thousand times more slowly.
> The table assumes the ground-state reservoir has already reached its new
> equilibrium value. It has not. **(Ch 3)**
>
> **The mechanism.** The line ratio inherits that staleness through exactly two
> supply channels — atoms fed down from the ground state, and atoms fed up from
> the ion continuum. Two shells fed in different proportions respond
> differently, and the difference is the error. It has a closed form.
> **(Ch 3)**
>
> **Does the model deserve belief?** Gates, external benchmarks, two corrections
> found by audit, and two attacks that were survived. **(Ch 4)**
>
> **What it says.** A map of the error, an interior maximum in density, and the
> uncomfortable finding that timescale separation does not predict where the
> diagnostic fails. **(Ch 5)**
>
> **What it cannot say.** Optical thickness, transport, molecules — quantified,
> not hedged. **(Ch 6)**
>
> **What it means.** For the spectroscopist holding the spectrometer. **(Ch 7)**

**Every chapter opens by restating where the reader is in this arc** — two or
three sentences, no formalism — **and closes by handing off to the next.**

---

## 3. First principles — the operational rules

"From first principles" means a specific, checkable thing:

**R1. Nothing appears without being built.** Every symbol, every concept, every
equation is either (a) in the assumed-knowledge list of §1, (b) constructed
earlier in the thesis with a cross-reference, or (c) built where it appears.
There is no fourth category. "It is well known that" is banned.

**R2. Physical picture before formalism, always.** The words come first, then
the symbols, then the equation. Never the reverse order.

**R3. Name it before you formalise it.** "Two clocks" before $\taurel$ and
$\tauqss$. "Fed from above and below" before $\nzero$ and $\none$. "How stale
the reservoir is" before $\ln x$. The name is what the reader remembers; the
symbol is what they look up.

**R4. Every equation gets three things** — what each symbol means, the units of
both sides, and one sentence saying what it *does* physically. A displayed
equation with no following sentence of interpretation is a defect.

**R5. Dimensional check on every important result.** State it. It costs one
line and it is the cheapest error-detector in the thesis.

**R6. Every non-trivial derivation ends in a worked number.** Symbolic work
hides unit errors and inverted ratios; arithmetic exposes them. Carry the
benchmark point ($\Te = \benchTe$, $\nel = \benchne$) through the whole thesis
as the running example, so the reader accumulates familiarity with one place in
parameter space rather than meeting a new one each chapter.

**R7. Say what you expected before you say what happened.** "The obvious guess
is $X$. It is wrong, and the reason is interesting." A number with no prior
expectation can only be rationalised, never falsified — and reads as a data
dump.

**R8. Numbers arrive with meaning.** Never "$\taurel = 2.28$ ns" alone. Write
"2.28 nanoseconds — roughly a hundred thousand times shorter than the ELM that
disturbs it." Every quantity is anchored to something the reader can picture.

**R9. One idea per paragraph.** If a paragraph carries two, it carries none.

**R10. Keep the mistakes in.** The ℓ-mixing $F(U_m)$ error, the eigenvalue
filter, the misnamed central quantity. A thesis that found and bounded three of
its own errors is more credible than one reporting none. These go in the body
as sections, not in an appendix as an apology.

---

## 4. Chemical-engineering bridges

These are **genuine correspondences**, not decorations. Use them; they are the
reader's fastest route in.

| Plasma object | ChemE object the reader already owns |
|---|---|
| CR rate matrix $\Lmat$ | rate-constant matrix of a linear reaction network |
| Excited states | reactive intermediates (radicals) |
| Ground state $\ngs$ | the bulk reactant — the reservoir holding all the mass |
| **QSS approximation** | **Bodenstein steady-state approximation.** The same approximation, the same justification, the same failure mode |
| $\taurel$ vs $\tauqss$ | fast intermediate turnover vs slow reactant consumption |
| $M = \tauqss/\taurel$ | the stiffness ratio that licenses the pseudo-steady state |
| Schur complement $\Omqss$ | eliminating intermediates to get an effective overall rate |
| Recombination source $\Svec$ | a constant feed stream into a CSTR |
| Ionisation | irreversible consumption to product |
| Two supply channels $\nzero$, $\none$ | two feed streams into the same reactor, mixed in shell-dependent proportion |
| Detailed balance | thermodynamic consistency of forward/reverse rate constants |
| Optical thickness / trapping | product re-adsorption; a reaction slowed because the product cannot leave |
| Line ratio $\Robs$ | a composition measurement used to infer reactor conditions |
| The lookup table | a calibration curve built under assumptions the plant then violates |

**The single most useful sentence in the thesis** is the one that says: *the
approximation being tested is the Bodenstein steady-state approximation, applied
to atomic level populations instead of radical concentrations.* Say it early,
in Chapter 1 or 2, in those words.

---

## 5. Section template

Every `\section` follows this shape, in this order:

1. `% QUESTION:` comment — the question in one sentence. If it cannot be
   written, the section has no reason to exist. (Rule already at
   `thesis_main.tex:19`.)
2. **The question, in prose**, to the reader.
3. **The physical picture** — words and, where it helps, a figure. No symbols
   yet.
4. **The formalism** — symbols defined at first use with units, then the
   derivation, nothing skipped as "standard".
5. **The worked number** at the benchmark point.
6. **The result with its caveats in the same paragraph** — never a footnote,
   never deferred to a later section.
7. **The bridge** — one or two sentences to what comes next.

---

## 6. Prose rules

- Short sentences. One subordinate clause is usually one too many.
- Active voice and the first person plural where a choice was made: "we chose
  $n_{\max} = 15$ because…" is honest; "$n_{\max} = 15$ was chosen" hides the
  agent and the reason.
- **Define, then abbreviate.** "collisional–radiative (CR)" once per chapter,
  then "CR".
- No sentence may contain two unexplained pieces of jargon.
- Tables and figures are referenced in the text and interpreted in the text. A
  float the prose does not discuss should be cut.
- **Every figure caption is self-contained** — a reader flipping through the
  thesis must be able to understand the figure from its caption alone.
- Consistent notation: **every symbol comes from the macro block in
  `thesis_main.tex` Part 2.** Never define notation inside a chapter. If a
  symbol is missing, it gets added to `thesis_main.tex`, not invented locally.

---

## 7. Numbers, scope, and honesty

**N1. Provenance.** Every number traces to a named script, file, and grid point.
Where the text cannot carry that, a table or the caption does.

**N2. Every number carries its scope.** This project's characteristic failure
mode is a regime-restricted number quoted globally — it has now happened twice
(the $\tauqss$ floor, quoted over 400 points when it is the minimum over 346;
the Ly-α insensitivity, true at the benchmark and false by 27× at the cold
corner). Before writing any extremum, state the set it is an extremum *over*.

**N3. Nothing enters a chapter that is not ▶️ or ✅ in the register.**

**N4. A failing check is reported, never repaired.** Gate D fails; say so and
diagnose it. The $r_1$ deficit is open; say so.

**N5. Distinguish what was measured from what was assumed** in every sentence
that could be read either way.

---

## 8. The acceptance checklist

Apply to every section before considering it done:

- [ ] The `% QUESTION:` comment is present and answerable in one sentence.
- [ ] A reader with §1's background could follow it unaided.
- [ ] Every symbol was defined at first use, with units, from the macro block.
- [ ] Every acronym was expanded at first use in this chapter.
- [ ] The physical picture precedes the formalism.
- [ ] There is at least one worked number, at the benchmark point where possible.
- [ ] Every displayed equation is followed by a sentence saying what it does.
- [ ] Every quoted number carries its provenance and its scope.
- [ ] Expectations were stated before results.
- [ ] Caveats sit in the same paragraph as the claim.
- [ ] The section bridges to what follows.
- [ ] No paragraph carries two ideas.
