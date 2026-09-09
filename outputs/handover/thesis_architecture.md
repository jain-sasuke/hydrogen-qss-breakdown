# Thesis Architecture — The Story

**Written 23 August 2026**, after the review that the existing draft "reads like
a report, not a story," and after Findings 09 established what the work actually
measures.

**Defence ~21 September. ~29 days.**

---

## PART 1 — The story, in one page

Everything below is a way of telling this. If a paragraph doesn't advance it,
cut the paragraph.

> A physicist points an optical fibre at the ITER divertor and measures the
> brightness of two hydrogen lines: $H_\alpha$ and $H_\beta$. Their ratio is
> supposed to tell her the electron temperature and density. She looks the ratio
> up in a table.
>
> **The table assumes the plasma has stopped changing.** Not that it is slow —
> that it has *finished*. Ionisation and recombination are assumed to have
> settled into balance.
>
> The divertor never stops changing. ELMs arrive every few milliseconds; the
> plasma detaches and reattaches. So: how wrong is the table, when?
>
> The textbook answer says the thing to worry about is whether the *excited
> states* keep up — they are the fast part, and the standard shortcut assumes
> they equilibrate instantly. This thesis builds a time-dependent
> collisional–radiative model of hydrogen to test exactly that.
>
> **The excited states are fine.** They equilibrate to eight significant
> figures. The shortcut everybody worries about is not the problem.
>
> **The ground state is the problem** — and the table assumes it too. After a
> temperature step the excited levels settle within nanoseconds into equilibrium
> with a ground-state population that has not moved, and will not move for
> microseconds to seconds. During that window the atoms are internally
> consistent and the *table* is wrong, by a median of 7% and by at least 39% in
> the cold, detached corner. Long enough to outlast an ELM.
>
> Why? Because excited levels are fed from two directions — up from the ground
> state, down from the ions — and levels $n=3$ and $n=4$ draw on those two
> supplies in different proportions. When the supply mix is stale, their ratio
> is wrong. The error is $|f_3 - f_4|\cdot|\Delta\ln n_{1s}|$, and it vanishes
> when either supply dominates completely.
>
> **It is largest exactly where the two supplies compete — which is detachment,
> the transition the diagnostic is used to characterise.**

**The reversal in the middle is the thesis.** You go looking for one thing,
prove it isn't broken, and find the broken thing next to it. Do not bury that —
Chapter 1 should promise it and Chapter 5 should deliver it.

---

## PART 2 — Title

**Recommended:**

> ## The Cost of Assuming Ionisation Balance
> ### Transient Error in Balmer-Ratio Diagnostics of Divertor Hydrogen Plasmas

It names the assumption tested, the observable, and the system. It is findable
by someone searching for exactly this problem. And it does not claim the thing
the work disproved.

**Alternatives, if the supervisor prefers:**

| | Title | Note |
|---|---|---|
| B | *Fast Atoms, Slow Plasma: Quantifying Ionisation-Balance Error in Time-Dependent Collisional–Radiative Modelling of Divertor Hydrogen* | Leads with the physical picture; longer |
| C | *When Does the Balmer Ratio Lie? A Two-Channel Criterion for Collisional–Radiative Diagnostics of Divertor Hydrogen* | Leads with the question; the criterion is the contribution |

**Do not use** anything containing *"quasi-steady-state breakdown."* The work
shows the QSS closure is exact to $10^{-8}$ in the regime tested. A title naming
a breakdown that does not occur will be the first thing an examiner attacks.

**This is a supervisor conversation**, not a unilateral change — the registered
title is different.

---

## PART 3 — Voice

The supervisor's instruction: *a first-year master's student should follow it
top to bottom.* That is a specific, testable standard, not a style preference.

**Per chapter, in this order:**

1. **The physical question.** Why anyone should care, before any formalism.
2. **Every symbol defined at first use, with units.** No exceptions.
3. **Intuition.** ChemE analogies where they are genuine correspondences —
   reaction networks, CSTR feed streams, the Bodenstein steady-state
   approximation, transport. These are real here, not decoration.
4. **Mathematical rigour.** The derivation, complete, nothing skipped as
   "standard."
5. **The result, with its caveats attached** — in the same paragraph, not in a
   footnote or a later section.
6. **Why it matters, and the bridge to the next chapter.**

**Six rules that turn a report into a story:**

- **Every section answers a question.** If you cannot write the section's
  question in one sentence, the section has no reason to exist.
- **Say what you expected before you say what happened.** "The obvious guess is
  X. It is wrong, and the reason is interesting." A result with no prior
  expectation reads as a data dump.
- **Name the thing before you formalise it.** "Two clocks" before
  $\tau_{\rm relax}$ and $\tau_{\rm QSS}$. "Fed from above and below" before
  $r_0$ and $r_1$.
- **One idea per paragraph.** If a paragraph has two, it has none.
- **Numbers arrive with meaning.** Not "$\tau_{\rm relax} = 2.28$ ns" but
  "2.28 nanoseconds — a hundred thousand times shorter than the ELM that
  disturbs it."
- **Keep the mistakes in.** The ℓ-mixing bug, the eigenvalue filter, the
  misnamed metric. A thesis that found and fixed three of its own errors is more
  credible than one that reports none. Put them in Chapter 4 as a section, not
  in an appendix as an apology.

**The test:** hand a section to someone with your own starting background —
ChemE, no plasma physics. If they cannot follow it unaided, what is missing is
*intuition*, not detail.

---

## PART 4 — Chapter spine

Each chapter below has: the question it answers, the story beat it carries, and
what it is built from.

### Chapter 1 — Reading the Light from a Divertor

**Question:** What does a spectroscopist actually measure, and what does turning
that measurement into a temperature assume?

**Beat:** Set up the problem and promise the reversal. End Chapter 1 with the
question the thesis answers, and hint that the answer is not the expected one.

**Contents:** the divertor and why it matters for ITER · what a spectrometer
records and what it does not (absolute intensity needs path length and neutral
density; a ratio needs neither) · the inversion chain: $I_{H\alpha}/I_{H\beta}
\to$ table $\to (T_e, n_e)$ · the assumption hidden in the table · why the
divertor is never in equilibrium · the question, stated plainly.

**From:** literature. This chapter needs real citations for the ITER divertor
parameter range — the current draft asserts $T_e = 1$–5 eV, $n_e = 10^{13}$–
$10^{15}$ with no source, and that must be fixed or removed.

### Chapter 2 — The Atoms

**Question:** What happens to a hydrogen atom in a plasma, and how well do we
know the rates?

**Beat:** Establish that the inputs are trustworthy, and be honest about where
they are not.

**Contents:** the 43-state space and why $\ell$ is resolved to $n=8$ · what a
cross section is and why it must be Maxwellian-averaged to become a rate
coefficient · electron-impact excitation (CCC), and why $\Delta n = 0$
transitions were excluded on the data provider's instruction · ionisation ·
radiative and three-body recombination · spontaneous emission · **proton-impact
$\ell$-mixing, and the structural argument for it**: $2s \to 1s$ is
E1-forbidden, $\Delta n = 0$ electron impact is excluded, so proton $\ell$-mixing
carries **99.92%** of the 2S loss rate and the $n=2$ manifold is unphysical
without it · the Anderson benchmark and the $n=5$ story.

**From:** derivations 01, 02, 04b, 08 · B1 investigation.

### Chapter 3 — Two Clocks

**Question:** If the rates are known, why is solving the problem hard — and what
shortcut does everyone take?

**Beat:** The conceptual core. This is where the thesis earns its keep and where
the two-reference-state distinction is established.

**Contents:** the master equation, built from a population balance on each level
(the ChemE reaction-network analogy is exact here) · the matrix form
$\dot{\mathbf n} = L\mathbf n + \mathbf b$, and why recombination is a feed
stream rather than a matrix element · conservation as a check on the assembly ·
**the two clocks**: excited states in nanoseconds, ionisation balance in
microseconds to seconds, and the eigenvalue structure that separates them · the
QSS shortcut as Bodenstein applied to the whole excited manifold at once ·
adiabatic elimination and the Schur complement · **the two reference states, and
why they are not the same** (Findings 09 §2) · Fujimoto's two-channel
decomposition, and what $f_3 - f_4$ means physically · the error measure, chosen
from a stated criterion rather than by preference.

**From:** derivations 01, 03, 04, 05, 07, 08 — these are already written in this
voice. **Chapter 3 is substantially assembly.**

### Chapter 4 — Does the Model Work?

**Question:** Why should anyone believe a number this model produces?

**Beat:** Credibility. Including the three errors found by the project's own
audit — this is where honesty becomes an asset.

**Contents:** detailed balance · coronal and Saha limits · timescale hierarchy ·
**the external benchmark against Fujimoto Table 4.1** ($r_0$ agreeing to 0.1–1%
for $n \ge 4$; $r_0(2) = 0.7392$ vs 0.730 at low density, the discriminating
case) and Appendix 4B (both timescales within a factor of three at a grid point
his example lands on) · Anderson RMPS, and the honest $n=5$ account · **§4.x
Errors found and corrected**: the $\ell$-mixing $F(U_m)$ error, the eigenvalue
filter that corrupted 19 of 400 grid points, and the misnamed metric · why Gate D
cannot pass, stated as a consequence of the open-system boundary rather than as
a failure.

**From:** derivations 02, 04b, 08 · findings 09 · verification ledger.

### Chapter 5 — What the Model Says

**Question:** How wrong is the table, and where?

**Beat:** The reversal, delivered. QSS is exact; the assumption underneath it is
not.

**Contents:** the two-stage relaxation, shown in time · **the QSS closure is
exact to $10^{-8}$ on the plateau** — state this before anything else, because it
frames everything · the CRE error: median 7%, at least 38.7% at the low-$T_e$
edge · the mechanism $|f_3-f_4|\cdot|\Delta\ln n_{1s}|$, predicting the measured
error to within a factor 1.35 · **the ridge**, and why it sits at the
ionizing–recombining crossover, which is detachment · why the error is not
averaged away over an ELM · why timescale separation does not predict it, and
why the apparent correlation is a temperature proxy.

**From:** derivations 05, 07 · findings 09 · the divertor map, plateau gridmap,
and Balmer regime figures.

### Chapter 6 — What This Model Cannot Say

**Question:** Where would you not trust this?

**Beat:** Bound the claim before an examiner does.

**Contents:** optically thin — and at the worst point $n(1s) \approx
1.5\times10^{15}$ cm⁻³, 15× denser than the thickest case tested, with a Ly-series
probe moving $\varepsilon$ by $-13\%$ to $-77\%$ · **no molecular channels**, in a
96.6%-neutral gas at 1 eV where MAR feeds $n=3$ directly · $n_{\max} = 15$, with
the terminal-shell excess **measured** ($r_1(15)$ high by 4.9–6.2× against
published values, while $n=10$ agrees to 4–20%) rather than assumed · open
system: the ion is a fixed reservoir, which is why Gate D cannot compare to ADAS
· Maxwellian electrons · $T_i = T_e$ in $\ell$-mixing · uniform plasma, no
transport · the low-$T_e$ grid edge, and why 1 eV is where the model must stop.

### Chapter 7 — What It Means

**Question:** What should a modeller or a diagnostician do differently?

**Beat:** Hand over something usable.

**Contents:** the criterion, in a form someone can apply to their own CR matrix ·
where in the divertor operating space to distrust a $(T_e, n_e)$ inversion ·
what a time-dependent inversion would require · the non-normality and
Mori–Zwanzig work as the natural continuation · what would settle the open
questions (the low-$p$ $r_1$ deficit; opacity; molecules).

---

## PART 5 — Writing order and schedule

**Write Chapter 3 first.** It is the conceptual core, it is substantially
assembly from derivations already in the right voice, and writing it will settle
the notation everything else uses. Then 5 (the result), 4 (the credibility), 2
(the inputs), then 6, 7, and 1 last — Chapter 1 is easiest to write once you know
exactly what it is promising.

| days | content |
|---|---|
| 1–2 | Notation pass: fix $\varepsilon^{\rm QSS}$ vs $\varepsilon^{\rm CRE}$, the six withdrawn numbers, the pre/post-step $M$ convention. **Everything else inherits from this** |
| 3–7 | **Ch. 3** |
| 8–12 | **Ch. 5** + its figures |
| 13–16 | **Ch. 4** + its figures |
| 17–20 | **Ch. 2** + its figures |
| 21–23 | **Ch. 6, 7, 1** |
| 24–26 | Full read-through, abstract, front matter |
| 27–29 | **Frozen document.** Mock defence |

**The figures are the schedule's real content**, not the physics. One of ~141 is
current. Regenerate them *as each chapter is written*, not in a batch at the end.

**Supervisor:** the title change needs his agreement, and the sooner that
conversation happens the less it costs.

---

## PART 6 — What must be settled before Chapter 3 is written

1. **Notation for the two references.** $\varepsilon^{\rm QSS}$ and
   $\varepsilon^{\rm CRE}$, or better names. Fixed once, used everywhere.
2. **The $\varepsilon$ zoo.** At least six incompatible definitions share the
   name `eps_step` across the repo. One primary measure, stated criterion,
   others named differently.
3. **Pre/post-step $M$.** Three values at the benchmark point across three files
   (9982, 8243, 4856), each individually correct, none distinguished.
4. **The observable.** $n_3/n_4$ shell ratio or A-weighted $H_\alpha/H_\beta$?
   They agree to 0.06%, so either is defensible — but say which, once.
5. **The benchmark point.** Grid [23,5], the point nearest $T_e = 3$ eV,
   $n_e = 10^{14}$ — an illustrative anchor, not an ITER operating point, and
   with no citation. Define it once in Chapter 3 and never call it an ITER
   reference.
