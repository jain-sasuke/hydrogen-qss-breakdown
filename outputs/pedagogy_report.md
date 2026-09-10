# Pedagogy report — GATE 8

**Agent:** pedagogy-agent. **Date:** 10 September 2026.
**Task:** read the thesis as the reader defined in `outputs/writing_specification.md` §1
(final-year B.Tech / first-year M.Tech Chemical Engineering, no plasma physics, owns
ODEs, eigen-decomposition, linear algebra, reaction kinetics, CSTR/PFR, and the
Bodenstein steady-state approximation) and report exactly where they get stuck.

**Read, in reading order:** `chapter1.tex` (847 ll.), `chapter2.tex` (1335),
`chapter3.tex` (1638), `chapter4.tex` (1478), `chapter5.tex` (1540),
`chapter6.tex` (1077), `chapter7.tex` (437), plus `thesis_main.tex`,
`thesis_main.toc`, `thesis_main.log`, `thesis_main.aux`.

**Caveat on line numbers.** The chapter files were being edited by another process
while I read. Between my first and last pass `chapter4.tex` grew from 860 to 1478
lines and four other files shifted by a few lines. Every line number below was
re-verified against the file state at the end of my read, and every stuck point is
also identified by its `\label` and by a quoted sentence, so the item survives
further drift.

**Overall.** This is unusually good expository writing for a technical thesis. The
reader's background is correctly judged almost everywhere: nothing is over-explained,
the ChemE bridges are load-bearing rather than decorative, and Chapters 1, 2, 4, 6
and 7 follow the section template of writing-spec §5 closely. The problems are
concentrated in two places: **the build state** (§0), and **Chapter 3** (§3), which
is the only chapter that abandons the section template and is exactly the chapter the
reader is most likely to abandon in return.

---

## 0. Two structural blockers that dominate everything else

These are not prose defects. They are the two places where the reader's experience
breaks completely, and they outrank every item below.

### 0.1 Chapters 4 and 7 are one page each in the built PDF

`thesis_main.tex` `\input`s chapters 1, 2, 3, 5, 6 and carries **inline section
skeletons** for Chapter 4 (lines 154–203) and Chapter 7 (lines 212–234). It never
`\input`s `chapter4.tex` or `chapter7.tex`, which exist and are 81 KB and 27 KB.

From `thesis_main.toc`:

```
4  Does the Model Work?        page 66     (next chapter starts p.67)
7  Conclusions                 page 108    (appendix A starts p.109)
```

The reader arrives at Chapter 4 after Chapter 3 has promised, seven separate times,
that validation is coming — and finds five empty headings. Then Chapter 5 opens
(`chapter5.tex:63`) with *"Chapter~\ref{ch:validation} established that the rate
matrix these questions are asked of reproduces independent atomic data"*, which the
reader has not seen. Chapter 6 §6.7 opens with *"Chapter~\ref{ch:validation}
presented the comparison against Fujimoto's published population coefficients as the
strongest external check this model passes"* — again, unseen. Then the thesis ends
on an empty Conclusions chapter.

- **Gap type:** structural, not intuition or detail.
- **Smallest fix:** replace the two skeleton blocks in `thesis_main.tex` with
  `\input{chapter4}` and `\input{chapter7}`. Both files already reproduce the
  skeleton's `\label`s (`chapter4.tex:9–11`, `chapter7.tex:19`), so no
  cross-reference breaks. `chapter7.tex` is titled *What It Means*; the skeleton is
  titled *Conclusions* — pick one.
- Note that both files' own headers say this needs doing (`chapter4.tex:7–12`,
  `chapter6.tex:10–15`), and `chapter5.tex:40–45` carries a **stale** version of the
  same note claiming `thesis_main.tex` still has a Chapter 5 skeleton. It does not;
  Chapter 5 is correctly `\input`. Delete that note.

### 0.2 Twenty-one drafting markers print in the body, in red

| file | `\todo{}` | `[UNVERIFIED:` | `[SOURCE REQUIRED:` | `\needcite` |
|---|---|---|---|---|
| chapter1 | 3 | – | – | – |
| chapter2 | 1 | – | – | – |
| chapter3 | 0 | – | – | – |
| chapter4 | 6 | 2 | – | 1 |
| chapter5 | 0 | 6 | 3 | – |
| chapter6 | 8 | – | – | – |
| chapter7 | 0 | 3 | – | – |

`\todo` renders red (`thesis_main.tex:46`); the bracketed markers are `\emph{}` and
render as italic prose indistinguishable from argument. Several sit mid-derivation —
`chapter5.tex:340–345` interrupts the sentence establishing the density of maximum
error; `chapter5.tex:611–617` interrupts the provenance of Table 5.5.

- **Gap type:** structural. The honesty is a project value and the *content* of these
  markers should survive to the examiner. The defect is that they are unmarked as
  scaffolding and are interleaved with the argument.
- **Smallest fix:** one macro, `\open{}`, rendering as a visually distinct
  block-level note (indented, ruled, not red), plus a list-of-open-items at the front
  matter. Costs nothing and turns 21 interruptions into a feature.

---

## 1. Chapter 1 — *Reading the Light from a Divertor*

Nearly clean. The reader is correctly placed, the ladder (question → picture →
formalism → number) is followed in every section, and the acronym discipline is
perfect: SOL (l.77), CR (l.359), LTE (l.368), ADAS (l.399), CRE (l.471), QSS (l.510),
ELM (l.547), MAR (l.713) — all expanded at first use.

**The Bodenstein sentence is present and correctly placed.** `chapter1.tex:514–520`,
inside the `description` item defining the QSS approximation:

> *"**The approximation being tested in this thesis is the Bodenstein steady-state
> approximation, applied to atomic level populations instead of radical
> concentrations.**"*

It is repeated at the very top of Chapter 2 (`chapter2.tex:31–33`), which is the
right place for it. Writing-spec §4's "single most useful sentence" requirement is
satisfied.

### 1.1 `chapter1.tex:425–431` — "exact" is asserted, then quietly withdrawn

> *"Because the population balances are linear in the populations, the mixture is
> exact rather than approximate."* (l.425–426), introducing
> $n_p = a_p \ngs + c_p \nion$.

Eighty-five lines later, `chapter1.tex:511–514` says the QSS approximation *"is what
makes $a_p$ and $c_p$ in Eq.~(1.5) functions of $(\Te,\nel)$ alone."*

A reader who follows algebra — and this one does — will stop and ask: if the split
was exact by linearity, what work is QSS doing? The resolution is that superposition
holds in the full time-dependent system too, but there $a_p$ and $c_p$ are
*functionals of history*, not functions of $(\Te,\nel)$. That is never said.

- **Missing prerequisite:** that "exact" here means exact *at steady state*.
- **Gap type:** intuition. The reader can do the algebra; they cannot see which of
  two true statements is being made.
- **Smallest fix:** one clause at l.426 — *"exact rather than approximate, once the
  excited levels have settled; §1.5 says what makes that qualifier necessary."*

### 1.2 `chapter1.tex:391` — Alcator C-Mod is not identified

> *"the Alcator C-Mod measurements of Pigarov, Terry and Lipschultz"*

TCV and MAST-U are both introduced as *"the TCV tokamak"* (l.482) and *"the TCV and
MAST-U tokamaks"* (l.715). C-Mod is not.

- **Gap type:** detail. Two words.
- **Smallest fix:** "the Alcator C-Mod tokamak".

### 1.3 Nothing else

$a_p$, $c_p$, $\ngs$, $\nion$, $A_{qp}$, $g$, $n$, $\ell$, $\Robs$ are all defined at
or before first use with units. Eq. (1.1) through (1.6) each carry a following
interpretation sentence. The Damköhler framing at l.613–620 is exactly the right
bridge and is not over-explained.

---

## 2. Chapter 2 — *The Atoms*

Also strong. Acronyms clean: CCC (l.397), TICS (l.499), RMPS (l.1003), ADAS (l.749),
E1 (l.185). The `% QUESTION:` comment is present on all 11 sections. Every worked
number carries its file and array index. The $2s$ argument in §2.5.3 is the best
piece of physical writing in the thesis.

### 2.1 `chapter2.tex:890` — $\mu = m_{\mathrm H}/m_e = 918.08$ contradicts l.833

> *"$\mu = m_{\mathrm H}/m_e = 918.08$"* (l.890)

against, fifty-six lines earlier:

> *"a proton, 1836 times heavier than an electron"* (l.833)

$m_{\mathrm H}/m_e = 1836.15$. The number 918.08 is the **reduced mass** of the
proton–hydrogen-atom pair in electron-mass units, $\approx m_p/2m_e$. The number is
right for PSM20; the symbol definition attached to it is wrong, and the reader has
been handed 1836 explicitly a page earlier so they *will* notice.

- **Missing prerequisite:** that $\mu$ in Eq. (2.13) is a reduced mass, not a mass
  ratio.
- **Gap type:** detail, but a credibility-costing one — this is the reader's first
  chance to catch the thesis in an arithmetic error, and it looks like one.
- **Smallest fix:** *"$\mu = m_{\mathrm H}m_p/[m_e(m_{\mathrm H}+m_p)] = 918.08$, the
  reduced mass of the colliding pair in units of the electron mass"*.

### 2.2 `chapter2.tex:893` — $D_{ji} \sim 6n^4$ drops a factor of $\ell_>$

> *"Note $D_{ji} \sim 6n^{4}$ for $\ell_> \ll n$, recovering the $n^{4}$ of
> Eq.~(2.12)"*

With $D_{ji} = 6n^2\ell_>(n^2-\ell_>^2)$, the small-$\ell_>$ limit is
$6n^4\ell_>$, not $6n^4$. The scaling claim survives; the expression as written does
not.

- **Gap type:** detail. A reader doing the substitution — which the sentence invites
  — gets a different answer.
- **Smallest fix:** $D_{ji} \sim 6n^{4}\ell_>$.

### 2.3 `chapter2.tex:51` — $I_n$ and Chapter 1's $E_n$ are never connected

Chapter 1 Eq. (1.1) gives $E_n = -13.6\,\mathrm{eV}/n^2$, negative, "measured from
the free-electron state". Chapter 2 Eq. (2.1) gives $I_n = I_{\mathrm H}/n^2$,
positive, "the energy needed to pull the electron free". Same formula, opposite sign,
different symbol, one chapter apart, and the identity $I_n = -E_n$ is never written.

- **Gap type:** intuition, low severity. The reader will work it out, but the
  hesitation is avoidable and this is the second thing they meet in Chapter 2.
- **Smallest fix:** half a sentence at l.56 — *"$I_n = -E_n$ of
  Eq.~\eqref{eq:bohr_levels}, with the sign convention reversed because a binding
  energy is naturally quoted positive."*

### 2.4 `chapter2.tex:875–877` — PSM20 is named but never unpacked

> *"the Debye-cutoff form of the Pengelly--Seaton treatment given by Badnell
> \emph{et al.} --- called PSM20 in the code and in this thesis"*

Writing-spec §1 lists PSM20 among the acronyms that must be expanded at first use in
every chapter. It is named-by-fiat rather than expanded, here and nowhere else in the
thesis. This is probably acceptable — it is a dataset label, not a phrase — but the
reader is told the letters mean something and never told what.

- **Gap type:** detail.
- **Smallest fix:** either say what the letters stand for, or say explicitly that
  PSM20 is a label for the dataset rather than an abbreviation.

### 2.5 A correction Chapter 4 makes that Chapter 2 does not carry

`chapter2.tex:922–924` states that freezing the Debye density in $U_m$ varies $F$ by
*"about \SI{30}{\percent} across the grid"*. `chapter4.tex:1155–1161` measures it and
reports *"a span of $\times3.7$ for $n=2$ and $\times50$ for $n=8$"*, calling the
Chapter 2 figure *"wrong by a wide margin"*. Chapter 4 carries its own `\todo` to fix
Chapter 2. Flagging here so the item is not lost: **a reader who reads both chapters
meets a number and its refutation, and the earlier one is unmarked.**

---

## 3. Chapter 3 — *Two Clocks* — the chapter that loses the reader

1638 lines, the mathematical core, and the only chapter that abandons the writing
spec's section template. This is where proportionate attention belongs and where
nearly all the remaining work is.

### 3.0 The systematic finding: Chapter 3 drops the template

| chapter | `\section`s | `% QUESTION:` comments |
|---|---|---|
| 1 | 9 | 9 |
| 2 | 11 | 11 |
| **3** | **7** | **0** |
| 4 | 13 | 13 |
| 5 | 11 | 11 |
| 6 | 10 | 10 |
| 7 | 5 | 5 |

Chapter 3 also carries none of the italic *"The question: …"* prose openers that
Chapter 1 puts under every section heading, and none of the *"The question, and the
picture behind it"* subsections that Chapter 6 uses.

This is not bookkeeping. The template exists because it forces steps 1 and 2 of the
ladder — physical question, governing principle — before the formalism. Chapter 3 is
the chapter where the reader most needs to be told *why we are computing this* before
they are shown *what we are computing*, and it is the only chapter that does not tell
them. Sections 3.1–3.4 survive on the strength of the prose. Section 3.5, which is
150 lines of departure coefficients, Saha normalisations, escape factors and logistic
algebra, does not.

- **Gap type:** intuition, systematic.
- **Smallest fix:** add the seven `% QUESTION:` comments, and add the one-sentence
  italic question opener to §3.3, §3.4 and §3.5. Writing-spec §5.1 says that if the
  question cannot be written the section has no reason to exist; all seven can be
  written, so this is cheap.

### 3.1 Acronyms: Chapter 3 expands none of them

Every acronym in Chapter 3 appears unexpanded, in a chapter the reader may open
weeks after Chapter 1:

| acronym | first use in ch3 | expanded in ch3? |
|---|---|---|
| CR | l.484 *"the CR equilibrium of $\Lmat^-$"* | no ("collisional--radiative" is spelled at l.51 but never paired) |
| QSS | l.699 (inside `\boxed{\nF^{\mathrm{QSS}}}`) | no ("quasi-steady-state" at l.617/646, never paired with "(QSS)") |
| ADAS | l.744 | no |
| CRE | l.1012 (`\bdep^{\rm CRE}`) | no |
| SCD | l.1402 | no |
| ACD | l.1405 | no |

Chapters 1, 4, 5, 6, 7 all expand QSS at first use; Chapter 3, which is *about* QSS,
does not. Chapter 6 does this exactly right with a dedicated three-abbreviation block
at `chapter6.tex:60–70`.

- **Gap type:** detail, but writing-spec §1 makes it non-negotiable.
- **Smallest fix:** copy Chapter 6's block into Chapter 3's opening "What this chapter
  takes as given" paragraph (l.20–30), covering CR, QSS, CRE, and expand ADAS/SCD/ACD
  at l.744 and l.1402.

### 3.2 `chapter3.tex:404–439` — the $\eta$ digression forward-references three undefined objects

**This is the worst single stuck point in the thesis.** The reader is in §3.3.3
("What the two modes are"), being shown that the slow mode is the ground state. Then:

> *"Its eigenvalue is closely related to, but not equal to, the direct ionisation rate
> of the ground state. Writing $|\lambda_0| = \eta\,Q_{1s}\,\nel$ …"* (l.404–407)

and immediately:

> *"Equation~\eqref{eq:reduced} shows exactly this — $\eta$ is the ratio of the
> effective ionisation rate to the direct one, $\eta = |\Omqss|/(Q_{1s}\nel)$"*
> (l.415–418)

> *"The denominator is emphatically \emph{not} $|\Lmat_{SS}|$ … Since the Schur
> complement returns part of that flux to the ground state, $|\Omqss| < |\Lmat_{SS}|$
> always"* (l.422–427)

> *"the two agree only to $O(\Msep^{-1})$"* (l.430–431)

At this point in the chapter:

- $\Omqss$ is undefined. It is defined at l.728, **324 lines later**.
- Eq. `eq:reduced` does not exist yet. It is at l.730.
- $\Lmat_{SS}$ is undefined. The block partition is Eq. (3.13) at l.677, **273 lines
  later**.
- "Schur complement" has not been introduced. It arrives at l.735.
- $\Msep$ is defined 15 lines *later*, at l.451.

Four undefined symbols and a forward-referenced equation, in a 35-line paragraph, in
the middle of the chapter's first substantive result. The reader cannot reconstruct
this; they can only accept it or stop. Most stop.

- **Missing prerequisite:** the entire QSS elimination of §3.4.
- **Gap type:** structural, presenting as intuition. The physics of $\eta$
  (stepwise ionisation is faster than direct) is well explained and belongs in the
  thesis; it is the *placement* that is wrong.
- **Smallest fix:** cut l.404–439 down to its physical content — three sentences:
  *"$|\lambda_0|$ is not the direct ground-state ionisation rate. It is larger, by a
  factor running from 1.16 to 29.5 across the grid and equal to 3.12 at the benchmark,
  because ground-state atoms are also excited into the manifold and ionised from
  there. §3.4 gives that factor a closed form."* Then move the algebra
  ($\eta = |\Omqss|/Q_{1s}\nel$, the $|\Lmat_{SS}|$ comparison, the $O(\Msep^{-1})$
  caveat) into §3.4.3, directly after Eq. (3.19), where every object in it exists.

### 3.3 `chapter3.tex:461–513` — "Which operator $\Msep$ belongs to" arrives 50 lines before the reader needs it

A 53-line `\subsubsection*` on notational hygiene, placed inside §3.3, whose stated
purpose (l.463–464) is:

> *"because it is the source of an apparent contradiction between this chapter and
> Chapter~\ref{ch:results}"*

The reader has not read Chapter 5 and has met no temperature step. They are asked to
absorb $\Msep^-$ vs $\Msep^+$, $\mathrm{d}\ln\Msep/\mathrm{d}\ln\Te \approx -4$, the
$10^{1/49}$ grid ratio, a four-interval step, and a *rejected alternative convention*,
all in service of a conflict they cannot see.

Writing-spec R7 ("say what you expected before you say what happened") and R2
("physical picture before formalism") are both violated: there is no expectation to
frame this, because the phenomenon it disambiguates has not happened yet.

- **Missing prerequisite:** the temperature-step experiment of §5.1.
- **Gap type:** intuition — specifically, motivation. The content is correct and worth
  keeping.
- **Smallest fix:** move the whole subsubsection to Chapter 5 §5.1.1, where
  `chapter5.tex:164–167` already writes *"Write $\Lmat^-$ and $\Lmat^+$ for the
  operators before and after, in the notation of Section~\ref{sec:two_clocks}"* — i.e.
  Chapter 5 is already pointing back at it and would host it naturally. Leave one
  sentence in §3.3: *"$\Msep$ is a property of a matrix, not of an experiment; §5.1.1
  says why that distinction has to be made explicit."*

### 3.4 §3.5.2 — the reader is never told that $\bdep\Zsb{1}\nel$ *is* Chapter 1's third argument

**This is the highest-value single addition available in Chapter 3.**

Chapter 1 §1.5 gives the reader the whole point of the thesis in a form they can hold
(`chapter1.tex:455–460`):

> *"dividing through by $\nion$ shows that the observable depends on the atomic
> coefficients … \emph{and} on the ratio $\ngs/\nion$, which is not. That ratio is a
> third, independent quantity."*

Chapter 3 §3.5.2 then defines $\bdep \equiv \ngs/(\Zsb{1}\nel\nion)$ (l.911) and works
in $\bdep$, $\Zsb{1}$, $\Zsb{1}\nel$, $\bdep\Zsb{1}\nel$ and
$x = \ln(\bdep\Zsb{1}\nel)$ for the next 400 lines. From Eq. (3.35),
$\bdep\Zsb{1}\nel = \ngs/\nion$ exactly — the combination that appears in every one of
those equations *is* Chapter 1's third argument. **Chapter 3 never says so.**

The reader therefore spends §3.5 and §3.6 manipulating a two-factor product whose
meaning they cannot see, having been given a perfectly clear physical picture of that
same quantity two chapters earlier.

The proof that this is the gap: `chapter5.tex:485–487` finally writes it —

> *"$u \equiv \ngs/\nion = \bdep\,\Zsb{1}(\Te)\,\nel$, a dimensionless ground-state
> population per unit ion density"*

— and `chapter7.tex:57` writes it again, calling it *"the third argument, the
neutral-to-ion ratio $u = \ngs/\nion$"*. The identity arrives 1100 lines after the
reader needed it.

- **Missing prerequisite:** none. The reader has everything; the sentence is simply
  absent.
- **Gap type:** intuition, severe.
- **Smallest fix:** two sentences immediately after Eq. (3.35) at l.915 — *"Note what
  the product $\bdep\Zsb{1}\nel$ is: rearranging Eq.~\eqref{eq:departure} gives
  $\bdep\Zsb{1}\nel = \ngs/\nion$, the neutral-to-ion ratio that
  Section~\ref{sec:the_assumption} identified as the third argument a two-axis table
  must dispose of. Everything below is written in $\bdep$ rather than in $\ngs/\nion$
  only because $\bdep$ measures that ratio against its thermodynamic reference."*
  Every subsequent equation in §3.5–§3.6 then reads physically.

### 3.5 §3.4.3 — the Bodenstein bridge stops one step short of the Schur complement

Writing-spec §4 lists *"Schur complement $\Omqss$ ↔ eliminating intermediates to get
an effective overall rate"* as a genuine correspondence to be used. The chapter sets
it up perfectly and then does not close it.

At l.664–671 the analogy is stated:

> *"For a reaction $A \to I \to P$ with a short-lived intermediate, one sets
> $\mathrm{d}[I]/\mathrm{d}t = 0$ and solves algebraically for
> $[I] = (\text{production})/(\text{loss})$."*

At l.735–740 the Schur complement is explained — well — but purely in plasma terms:

> *"$\Lmat_{SS}$ is the direct loss from the ground state, and
> $-\Lmat_{SF}\Lmat_{FF}^{-1}\Lmat_{FS}$ is the correction for atoms that leave the
> ground state, wander through the excited manifold, and either come back or are
> ionised on the way."*

The reader owns the ChemE half of this completely: back-substituting $[I]$ into
$\mathrm{d}[A]/\mathrm{d}t$ gives $\mathrm{d}[A]/\mathrm{d}t = -k_{\rm eff}[A]$, and
$k_{\rm eff}$ is a $1\times1$ Schur complement. Telling them that converts $\Omqss$
from a new object into one they have computed by hand in a kinetics course.

- **Missing prerequisite:** none — the reader owns it; the link is not drawn.
- **Gap type:** intuition. This is the single cheapest win in the chapter after §3.4.
- **Smallest fix:** one sentence after l.740 — *"In the $A \to I \to P$ picture this
  is the step where substituting $[I]$ back into the balance on $A$ yields a single
  effective rate constant $k_{\rm eff}$ for $A \to P$. $\Omqss$ is that $k_{\rm eff}$,
  for 42 intermediates instead of one."*

### 3.6 `chapter3.tex:922` — $\chi_p = 13.606/p^2$ is exactly the trap Chapter 2 warned about

Chapter 2 §2.3.2 is a careful 35-line section distinguishing the Rydberg energy
(13.605693 eV) from the ionisation energy of real hydrogen (13.598434 eV), and closes
(`chapter2.tex:382–387`) with:

> *"the Rydberg energy is stored in the source under a name — `CHI_H` — conventionally
> reserved for the ionisation potential. **Anything that reads that symbol expecting
> an ionisation potential, a Saha or Boltzmann exponent in particular, is
> \SI{0.0534}{\percent} too large.**"*

Chapter 3 Eq. (3.36), the Saha–Boltzmann coefficient, is

> $\Zsb{p} = 10^{6}\frac{g_p}{2g_i}\left(\frac{h^2}{2\pi m_e e\Te}\right)^{3/2}
> \exp(\chi_p/\Te)$, with $\chi_p = \frac{13.606}{p^2}$

13.606 is the Rydberg. It is being used in a Saha exponent. That is the exact
construction Chapter 2 flagged as a trap, two chapters after flagging it, with no
acknowledgement.

Numerically it matters least where it matters most: the exponent error is
$\exp(0.0073/\Te)$, i.e. 0.7 % in $\Zsb{1}$ at $\Te = 1$ eV and 0.07 % at 10 eV. Since
$\bdep^{\rm CRE}$, the switching points $c_m/a_m$ and $u^{\rm peak}$ all carry
$\Zsb{1}$, this propagates — probably below the resolution of anything reported, but
that has not been said.

- **Gap type:** detail, but it is a **reported-not-repaired item**, not a pedagogy
  item, and per CLAUDE.md I am not proposing a fix to the number.
- **Smallest fix (pedagogy only):** state which constant Eq. (3.36) uses and why, with
  a cross-reference to §2.3.2 and one line bounding the effect. Either choice is
  defensible; the silence is not.

### 3.7 `chapter3.tex:971–1030` — a 60-line optical-depth detour inside the central derivation

The reader is mid-way through the chapter's core argument: §3.5.2 has just established
that $\bdep$ is a hidden dynamical variable, and §3.5.3 is about to ask whether
$\Robs$ actually depends on it. Between them sits `\subsubsection*{Where the optically
thin assumption holds}` — a table of escape factors at five grid points, a discussion
of slab thickness $D$, and the declaration that results are stated for
$\Te \gtrsim 2$ eV.

Every word of it is needed somewhere. None of it is needed *there*. It answers a
question ("is the matrix right?") that belongs to Chapter 4, and its consequence
(the $\Te \ge 2$ eV scope) is established properly and at length in Chapter 5 §5.9 and
Chapter 6 §6.1, both of which measure it rather than assert it.

- **Gap type:** structural. It breaks the one place in the thesis where the reader most
  needs an unbroken line of argument.
- **Smallest fix:** move the subsubsection to Chapter 4 (it is a statement about the
  matrix) or to the end of §3.5, after Eq. (3.51). Leave a two-line forward pointer
  where it currently sits.

### 3.8 `chapter3.tex:1262–1264` — the tanh derivation is buried inside a paragraph about terminology

The paragraph running l.1249–1273 carries three unrelated ideas: (a) that a large
$\bdep$ does not mean a ground-fed manifold; (b) that neither should be confused with
"ionising"; and then, with no paragraph break, (c) the entire derivation of the
chapter's closed-form bound:

> *"…and is not fixed by the magnitude of $\bdep$ alone. Writing
> $s = x - \tfrac12(x_3+x_4)$ and using $\ffrac{}(-y) = 1 - \ffrac{}(y)$, the
> difference $\ffrac{3}-\ffrac{4}$ is \emph{even} in $s$. Its maximum therefore sits
> at the midpoint…"*

That sentence *is* the proof of $\max|\ffrac{3}-\ffrac{4}| = \tanh(|\Delta|/4)$, which
Chapter 5 §5.5, Chapter 6 §6.3 and Chapter 7 §7.2 all lean on as the thesis's one
uniform bound. It gets a subordinate clause at the tail of a paragraph about
nomenclature. Writing-spec R9 (one idea per paragraph) is violated three ways over.

Additionally, `\ffrac{}` with an empty subscript renders as $f_{}$ — the reader must
infer that a *bare* logistic function $f(y) = 1/(1+e^{-y})$ is meant, distinct from
$\ffrac{3}$ and $\ffrac{4}$, which are that function evaluated at shifted arguments.

- **Missing prerequisite:** none; the reader can do the symmetry argument. What is
  missing is the signal that this is the derivation and not an aside.
- **Gap type:** structural presenting as intuition.
- **Smallest fix:** break the paragraph before "Writing $s = \dots$", and give the
  derivation its own two-sentence paragraph opening *"The maximum follows from a
  symmetry."* Define the bare logistic $f(y)$ explicitly rather than writing
  `\ffrac{}`.

### 3.9 A statement Chapter 4 refutes

`chapter3.tex:222–225`:

> *"An error in a single off-diagonal element, a transposed index, or a missing
> back-reaction all break this immediately."*

`chapter4.tex:168`, on that exact sentence: *"That claim is testable, and it is
false."* Table 4.1 shows three of four injected faults — including a transposed
excitation array and deleting all de-excitation — leave the residual unchanged in the
third digit.

A reader who reaches Chapter 4 is told that a claim they were given in Chapter 3 was
wrong. Chapter 4 handles this gracefully and honestly, and the demolition is one of the
best passages in the thesis. But the claim in Chapter 3 stands unqualified.

- **Gap type:** structural/continuity.
- **Smallest fix:** soften l.224 to *"…would break this if the diagonal were built
  independently of the off-diagonals. §4.2 measures how much of that is true, and the
  answer is less than one would hope."* This preserves the surprise Chapter 4 is
  structured around while stopping Chapter 3 from asserting something false.

### 3.10 A promise Chapter 4 says was not kept

`chapter3.tex:190–195`:

> *"At $\nel = \SI{e12}{\per\cubic\centi\metre}$ the three-body channel supplies under
> 10\% of the feed into any level … a fact used in Chapter~\ref{ch:validation} to test
> the two recombination datasets separately."*

`chapter4.tex:1245–1256` measures it: 30 of 36 levels exceed 10 % three-body feed at
that density, and *"no such separate test was run."*

- **Gap type:** continuity.
- **Smallest fix:** Chapter 4 already carries a `\todo` for this. Either drop the
  promise from Chapter 3 or state it as an open item there.

---

## 4. Chapter 4 — *Does the Model Work?* (not in the build)

Read from `chapter4.tex`. Pedagogically this is the strongest chapter in the thesis.
The severity framework (§4.1: tautological / weak / severe) is exactly the right
teaching device, the CSTR mass-balance analogy at l.99–105 is the correct ChemE
bridge and is genuinely explanatory, and the fault-injection table is the single most
convincing page in the document. `% QUESTION:` on all 13 sections. Acronyms: RMPS
(l.51), QSS (l.54), CCC (l.279), LTE (l.354), ADAS/SCD/ACD (l.996–998) — all
expanded.

### 4.1 `chapter4.tex:1245` — a cross-reference points at the wrong chapter

> *"\subsection{A statement in Chapter~\ref{ch:theory} that measurement contradicts}
> … Section~\ref{sec:ion_recomb} states that at $\nel = \SI{e12}{...}$ …"*

`sec:ion_recomb` is `chapter2.tex:487`, i.e. Chapter 2. The prose says Chapter 3, and
the accompanying `\todo` says `chapter3.tex:190--195` — which is correct, and is
`sec:rate_equation`. The `\ref` will render as "Chapter 3" (correct) but the section
reference will send the reader to §2.4.

- **Gap type:** detail.
- **Smallest fix:** replace `\ref{sec:ion_recomb}` with `\ref{sec:rate_equation}`.

### 4.2 §4.7.3 supersedes Chapter 6 §6.7, and Chapter 6 does not know

`chapter4.tex:979–988` withdraws the $r_1$ deficit as a statement about the model:

> *"**Status.** The $r_1$ comparison at low $p$ is withdrawn as a statement about this
> model and reported as an unresolved external comparison. … \todo{… This supersedes
> \texttt{chapter6.tex} \S\ref{sec:r1_deficit}, which still publishes the discrepancy
> as a model failure.}"*

`chapter6.tex:828–886` is titled *"An external benchmark this model fails, and has not
explained"* and says *"this is reported open because it is open"*, with no reference to
Chapter 4's five lines of evidence that the target is unverified.

A reader going 4 → 6 in order meets a withdrawal and then, twenty pages later, the
withdrawn claim restated as a model failure.

- **Gap type:** continuity, high severity for credibility.
- **Smallest fix:** Chapter 6 §6.7 becomes *"An open comparison whose target is
  unverified"*, three sentences, pointing at §4.7.3 for the evidence and keeping only
  the scope consequence (which is Chapter 6's actual job). Chapter 4's own header
  (l.14–25) already diagnoses this and two further Ch4/Ch6 overlaps
  (`sec:open_system` vs `sec:attacks`, `sec:nmax` vs `sec:convergence`).

---

## 5. Chapter 5 — *What the Model Says*

Scientifically the strongest chapter. The opening statement of both results before the
evidence (l.68–99) is exactly right for this reader, the reactive-intermediate framing
at l.130–135 is the correct bridge and is used as structure rather than decoration, and
§5.3 ("Why a percentage is not a result") is a genuinely instructive piece of writing.
`% QUESTION:` on all 11 sections. Acronyms QSS (l.71), ELM (l.114), CRE (l.125), LTE
(l.833) all expanded.

### 5.1 `chapter5.tex:843, 850, 857, 868, 1509` — $u^{\rm peak}$ is used five times and defined nowhere

It carries the chapter's headline mechanism sentence:

> *"**The maximum sits where $\ln(u^{\mathrm{CRE}}/u^{\mathrm{peak}})$ changes sign.**"*
> (l.868)

It heads a column of Table 5.6 (l.857). It appears in the chapter summary (l.1509). It
is never defined. From Chapter 3 the reader can reconstruct it — `chapter3.tex:1277`
gives *"the maximum is attained at $\bdep\Zsb{1}\nel = \sqrt{(c_3/a_3)(c_4/a_4)}$, the
geometric mean of the two switching points"* — but Chapter 3 never names that quantity,
and Chapter 5 never says that is what $u^{\rm peak}$ means.

- **Missing prerequisite:** a name for the reservoir at which $\ffrac{3}-\ffrac{4}$
  peaks.
- **Gap type:** detail, high severity — the reader cannot check the chapter's central
  mechanistic claim.
- **Smallest fix:** one clause at l.843 — *"where $u^{\rm peak} \equiv
  \sqrt{(c_3/a_3)(c_4/a_4)}$ is the reservoir at which the sensitivity peaks,
  Eq.~\eqref{eq:peak_sensitivity}"* — and one clause in Chapter 3 at l.1277 naming it
  there.

### 5.2 `chapter5.tex:1346` — $x$ acquires a third meaning

Three distinct objects are written $x$ in this thesis:

| symbol | meaning | where |
|---|---|---|
| $x$ | ground-density scale factor | `thesis_main.tex:84` (macro `\xscale`) |
| $x = \ln(\bdep\Zsb{1}\nel)$ | logarithmic reservoir variable | `chapter3.tex:1228`, and $x_3, x_4$ throughout §3.6 |
| $x = \ngs^{\rm new}/\ngs^{\rm old}$ | fractional collapse of the reservoir across a step | `chapter5.tex:1346`, Table 5.11 |

Worse, `chapter5.tex:1369` then writes *"$x = 0.506$ is $|\ln x| = 0.68$"* — a
logarithm of the third $x$, in a thesis where $\ln$ of the second $x$ would be
meaningless and where $\Delta\ln u$ is the same number. The reader has to hold three
$x$'s and know which one is being logged.

- **Gap type:** detail, but this is the reader's third meeting with the symbol and the
  one that matters most numerically.
- **Smallest fix:** rename the §5.9.1 quantity. It is $u^+/u^-$ up to the fixed
  $\nion$, so write it as such, or call it $\rho$. Writing-spec §6 requires the symbol
  to come from the macro block; none of the three uses does consistently.

### 5.3 `chapter5.tex:432–436` — a two-photon rate the model does not contain, presented as fact

> *"It decays instead by emitting two photons at once, at $8.229\,\si{\per\second}$, a
> lifetime of \SI{0.12}{\second} against roughly a nanosecond for the $2p$ level beside
> it."*

`chapter2.tex:191–193` is explicit that this rate is **not in the dataset**:

> *"In this dataset the total radiative loss from $2s$ is $\gamma_{2s} = 0$ identically;
> the two-photon decay hydrogen really possesses, of order \SI{10}{\per\second}, is not
> included"*

The Chapter 5 argument is a legitimate thought experiment — l.452–455 says *"Delete the
$\ell$-mixing entirely and leave $2s$ with nothing but two-photon emission"* — but l.432
states the rate as a property of the system whose matrix is under discussion, and a
reader who read Chapter 2 carefully will read it as a contradiction.

- **Gap type:** detail/continuity.
- **Smallest fix:** one clause at l.435 — *"at $8.229\,\si{\per\second}$, a rate this
  model's radiative dataset does not include (Section~\ref{sec:radiative})"*.

### 5.4 Chapter 3 promises two error measures; Chapter 5 runs on two different ones

`chapter3.tex:1477–1480`:

> *"**They are not the same question** … Naming them apart is the single most important
> definitional step in this thesis."*

— referring to $\epsQSS$ and $\epsCRE$. Chapter 5 then uses $\epsQSS$ and $\epsCRE$
once each (§5.1.2) and spends the remaining 1300 lines on $\epsstep$ and $\epsplat$,
built on a **third reference state**, $\Rpe$ (partial equilibrium), which Chapter 3
never mentions. I verified: the strings `Rpe`, "partial equilibrium" and
"partial-equilibrium" appear nowhere in `chapter3.tex`.

Chapter 5 is honest about this (l.160–162: *"both are properties of a temperature step
rather than of a single operator, which is why neither appears in Chapter 3"*), and
Chapter 4 §4.9.4 notes the thesis *"defines four of them separately"*. But the reader
finishes Chapter 3 believing the two-measure distinction is the definitional spine, and
then finds the results chapter running on a different pair.

- **Missing prerequisite:** the partial-equilibrium state, and a statement of how the
  four measures relate.
- **Gap type:** intuition/continuity. Moderate — Chapter 5 recovers well, but the
  reader carries an obsolete frame across a chapter boundary.
- **Smallest fix:** one short subsection at the end of Chapter 3 §3.5 naming the
  partial-equilibrium state $\Rpe$ physically (*"the state in which the fast variables
  have settled and the slow one has not — the object Chapter 5 measures"*), and a
  four-row table in Chapter 5 §5.1.2 relating $\epsQSS$, $\epsCRE$, $\epsstep$,
  $\epsplat$.

### 5.5 `chapter5.tex:897–899` — the Griem exponent 17/2 arrives from nowhere

> *"a version of Griem's LTE criterion … which places the boundary … at a density
> scaling as $n^{-17/2}$ … Moving the diagnostic from the $(3,4)$ pair to the $(4,5)$
> pair should therefore move the crest by $(5/4)^{8.5} = 6.7$. Measured: $7.2$."*

This is the chapter's sharpest quantitative test of the mechanism, and the whole
content is the exponent. Chapter 1 §1.7 introduced Griem's boundary qualitatively (*"at
what principal quantum number collisions become fast enough"*) but gave no scaling. The
reader is asked to accept 17/2 on a bare citation and then watch a prediction confirmed
to 7 %.

- **Missing prerequisite:** where $n^{-17/2}$ comes from — collisional rates rising
  roughly as $n^4$ against radiative rates falling as $n^{-9/2}$, so the crossing
  density scales as their ratio.
- **Gap type:** intuition. A reader who cannot see where the exponent comes from cannot
  judge whether $6.7$ vs $7.2$ is a real test or a coincidence.
- **Smallest fix:** one parenthetical — *"(the $17/2$ is the ratio of the collisional
  $n^4$ scaling to the radiative $n^{-9/2}$ scaling of §2.5 and §2.6)"* — if that is in
  fact the origin; otherwise one sentence stating what it is.

### 5.6 `chapter5.tex:1022` — units switch to m⁻³ once, mid-chapter

> *"The crest sits at $\SI{1.93e19}{\per\cubic\metre}$"*

Every other density in the thesis is cm⁻³. Chapter 6 repeats the switch at l.688. The
reader must convert to compare against the crest density
$1.93\times10^{13}\,\mathrm{cm}^{-3}$ quoted eight pages earlier.

- **Gap type:** detail.
- **Smallest fix:** *"$1.93\times10^{13}\,\si{\per\cubic\centi\metre}$, that is
  $1.93\times10^{19}\,\si{\per\cubic\metre}$ in the units the edge-modelling literature
  uses"*.

### 5.7 Four symbols in the headline equation are not in the macro block

`chapter5.tex:30–38` records this against itself:

> *"MACROS THIS FILE USES THAT ARE NOT YET IN thesis_main.tex PART 2. Per
> writing_specification.md Sec. 6 these must be added there, not defined here:
> `\Sbar`, `\ures`, `\gain`, `\sep`"*

$\overline{S}$, $u$, $G$ and $\Delta$ carry the chapter's central result
(Eq. 5.9, quoted at l.84 and again at `chapter7.tex:94`) and are written longhand in
three chapters. This is the mechanism by which one chapter comes to say 22.7 µs while
another says 15.3 — the exact failure `thesis_main.tex:16–17` exists to prevent.

- **Gap type:** structural.
- **Smallest fix:** add the four macros to `thesis_main.tex` Part 2 and use them.

---

## 6. Chapter 6 — *What This Model Cannot Say*

Pedagogically excellent. The dedicated abbreviation block at l.60–70 is the model the
other chapters should follow. The product re-adsorption analogy for radiation trapping
(l.88–94) is a genuine ChemE bridge doing explanatory work. §6.2's batch-reactor /
continuous-flow contrast is the clearest statement of the transport objection anywhere
in the thesis. Every section states the falsifier before the result. `% QUESTION:` on
all 10 sections.

### 6.1 `chapter6.tex:615–618` — new notation for objects that already have names

> *"Within the two-channel split the ground-fed fraction is exactly
> $\ffrac{p} = a_1 n_g/(a_0 + a_1 n_g)$, where $n_g$ is the ground-state density feeding
> the excitation channel and $a_0$, $a_1$ are the network responses of
> Section~\ref{sec:R_depends_on_b}."*

A reader who follows that cross-reference lands on `chapter3.tex:1051–1092`, where the
network responses are $\bm a$ and $\bm c$, the formula is
$\ffrac{m} = a_m\bdep\Zsb{1}\nel / (a_m\bdep\Zsb{1}\nel + c_m)$, and the symbols
$a_0$, $a_1$ and $n_g$ do not appear. The mapping is $a_1 \to a_m$, $a_0 \to c_m$,
$n_g \to \ngs$. Worse, $a_0/a_1$ collides visually with Fujimoto's $r_0/r_1$
(`chapter3.tex:1375`, `chapter6.tex:843–845`), where the index means the *opposite*
channel: $r_0$ is recombination-fed, $r_1$ is ground-fed, so $a_0 \leftrightarrow c_m$
happens to match $r_0$ by accident and $a_1 \leftrightarrow a_m$ matches $r_1$ — but
nothing tells the reader that, and $a_1$ against $\bm a$ against $a_3$/$a_4$ is three
different things.

- **Gap type:** detail, high severity. This is the one equation in Chapter 6 the reader
  must be able to re-derive, because §6.4's whole argument is scaling $n_g$ in it.
- **Smallest fix:** write the equation in Chapter 3's symbols:
  $\ffrac{m} = a_m\ngs/(a_m\ngs + c_m)$, which is the same expression and is exactly
  Chapter 3's Eq. (3.48) once $\bdep\Zsb{1}\nel$ is recognised as $\ngs/\nion$
  (see §3.4 above). Note that making the §3.4 fix makes this equation *self-evidently*
  Chapter 3's, which is the point.

### 6.2 `chapter6.tex:361, 1000` — the reservoir displacement is written $x$ here and $u$ in Chapter 5

> *"so $|\Delta\ln\xscale|$ roughly halves"* (l.361)
> *"displaces the reservoir by $|\ln\xscale|$ between $0.124$ and $0.682$"* (l.1000)

`\xscale` expands to $x$. Chapter 5 writes the identical quantity as $|\Delta\ln u|$
(l.523, 527, 787, 789, 809, 989 …) and Chapter 7 as $\Delta\ln u$ (l.131). `\xscale` is
used **only** in Chapter 6, nowhere else in the thesis. So the same central quantity has
two names, split by chapter.

- **Gap type:** detail, but it is exactly the class of defect writing-spec §6 exists to
  prevent.
- **Smallest fix:** use $u$ throughout, or retire `\xscale` from the macro block.

### 6.3 `chapter6.tex:180–182` — a cross-reference to the wrong section

> *"That is why the results of Chapter~\ref{ch:results} were declared for
> $\Te \gtrsim \SI{2}{\electronvolt}$ in the first place
> (Section~\ref{sec:two_references})."*

`sec:two_references` is Chapter 3 §3.5, *"Two reference states"*. The reader clicks and
lands on a section about $\Rqss$ and $\Rcre$. The scope declaration they are being
pointed at is either Chapter 3's buried subsubsection (§3.7 above, which should move
anyway) or, far better, Chapter 5 §5.9 `sec:scope`, which measures the boundary rather
than asserting it.

- **Gap type:** detail.
- **Smallest fix:** point at `sec:scope`.

---

## 7. Chapter 7 — *What It Means* (not in the build)

Very good. §7.2's three-step "procedure" is the correct form for the deliverable and is
the one place in the thesis where the reader is handed something to *do*. The
Michaelis–Menten remark at l.264 is a well-judged ChemE bridge — the reader owns
saturation kinetics and will recognise the logistic instantly.

The chapter also does what Chapter 3 should have done: `chapter7.tex:57` writes *"a
third argument, the neutral-to-ion ratio $u = \ngs/\nion$"*, explicitly closing the loop
back to §1.5.

### 7.1 Acronyms unexpanded in this chapter

| acronym | first use | expanded? |
|---|---|---|
| CR | l.117 *"For a CR matrix of one's own"* | no ("collisional--radiative" first appears at l.267, 150 lines later) |
| CRE | l.112 (`u^{\mathrm{CRE}}`) | no |
| ELM | l.354 *"the ELM-averaged bound"* | expanded at l.388, i.e. 34 lines **after** first use |

QSS is correctly expanded at l.40.

- **Gap type:** detail.
- **Smallest fix:** expand CR and CRE in the chapter opening (l.21–28); move the ELM
  expansion from l.388 to l.354.

### 7.2 `chapter7.tex:365–369` — the chapter flags its own contradiction with Chapter 6

> *"\emph{[UNVERIFIED: Section~\ref{sec:molecules} states that no bound on the molecular
> effect can be constructed from the data in this repository, and ADDENDUM~D.4
> constructs one from the two references cited in that same section. The two statements
> are inconsistent.]}"*

`chapter6.tex:565–567` does indeed say *"this work offers no bound on how much, and none
can be constructed from the data in this repository"*, and Chapter 7 l.359–363 then
quotes a factor-3-to-5 bound. This is correctly flagged rather than hidden; recording it
here so it is not lost in the list of open items.

---

## 8. Cross-cutting checks (the four the task asked for explicitly)

### 8.1 Acronyms, expanded at first use in each chapter

| | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | ch7 |
|---|---|---|---|---|---|---|---|
| CR | ✅ 359 | n/a | ❌ 484 | n/a | n/a | ✅ 61 | ❌ 117 |
| QSS | ✅ 510 | n/a | ❌ 699 | ✅ 54 | ✅ 71 | ✅ 62 | ✅ 40 |
| CRE | ✅ 471 | n/a | ❌ 1012 | n/a | ✅ 125 | ✅ 63 | ❌ 112 |
| ADAS | ✅ 399 | ✅ 749 | ❌ 744 | ✅ 996 | n/a | ✅ 748 | n/a |
| SCD / ACD | n/a | n/a | ❌ 1402/1405 | ✅ 997/998 | n/a | n/a | n/a |
| LTE | ✅ 368 | n/a | n/a | ✅ 354 | ✅ 833 | ✅ 596 | n/a |
| MAR | ✅ 713 | n/a | n/a | n/a | n/a | ✅ 532 | n/a |
| PSM20 | n/a | ⚠️ 875 named, not expanded | n/a | n/a | n/a | n/a | n/a |
| CCC | n/a | ✅ 397 | n/a | ✅ 279 | n/a | n/a | n/a |
| SOL | ✅ 77 | n/a | n/a | n/a | n/a | n/a | n/a |
| ELM | ✅ 547 | n/a | n/a | n/a | ✅ 114 | ✅ 66 | ⚠️ used 354, expanded 388 |
| RMPS | n/a | ✅ 1003 | n/a | ✅ 51 | n/a | ✅ 865 | n/a |

**Nine failures, six of them in Chapter 3.**

### 8.2 Terms introduced before the object they name (writing-spec R3)

R3 is honoured almost everywhere and conspicuously well in places — "two clocks" before
$\taurel$/$\tauqss$, "fed from below / from above" before $a_p$/$c_p$, "the plateau"
before $\Rpe$. Four violations:

1. **$\Omqss$ and "Schur complement"** are used in the $\eta$ argument
   (`chapter3.tex:415–428`) 300 lines before either is named or described (§3.2 above).
2. **$u^{\rm peak}$** is used before — in fact instead of — any description of the
   reservoir at which sensitivity peaks (`chapter5.tex:843`, §5.1 above).
3. **$\Msep^+$ / $\Msep^-$** are formalised at `chapter3.tex:486` before the temperature
   step they describe has been introduced anywhere (§3.3 above).
4. **$\bdep$** is formalised at `chapter3.tex:911` without ever being named as the thing
   Chapter 1 already described in words, the neutral-to-ion ratio (§3.4 above). This is
   R3's exact failure mode: the reader is handed the symbol before they are told which
   physical object it is.

### 8.3 Are the ChemE bridges structural or decorative?

| bridge (writing-spec §4) | verdict |
|---|---|
| rate matrix ↔ linear reaction network | **structural.** `chapter3.tex:68–73, 164–168`; used to motivate the whole matrix form. |
| excited states ↔ reactive intermediates | **structural.** `chapter5.tex:130–135` uses it to carry the three-timescale narrative. |
| ground state ↔ bulk reactant reservoir | **structural**, though never stated in those words; the reservoir framing does all the work in Ch5–Ch7. |
| QSS ↔ Bodenstein | **structural.** `chapter1.tex:514`, `chapter2.tex:31`, `chapter3.tex:664–671`, `chapter6.tex:62`. |
| recombination source ↔ CSTR feed | **structural.** `chapter3.tex:151–169` gives it a whole subsection titled *"Why recombination is a feed stream, not a matrix element"*; `chapter6.tex:742`. |
| **Schur complement ↔ eliminating intermediates for an effective rate** | **⚠️ set up and not closed** — see §3.5 above. The one bridge in the list that is not delivered. |
| ionisation ↔ irreversible consumption | **structural.** `chapter2.tex:496`. |
| two supply channels ↔ two feed streams | **structural.** `chapter6.tex:519–527`. |
| detailed balance ↔ forward/reverse consistency | **structural.** `chapter2.tex:300–303`. |
| trapping ↔ product re-adsorption | **structural.** `chapter2.tex:739–741`, `chapter6.tex:88–94`. |
| stiffness ratio $M$ | **structural.** `chapter2.tex:159–161`. |
| lookup table ↔ calibration curve the plant violates | **absent.** Never used, in any chapter. Low cost to add and it is the sharpest of the twelve for framing Chapter 1's complication. |

Eleven of twelve are genuine. The one gap that costs comprehension is the Schur
complement.

### 8.4 Does the Bodenstein sentence appear early, in those words?

**Yes.** `chapter1.tex:514–520` and, in a cleaner form, at the very top of Chapter 2
(`chapter2.tex:30–37`), before any physics:

> *"One sentence belongs before any physics, because it is the reader's fastest route
> into everything that follows. **The approximation this thesis tests is the Bodenstein
> steady-state approximation, applied to atomic energy levels instead of radical
> concentrations.** Anyone who has written $\mathrm{d}[\mathrm{R}^{\bullet}]/\mathrm{d}t
> \approx 0$ for a reactive intermediate has used it."*

This is the single best-judged paragraph in the thesis for this reader and it is
correctly placed. No change needed.

---

## 9. Prioritised list

**Blockers — the reader cannot finish the thesis without these.**

1. `\input{chapter4}` and `\input{chapter7}` in `thesis_main.tex` (§0.1).
2. Move the $\eta$ algebra out of `chapter3.tex:404–439` into §3.4 (§3.2).
3. State $\bdep\Zsb{1}\nel = \ngs/\nion$ at `chapter3.tex:915` (§3.4).

**High — each costs one to three sentences and buys a chapter.**

4. Define $u^{\rm peak}$ at `chapter5.tex:843` (§5.1).
5. Close the Schur-complement ↔ Bodenstein bridge at `chapter3.tex:740` (§3.5).
6. Move "Which operator $\Msep$ belongs to" to Chapter 5 (§3.3).
7. Add Chapter 3's acronym block; fix the nine acronym failures (§8.1).
8. Rewrite `chapter6.tex:615` in Chapter 3's symbols (§6.1).
9. Add the seven `% QUESTION:` comments to Chapter 3 (§3.0).
10. Soften `chapter3.tex:224` so Chapter 4 does not have to call it false (§3.9).
11. Reconcile Chapter 6 §6.7 with Chapter 4 §4.7.3 (§4.2).

**Medium.**

12. Break the paragraph at `chapter3.tex:1264` and give the tanh derivation its own
    (§3.8).
13. Move the optical-depth subsubsection out of `chapter3.tex:971–1030` (§3.7).
14. Fix $\mu = 918.08$ at `chapter2.tex:890` (§2.1).
15. Rename the third $x$ at `chapter5.tex:1346`; unify $\Delta\ln u$ vs $\Delta\ln x$
    (§5.2, §6.2).
16. Add the four missing macros to `thesis_main.tex` Part 2 (§5.7).
17. Name the partial-equilibrium state in Chapter 3 (§5.4).
18. Give the origin of the $n^{-17/2}$ exponent at `chapter5.tex:897` (§5.5).
19. Decide what to do about `\todo`/`[UNVERIFIED]` rendering (§0.2).

**Low.**

20. `chapter1.tex:426` — qualify "exact" (§1.1).
21. `chapter2.tex:893` — $6n^4\ell_>$ (§2.2).
22. `chapter2.tex:56` — connect $I_n$ to $E_n$ (§2.3).
23. `chapter3.tex:922` — state which hydrogen energy Eq. (3.36) uses, per §2.3.2 (§3.6).
24. `chapter5.tex:435` — note the two-photon rate is not in the matrix (§5.3).
25. `chapter6.tex:182` — repoint to `sec:scope` (§6.3).
26. `chapter4.tex:1245` — repoint to `sec:rate_equation` (§4.1).
27. `chapter1.tex:391` — "the Alcator C-Mod tokamak" (§1.2).
28. `chapter5.tex:1022` — give the crest density in cm⁻³ as well (§5.6).
29. `chapter2.tex:875` — say what PSM20 labels (§2.4).
30. Consider adding the calibration-curve bridge to Chapter 1 (§8.3).

---

## 10. What I did not do

Reported only. Nothing was rewritten, no file under `thesis_tex/` was modified, no
script was run, no number was checked for correctness — that is the skeptic's and the
verifier's job, not this gate's. Where I noticed a physics or arithmetic problem
(§2.1, §2.2, §3.6) I report it as a place the reader stumbles and leave the repair
decision open, per CLAUDE.md.
