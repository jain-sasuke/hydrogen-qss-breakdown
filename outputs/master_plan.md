# Master Plan — Thesis to Defense
**Replanned 14 July 2026, calibrated to demonstrated level**

**Defense:** 1 September 2026 — **7 weeks**
**Scope decision:** full rewrite, all seven chapters
**Student:** Nikhil Jain, M.Tech ChemE, IIT Kanpur · **Supervisor:** Prof. R. G. S. Pala

---

## PART 0 — Where you actually stand

Calibrated from the Q4 exam, the D1–D5 diagnostic, and the prerequisite
sequence completed 14 Jul. This is the honest assessment, not an encouraging one.

### Strong

| Skill | Evidence |
|---|---|
| Calculus, ODE solving | D1 solved cleanly by separation of variables |
| Matrix mechanics | D2 correct |
| Physical intuition | Derived the entire ℓ-mixing chain unaided; named Debye screening unprompted; got the QSS-target/step-distance insight in one sentence |
| Catching errors in *my* work | Caught the $\lambda$/$\tau$ units confusion; challenged unverifiable numbers and demanded reproducible scripts |
| Methodological instinct | Insisted on no-hardcoding, no-black-box, own-machine reproduction. These were your calls, not mine |

### Fixed during the prerequisite sequence

| Was weak | Now |
|---|---|
| Index convention ($L[i,j]$ direction) | Solid — read $L[0,2]$ correctly unprompted |
| Units discipline | Improved; still the thing to watch |
| Eigenvector interpretation | Solid — read exchange vs drain, ground-excluded modes, ℓ-mixing signature |
| Steady state vs eigenmode | Solid (P4) |
| Modal amplitudes, left eigenvectors | Derived by hand, verified numerically |
| Non-normality, $\mu(L)$ | Derived the criterion from $\frac{d}{dt}\|x\|^2$ |

### Still to prove

- The Q4 exam parts 2–5 (mathematical, physics, derivation, code) — **not yet attempted**
- Whether the prerequisite material holds under defense-style pressure

### Working pattern that has been shown to work

**Works:** one quantity per session · physics before math before code · derive
by hand before reading any implementation · run everything yourself · outside
LLM as a checkable claim, never an authority.

**Fails:** multiple quantities at once · code opened before you can predict what
it should contain · accepting numbers printed in chat as evidence · toy
derivation used to audit full-theory code.

---

## PART 1 — The learning pattern (how every session runs)

### The six-step rhythm, per quantity

1. **Physics** — the picture in words, from primary sources (Fujimoto, Griem,
   van der Mullen, Badnell), never from project notes.
2. **Hand derivation** — you derive, on paper. I ask and wait. I do not derive
   it for you and ask you to confirm; that produces recognition, not
   understanding, and recognition collapses under questioning.
3. **Worked example** — real numbers, carried to an actual value. Symbolic work
   hides unit errors and inverted ratios; arithmetic exposes them.
4. **Brutal physics test** — limits, signs, dimensions, magnitudes,
   conservation. Hostile on purpose.
5. **Code** — only now. Line by line, against the hand derivation.
6. **Note** — written up before moving on. An unwritten result is a result you
   will re-derive in three weeks.

### The five test modes

Sequence: **Intuition → Numerical → Derivation → Code → Defense.**

| Mode | Catches |
|---|---|
| Intuition | Can compute but cannot say what it means |
| Numerical | Sign errors, unit slips, convention confusion |
| Derivation | Memorised formula with no foundation |
| Code | Physics understood but implementation not |
| Defense | Real understanding that is not deployable under pressure |

Anti-patterns, both observed here: *derivation before intuition* → symbol
shuffling. *Code before numerical* → black boxes.

### Ground-truth hierarchy

**physics → math → code → documents.** Documents are consistency checks, never
authorities — including your own notes and mine. A document does not become true
by being pasted recently.

### The no-black-box rule

Every script whose output enters the thesis is understood line by line: what
goes in, how it computes, what comes out. Applies to code I write as much as
inherited code. The `verify_timescales.py` review already caught two wrong
diagnostics that ran cleanly and printed confident nonsense.

### Outside LLM council

Use freely for physics checks, doubts, cross-checks, reviews. Its track record
here: **good at concrete catches** (arithmetic errors, mislabels,
inconsistencies), **weak at knowing when a result should be abandoned rather
than merely qualified**. Feed it clean current context, ask it to *derive* not
*opine*, bring the answer back as a claim to be tested.

---

## PART 2 — Learning still required

### 2.1 Immediate: Q4 exam re-attempt (½ session)

Parts 2–5, not yet attempted. Purpose is to locate remaining gaps before Q5.

- **Q2 mathematical** — general solution; prove the slowest mode sets the
  relaxation time; explain Framing A vs B agreement and why not exact
- **Q3 physics** — $n_e^{-0.47}$ vs pure collisional $n_e^{-1}$; the
  $\lambda_1/\lambda_2$ near-degeneracy; why the ℓ-mixing fix moved
  $\tau_{\rm relax}$ by only 0.12%
- **Q4 derivation** — 2-state eigenvalues exactly, then via the approximations
- **Q5 code** — write the timescale extraction from memory; find the bug in a
  filtered-eigenvalue snippet

**Gate:** if parts 2–5 pass, proceed to Q5. If not, targeted repair first.

### 2.2 Q5 — the central argument (4–5 sessions)

The thesis's main claim. Everything so far has been setup.

| Session | Content |
|---|---|
| **B** | Scalarise $\delta\mathbf n(0^+)$ → $\varepsilon^{L^2}_{\rm step}$ and $\varepsilon^{\rm ratio}_{\rm step}$ (Balmer), on paper, full 43-state rigour |
| **C** | Brutal physics test: small-step limit, large-step limit, $M\to\infty$, signs, dimensions |
| **D** | Reconcile the two $\tau_{\rm QSS}$ definitions; introduce the drive timescale and $De = \tau_{\rm relax}/\tau_{\rm drive}$ |
| **E** | Non-normality applied: $P_{\rm slow}$ via left eigenvectors; explain $\varepsilon_{\rm res}>\varepsilon_{\rm step}$; **test in the Balmer observable, not $L^2$** |
| **F** | Code review of `qss_analysis.py` and `Balmer_transient_ratio.py`; write `derivation_05.md` |

### 2.3 Q6 — Hα artifact (2 sessions)

Prove the corrected transient is $+43.7\%/-30.3\%$ monotonic. A special case of
Q5's step response applied to a real diagnostic line.

### 2.4 Q7 — the S criterion (2 sessions)

Derive $\varepsilon_{\rm step}\approx S\,\delta T_e$ with $S=\Delta E/kT_e^2$ —
the analytic explanation for the empirically density-independent contours.

---

## PART 3 — Full codebase audit

Per the no-black-box rule. Ordered by how load-bearing each script is.

| Priority | Script | Why | Status |
|---|---|---|---|
| 1 | `cr_context.py` | Everything else depends on it | ⏳ |
| 2 | `verify_timescales.py` | Produces the two-timescale and mode-identity results | ⏳ |
| 3 | `verify_boundary_descent.py` | Produces the Griem-descent result | ⏳ |
| 4 | `qss_analysis.py` | Produces the Q5 headline figures. **Known bug:** `eigs < -1.0` filter silently caps $\tau_{\rm QSS}$ at ~1 s | ⏳ |
| 5 | `Balmer_transient_ratio.py` | The diagnostic observable for Q5/Q6 | ⏳ |
| 6 | `assemble_cr_matrix.py` | Builds $L$; conservation verified, rest not | 🔄 partial |
| 7 | `radiative_rates.py` | Builds `gamma_bundled`, indirectly verified only | ⏳ |
| — | `compute_lmix.py` | Audited, bug found, fixed, bounded | ✅ |
| — | `compute_K_CCC.py` | Audited in Q2 | ✅ |

**Audit sequence per script:** upload → walkthrough (inputs, computation,
outputs, every magic number traced) → run on your machine → check against hand
derivation → discuss → document.

---

## PART 4 — Thesis check and rewrite

### 4.1 Phase 2 — reconcile (week 3)

**Claims ledger.** Read Ch. 3 and Ch. 5 line by line; every number tagged
{verified | stale | artifact | unbacked}.

**Corrections already identified:**

| # | Thesis says | Measured | Action |
|---|---|---|---|
| B1 | $\tau_{\rm relax}\propto n_e^{-1.00}$ | $n_e^{-0.46}$ to $n_e^{-0.53}$ | Correct — roughly half |
| B2 | 25 ns / 15.3 µs / $M$=611 | 2.277 ns / 22.73 µs / 9982 | Replace; spurious eigenmode |
| B3 | Hα −46% dip at 19 ns | +43.7%/−30.3% monotonic | Rewrite section |
| B4 | "three timescale groups" | One gap, then a continuum | Reframe |

Then regenerate every affected figure from the corrected `L_grid.npy`.

### 4.2 Phase 3 — rewrite (weeks 4–6)

Story-first, readable by a first-year graduate student: physical question →
definitions → intuition → rigour → result → why it matters.

| Chapter | Fed by | Week |
|---|---|---|
| Ch. 2 Atomic Data | Q2, Q4b | 4 |
| Ch. 3 Theory | Q1, Q3, Q4, Q4c, prerequisite notes | 4 |
| Ch. 4 Validation | Q2 Gate A, Q4 Gate E, ℓ-mixing robustness | 5 |
| Ch. 5 Results | Q5, Q6, Q7 | 5 |
| Ch. 1, 6, 7 | everything | 6 |

**Key leverage:** the dossier notes are already near-chapter prose. Ch. 3 is
substantially *assembly*, not fresh writing. Write chapters as the dossier
closes — do not wait for all seven quantities.

---

## PART 5 — Result backlog

Maintained in `thesis_grade_results_backlog.md`. Graduation:
💡 Idea → 📐 Derived → 💻 Coded → ▶️ Run → ✅ Verified. **Nothing enters the
thesis before ✅.**

**In hand (▶️ Run, need write-up):** boundary-level descent · one-gap spectrum ·
non-normality $\mu(L)$

**Verified (✅):** ℓ-mixing bug found, quantified, bounded

**Ideas (💡), highest value first:**

| # | Idea | Why it matters |
|---|---|---|
| C9 | Where does Balmer common-mode cancellation fail? | **Reframes the diagnostic story.** At the ITER ref, $\varepsilon^{L^2}=1.17$ but $\varepsilon^{\rm ratio}=0.038$ — 31× smaller, because $n_3$ and $n_4$ move together. "Diagnostics are biased" is *not* supported; "here is where the cancellation breaks" is the better paper |
| C10 | Excitation-fed or recombination-fed? Which shell is more $T_e$-sensitive? | **Decides how the $\varepsilon^{\rm ratio}$ sign is explained.** Preliminary data contradicts the Boltzmann expectation: $n_3$ is more sensitive than $n_4$, and heating *depletes* both. Points to a recombining regime — but the stand-in source may be forcing it. Connects to the ionising→recombining transition that defines detachment |
| C1 | Map $\mu(L)$ across the grid | If it predicts QSS breakdown better than $M$, that is a *result*, not a figure |
| C6 | Truncation sensitivity ($n_{\max}$ = 12, 15, 20) | At low density the relaxation mode has weight on bundled states — the cutoff may be affecting $\tau_{\rm relax}$. An examiner may well ask |
| C8 | Is the ℓ-distribution statistical at both step endpoints? | The $A$-coefficient cancellation — which makes $\varepsilon^{\rm ratio}$ independent of atomic rates — depends on it. Expected fine (ℓ-mixing is 1000× faster), but unverified |
| C5 | Does $\lvert v_{\lambda_1,3D}\rvert$ predict the Hα transient? | Links eigenstructure directly to the observable |
| C2 | Participation ratio across the spectrum | Turns "collective mode" into a number |
| C3 | Classify the fast continuum | Makes "quasi-continuum" structural rather than descriptive |
| C7 | Cold-corner exclusion criterion | 19 points have $\tau_{\rm QSS}>1$ s, inflating the quoted $M$ range |
| C4 | Pseudospectra | Strengthens non-normality, but may belong in the paper. **Decide before spending time** |

---

## PART 6 — Seven-week schedule

| Week | Dates | Content |
|---|---|---|
| **1** | Jul 14–20 | Q4 exam · Q5 sessions B–F |
| **2** | Jul 21–27 | Q6 · Q7 · **dossier closed** |
| **3** | Jul 28–Aug 3 | Claims ledger · corrections · regenerate figures · codebase audit priorities 1–5 |
| **4** | Aug 4–10 | Write Ch. 3 and Ch. 2 |
| **5** | Aug 11–17 | Write Ch. 5 and Ch. 4 |
| **6** | Aug 18–24 | Write Ch. 1, 6, 7 · full read-through |
| **7** | Aug 25–31 | **Buffer** — supervisor feedback, revision, submit |

**Week 7 is buffer, not work.** Protect it. Something will go wrong; that is
what it is for.

**Critical dependency:** every day the dossier slips is a day stolen from
writing, and writing has no slack.

---

## PART 7 — This week's actions

1. **Prof. Pala** — confirm full-rewrite scope; show the corrected numbers
   ($\tau_{\rm relax}=2.277$ ns, $M=9982$); flag the
   $\varepsilon_{\rm res}>\varepsilon_{\rm step}$ finding and the non-normality
   mechanism; get his read on the journal target.
2. **Prof. Bray email** — drafted; verify the Klein-Rosseland figures (870
   pairs, 0.9995, 98.7%) against `qc_ccc.py` output before sending.
3. **Q4 exam** — parts 2–5.
4. **Start Q5.**

---

## PART 8 — Standing decisions

| Decision | Settled as |
|---|---|
| $M$ convention | $M=\tau_{\rm QSS}/\tau_{\rm relax}$, large = QSS necessary condition holds |
| $\tau_{\rm relax}$ definition | Framing A ($\lambda_1$ of the full matrix); agreement with Framing B (<0.35%) reported as robustness |
| $\tau_{\rm QSS}$ definition | **OPEN** — eigenvalue vs target-motion. Q5 Session D |
| benchmark point | Illustrative anchor only. Prove with the grid, illustrate with the point. Grid values $T_e=2.947$ eV, $n_e=1.389\times10^{14}$ cm⁻³ |
| Q5 error measure | Both: $L^2$ norm (mathematical) and Balmer ratio (diagnostic) |
| Boundary descent | Report as a discrete staircase. **No fitted exponent** ($R^2\approx0.91$; slope shifts 20% when saturated points are dropped) |
| Spectrum framing | One gap, then a quasi-continuum. **Not** "three groups" |
| Transient growth | Always state the norm and the perturbation |
| Journal | Decide after the dossier. PRE / JQSRT / JPP class. Not Nature-tier for paper 1 |

---

## PART 9 — Scope limitations to state explicitly in §1.5

1. Maxwellian electrons (Q2 — non-Maxwellian breaks detailed balance)
2. $T_i = T_e$ in ℓ-mixing (Q4b — the relevant temperature is the proton one)
3. Optically thin (bounds the diagnostic-bias claim to a lower limit)
4. $n_{\max} = 15$ truncation, $n\ge9$ bundled (pending C6)
5. Uniform plasma, no transport
6. Ground state tracked, not frozen (confirmed: index 0 of $\mathbf n$)
