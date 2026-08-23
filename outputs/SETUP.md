# Setup — Thesis Writing, Verification, and Backlog System

Three tools, one shared memory, and a set of methods that make errors visible
instead of plausible.

---

## 1. The architecture

```
                   ┌──────────────────────┐
                   │   THE FILES          │
                   │  master_plan.md      │  ← the memory. Outlasts every
                   │  backlog.md          │    conversation. Authoritative.
                   │  derivation_*.md     │
                   └──────────┬───────────┘
                              │  all three read these
          ┌───────────────────┼───────────────────┐
          │                   │                   │
   ┌──────▼──────┐    ┌───────▼───────┐   ┌───────▼───────┐
   │ CLAUDE CODE │    │  CLAUDE CHAT  │   │    CHATGPT    │
   │  terminal   │    │               │   │               │
   │ runs code   │    │ derivation    │   │ independent   │
   │ reads repo  │    │ writing       │   │ cross-check   │
   │ = EVIDENCE  │    │ review stances│   │               │
   └─────────────┘    └───────────────┘   └───────────────┘
                              │
                        ┌─────▼─────┐
                        │    YOU    │  ← the only one who decides
                        └───────────┘
```

**Claude Code produces evidence.** It touched your files; it either ran or it
did not. Neither chat model can fake that.

**The two chat models produce interpretations.** Independent, and therefore
mutually checkable — but see §4.5 on why their *agreement* is weak evidence.

**The files are the memory.** This is what lets you switch chats without loss,
and what stops either model from re-importing a claim that was already retracted.

---

## 2. Staged setup — check after each step

Do these in order. **Stop at any check that fails** — a localised failure is
cheap to fix, a buried one is not.

### Step 1 — Files into the repo

```bash
cd /Users/phi/Desktop/non_markovian_cr
cp ~/Downloads/CLAUDE.md .
mkdir -p outputs
cp ~/Downloads/master_plan.md                       outputs/
cp ~/Downloads/thesis_grade_results_backlog.md      outputs/
cp ~/Downloads/prerequisite_notes_linear_algebra.md outputs/
cp ~/Downloads/SETUP.md                             outputs/
cp ~/Downloads/derivation_*.md                      outputs/
```

**Check:** `ls CLAUDE.md && ls outputs/` — expect CLAUDE.md plus master plan,
backlog, prerequisite notes, SETUP, and derivation notes 02, 03, 04, 04b, 04c, 05.

### Step 2 — Verification scripts

```bash
cp ~/Downloads/cr_context.py               src/validation/
cp ~/Downloads/verify_timescales.py        src/validation/
cp ~/Downloads/verify_boundary_descent.py  src/validation/
cp ~/Downloads/verify_ramp_vs_step.py      src/validation/
```

**Check:** `cd src/validation && python cr_context.py`

Expect the repo root, `L_grid (50, 8, 43, 43)`, both grids, 43 states starting
`1S, 2S, 2P, 3S`, n range 1–15, ground index 0. **This is load-bearing** — if
the state ordering is misread here, every downstream eigenvector interpretation
is wrong.

### Step 3 — Confirm the ℓ-mixing correction is in place

```bash
grep -c "_psm20_F" src/rates/compute_lmix.py
```

**Check:** `0` means the old $F=1$ version is still there → copy the corrected
one. Anything ≥1 means it is already correct.

### Step 4 — Reproduce a known result

```bash
cd src/validation && python verify_timescales.py --skip-grid
```

**Check:** $\tau_{\rm relax} \approx 2.276913\times10^{-9}$ s, largest gap
9981.93, Framing A vs B 0.0098%, $\lambda_0$ ground weight 1.0000, transient
growth in 3/3 trials.

**This is the real test of the whole chain** — files, scripts, data, and
corrections all consistent. If it reproduces, the setup is sound.

### Step 5 — Claude Code

Open Claude Code in the repo root:

> Read `CLAUDE.md`, then `outputs/master_plan.md` and
> `outputs/thesis_grade_results_backlog.md`. Summarise back to me: what this
> project is, what your role is, and the top three rules you're operating under.
> Don't run anything yet.

**Check:** it should name *report don't repair*, *no hardcoding*, *no black
boxes*, and know the benchmark point values. **If it offers to fix anything, the
CLAUDE.md hasn't landed** — adjust before letting it touch the repo.

### Step 6 — Start the new Claude chat

**Upload these ten:**

```
master_plan.md
thesis_grade_results_backlog.md
prerequisite_notes_linear_algebra.md
SETUP.md
derivation_02_detailed_balance.md
derivation_03_qss_steady_state.md
derivation_04_two_timescales.md
derivation_04b_ell_mixing.md
derivation_04c_non_normality.md
derivation_05_qss_breakdown.md
```

Plus the repo tree (`tree -I "e-H_XSEC_LS"` or similar).

**Do NOT upload an old chat transcript.** It contains superseded claims that look
exactly as authoritative as the live ones — the 25 ns value, the $p_G\approx17$
arithmetic error, the "three groups" framing, the sum-of-components diagnostic.
The notes exist so the transcript can be discarded.

**Opening prompt:**

> M.Tech thesis — time-dependent collisional-radiative modelling of hydrogen
> plasma, quantifying quasi-steady-state validity in ITER divertor conditions.
> IIT Kanpur, Chemical Engineering. Supervisor Prof. R. G. S. Pala. Defense
> 1 September 2026. Full rewrite, all seven chapters.
>
> **Read `master_plan.md` first**, then `SETUP.md` §8 — that section describes an
> open finding that changes the central claim and needs independent
> re-verification. **These files supersede anything older**; several earlier
> numbers turned out to be stale or artifacts, and re-importing them has caused
> real problems.
>
> **What I need from you:**
>
> 1. **Verify every result before it enters the thesis.** I run everything on my
>    own machine via Claude Code and paste output here. Numbers printed in chat
>    are not evidence. Never substitute a stand-in for missing data — stop and ask.
>
> 2. **Write the thesis as a story.** My supervisor's instruction: a first-year
>    graduate student should follow it top to bottom. Per chapter: physical
>    question → every symbol defined → intuition (ChemE analogies where natural —
>    I come from reaction kinetics and transport) → mathematical rigour → result
>    → why it matters and how it connects to the next chapter.
>
> 3. **Maintain the backlog.** Results graduate 💡 → 📐 → 💻 → ▶️ → ✅. Nothing
>    enters the thesis before ✅. Add ideas the moment they surface.
>
> 4. **Two review stances, which I'll invoke by name.** *"Skeptic pass"*:
>    assume the result is wrong and try to break it — units, limits, signs, what
>    would refute it, does it survive a different definition or norm, what would
>    you attack as the examiner. Do not be reassuring. *"Publication pass"*:
>    judge honestly whether this is publishable and where, whether the claim is
>    supported or overstated, what a referee would reject.
>
> **Method — hold me to this:** one quantity per session; physics → my hand
> derivation → worked example → brutal physics test → code → written note.
> **I derive, you ask and wait** — don't derive it for me and ask me to confirm.
> Ground truth: physics → math → code → documents; documents are never
> authorities, including yours and mine. No black boxes. Push back hard when I'm
> wrong.
>
> **Status:** Q1–Q4 ✅, Q4b ✅, Q4c ✅, Q5 Sessions B–D ✅. Verified at ITER
> reference ($T_e=2.947$ eV, $n_e=1.389\times10^{14}$ cm⁻³): $\tau_{\rm QSS}=22.73$
> µs, $\tau_{\rm relax}=2.277$ ns, $M=9982$.
>
> **First task: work through the `SETUP.md` §8.3 re-verification checklist**,
> especially item 4 (backlog C11) — whether a QSS breakdown regime exists
> anywhere on the grid. That question decides whether this thesis is a breakdown
> demonstration or a validity map.
>
> Start by reading the master plan and the backlog, then tell me what you think
> is wrong with the plan or missing from it.

**Check:** it should read the files before answering, know the reference values,
and — ideally — push back on something. A fresh reading often catches what we
have stopped seeing.

### Step 7 — ChatGPT in the loop

No setup needed. When a result matters, paste the **raw Claude Code output**
cold to both models, before either has seen the other's interpretation. See §4.5
on why disagreement is the useful signal and agreement is not.

---

## 3. The working loop

```
   1. DERIVE          chat — you on paper, Claude asks and waits
        │
   2. PREDICT         chat — write down the expected answer BEFORE computing,
        │                    and what result would refute the claim
        │
   3. RUN             Claude Code — on your machine, your data
        │
   4. CROSS-CHECK     paste the RAW output to both chat models, cold
        │
   5. RESOLVE         you arbitrate; disagreement is the signal
        │
   6. RECORD          update the backlog entry and its graduation state
```

**Step 2 is the one people skip and the one that matters most.** A prediction
made before the computation turns the run into a test. A number produced with no
prior expectation can only be rationalised, never falsified.

### Prompt templates

**Claude Code:**

> Read `outputs/master_plan.md` and `outputs/thesis_grade_results_backlog.md`
> for context. Run `src/validation/verify_ramp_vs_step.py` with the real
> `S_grid.npy`. I need Test 3: the FULL/scalar ratio column. Report only — do
> not modify anything, do not substitute a stand-in source if S_grid.npy is
> missing, stop and tell me instead. State which files it loaded and whether the
> reference values reproduce.

**Both chat models, given the raw output:**

> Here is the raw output of [script] run on my data. Before interpreting: what
> would this output look like if [claim] were FALSE? Then tell me which it
> resembles. Flag any number whose meaning depends on a definition, norm, or
> weighting choice.

---

## 4. Methods that actually reduce error

Beyond "check your work." Each of these caught something real in this project.

### 4.1 Predict before you compute

Write the expected answer down first. This converts a computation into a test.

*Worked here:* you predicted ε_L2 > ε_ratio by common-mode reasoning before we
ran anything; the run gave 31×, and the mechanism (n₃ +18.7%, n₄ +14.4%) was
visible in the numbers. The prediction is what made the confirmation meaningful.

### 4.2 State what would refute it

Before accepting a result, name the observation that would kill it. If nothing
would, the claim is not empirical.

*Worked here:* "if the ℓ-distribution is not statistical at both endpoints, the
A-coefficient cancellation fails" — now backlog C8, a specific falsifier rather
than a vague worry.

### 4.3 Severity, not just consistency

A test that a wrong result would also have passed tells you nothing. Ask: *would
this check have caught the error if the error were present?*

*Worked here:* the QC gates on the CR matrix (conservation, signs, NaN) all
passed on the artifact-laden matrix. They were consistent but not severe. Only
the eigenvector inspection — which a wrong matrix would have failed — was severe.

### 4.4 Sensitivity to definitions

Any result that depends on an arbitrary choice must be tested under the
alternatives. If the sign flips, it is an artifact.

*Worked here:* the boundary descent was tested under three weightings (v²
ground-excluded, |v| ground-excluded, v² ground-included). All gave negative
slopes → the descent is real. But the magnitude moved 4×, which is why **no
fitted exponent is quoted**.

### 4.5 Disagreement is the signal; agreement is weak

Two language models agreeing may share the same training bias. Their
*disagreement* is what locates real uncertainty.

*Track record here, both directions:*

| ChatGPT caught | Claude caught |
|---|---|
| p_G ≈ 17 arithmetic error | its over-trust in the crude n_cr formula |
| figure T_e mismatch (3.24 vs 2.947 eV) | "clustered spectrum" — it is a continuum |
| bundled/resolved visual bias | norm-dependence of transient growth |
| one-channel boundary formula | under-concluding on the staircase |

Neither caught everything. **Feed both the raw output, not each other's
interpretations** — otherwise you get a critique of a model rather than an
independent read.

### 4.6 Provenance on every number

"τ_relax = 2.28 ns" is not a result. "τ_relax = 2.277 ns from eig(L_grid[23,5]),
L_grid regenerated 14 Jul after the F(U_m) fix, reproduced on my machine" is.

*Worked here:* the 25 ns vs 2.28 ns dispute was only resolvable because
provenance existed — one traced to a pre-correction matrix, the other to the
current one.

### 4.7 Record corrections in place; never silently edit

"An earlier draft claimed X; this was an arithmetic error; corrected to Y; the
conclusion that depended on it is retracted." A silently fixed note lets the
same wrong path be re-derived in three weeks.

### 4.8 Units on everything

If a quantity comes out with the wrong dimensions, the formula is wrong — no
further checking needed.

*Worked here:* M as time² instead of dimensionless; τ written as s⁻¹; λ written
in ns. Three separate catches, two of them by you against Claude.

### 4.9 Derive at the rigour of the code you audit

A toy derivation cannot adjudicate full-theory code — any disagreement is
ambiguous between "code bug" and "expected toy/full gap".

*Worked here:* the ℓ-mixing audit cost extra rounds precisely because the
derivation was order-of-magnitude while the code implemented full PSM20.

### 4.10 Separate the claim from the observable

A result can be true in one measure and false in another.

*Worked here:* transient growth is present in L² and **absent** in L¹ and for a
ground-state perturbation. Every claim must state the norm and the perturbation.

---

## 5. The two review stances

Invoke by name in chat. These are *stances*, not separate agents — claude.ai
cannot spawn subagents, and calling them agents would overstate what they are.

### "Skeptic pass"

> Assume this result is wrong and try to break it. Check units, limits, signs.
> What would refute it? Does the conclusion survive a different definition,
> norm, or weighting? Which claims rest on unverified assumptions? What would
> you attack if you were the examiner? Do not be reassuring.

Use before anything is promoted to ✅, and before any claim enters a chapter.

### "Publication pass"

> Judge quality honestly. Is this publishable, and where? Is the central claim
> actually supported, or overstated? What would a referee reject? What is
> missing? How does it sit against the existing literature? Tell me plainly if
> it is not good enough.

Use when a chapter drafts, and before choosing a journal.

**These must be able to return bad news.** A skeptic pass that concludes
"looks good" every time is decorative. Ask directly: *what is the strongest case
that this is wrong?*

---

## 6. Story-first writing

Supervisor's instruction: a first-year graduate student should follow it top to
bottom.

Per chapter:

1. **The physical question** — why anyone should care, before any formalism
2. **Every symbol defined** at first use, with units
3. **Intuition** — ChemE analogies where natural (reaction kinetics, CSTR feed,
   Bodenstein steady state, transport). These are genuine correspondences here,
   not decoration
4. **Mathematical rigour** — the derivation, complete
5. **The result** — with its caveats attached, not in a footnote
6. **Why it matters, and what comes next** — the bridge to the following chapter

**The test:** hand a section to someone with your starting background. If they
cannot follow it unaided, intuition is missing — not detail.

**Leverage:** the derivation notes are already written this way. Chapter 3 is
substantially assembly, not fresh composition.

---

## 7. Backlog discipline

`thesis_grade_results_backlog.md` is the register.

**Graduation:** 💡 Idea → 📐 Derived → 💻 Coded → ▶️ Run → ✅ Verified

**To reach ✅ a result needs:** reproduction on your machine · a sensitivity
check against definitional choices · written caveats · a named thesis home ·
a stated falsifier that did not occur.

**Nothing enters the thesis before ✅.** A result that runs but is not understood
is a black box, and a black box is what produced the −46% Hα artifact.

**Add ideas the moment they surface** — one line under §C is enough. Do not stop
to derive; the point is not to lose it.

---

## 8. The first thing to do in the new chat

### 8.1 What was found on 14 Jul (and must be re-verified there)

A ramp-vs-step analysis was run on the real `S_grid.npy`. It produced a
**correction to the central claim** and a **correction to the error measure**.
Both need independent re-checking before anything is written into the thesis.

**Finding 1 — the $L^2$ error measure over all 43 states is dominated by the
ground state, and therefore does not measure QSS.**

At the benchmark point: ground population $8.80\times10^{-4}$ versus total
excited $1.21\times10^{-5}$ — the ground state is **73×** the entire excited
manifold. An $L^2$ norm over all states is effectively an error measure on
$n_{1s}$.

This matters because **QSS is an approximation about the excited manifold
only** — the ground state and $n_e$ are supposed to evolve freely. Including
$n_{1s}$ in the norm measures the wrong quantity.

Measured consequence (full 43-state integration, LSODA, convergence-checked
across three tolerances spanning $10^{-6}$ to $10^{-11}$ — identical to five
digits, so this is physics and not numerics):

| Drive $\tau_d$ | peak error, ALL states | peak error, EXCITED only | ratio |
|---|---|---|---|
| 1 µs | 1.697 | 0.0555 | 31× |
| 100 µs (ELM crash) | 0.205 | **0.00724** | 28× |
| 1 ms (fast detachment) | 0.0276 | **0.00104** | 27× |

Step-error magnitudes: $J = 1.780$ over all states, $J = 0.333$ excited-only.

**Finding 2 — with the correct measure, QSS is robust at all divertor
timescales.** The excited-manifold error is **0.7% at ELM timescales** and
**0.1% at detachment timescales**.

**Two wrong turns are recorded here deliberately, because both were caught only
by running on real data:**

1. *"The scalar suppression kills the step error, so the breakdown claim is
   unsupported."* — based on the single-mode scalar model, before the full
   system was integrated.
2. *"Non-normality defeats the suppression by 5000×, so the breakdown claim
   survives."* — the 5000× came from comparing the full-system error against a
   scalar prediction built on $\tau_{\rm relax}$, when the ground-dominated norm
   was actually relaxing on $\tau_{\rm QSS} = 22.7$ µs. Using the correct
   timescale, the scalar criterion agrees with the full system to within a
   factor of two.

Both were Claude's errors, both survived plausible-sounding reasoning, and both
died on contact with the student's own data. **This is the argument for the
run-it-yourself rule.**

### 8.2 The consequence for the thesis claim

**Not supported:** *"QSS breaks down in ITER divertor conditions."* With the
physically correct error measure, QSS holds to better than 1% at every
divertor-relevant drive timescale.

**Supported, and a stronger result:**

> QSS validity is governed by $De = \tau_{\rm relax}/\tau_{\rm drive}$, not by
> the timescale ratio $M$. We derive the suppression law
> $\varepsilon_{\rm peak}/J = De\,(1-e^{-1/De})$, verify it against full
> 43-state integration, and establish that QSS remains valid to better than 1%
> throughout the ITER divertor operating range — with a quantified failure
> threshold at $\tau_{\rm drive}\sim40$ ns, roughly 2500× faster than an ELM
> crash.

That is a **validity map with a derived criterion**, where the literature offers
assertion. Fujimoto and Capitelli state that QSS holds; this would establish
*where*, *by how much*, and *what would break it*.

**What survives untouched:** the two-timescale structure (one gap of $10^4$);
the boundary-level descent (A1); the moving-bottleneck density scaling (A5);
$\mu(L)>0$ and transient growth (A3) — still true and still interesting, it
simply is not what breaks QSS; and the ℓ-mixing bug found and bounded (A4).

### 8.3 Re-verification checklist for the new chat

Do these **before** accepting anything in §8.1–8.2. The findings above are
claims, not established results.

1. **Reproduce the ground/excited population ratio** at the benchmark point.
   Expect ground $\approx73\times$ the summed excited population. If not, the
   whole diagnosis is wrong.

2. **Reproduce the excited-only vs all-states error table.** Expect ~28× smaller
   excited-only errors, and ~0.7% at ELM timescales.

3. **Re-run the convergence check** (three tolerances). Expect identical peaks
   to five digits.

4. **Extend to the full grid — NOT YET DONE, and this is the open question.**
   The 0.7% figure is one grid point. $\tau_{\rm relax}$ reaches 38.9 ns at
   $T_e=1$ eV, $n_e=10^{12}$ cm⁻³ — 45× slower than at the reference. **Is there
   a corner of the operating range where the excited-only error becomes large?**
   Map $\varepsilon^{\rm excited}_{\rm peak}$ over all $(T_e,n_e)$ for each
   divertor drive timescale. This either confirms robustness everywhere or finds
   the regime where QSS genuinely fails.

5. **Re-derive the suppression law independently** rather than accepting it:
   $\varepsilon_{\rm peak}/J = De\,(1-e^{-1/De})$ from
   $\dot\delta = -\delta/\tau_r + J/\tau_d$. Check the limits.

6. **Decide which $\tau$ enters $De$ for the excited-only measure.** The
   all-states norm relaxes on $\tau_{\rm QSS}$ (ground-dominated); the
   excited-only norm should relax on $\tau_{\rm relax}$. **Verify this rather
   than assuming it** — mislabelling exactly this is what produced wrong turn 2.

7. **Skeptic pass** on the restated claim before it enters any chapter.

### 8.4 The conversation to have with Prof. Pala

The thesis is titled *"…Quantifying Quasi-Steady-State Breakdown…"*. If the
honest answer is *"QSS is robust in the ITER divertor; here is the validity
boundary and the criterion that predicts it"*, that is a change of framing the
supervisor must be part of. **Raise it early — this week, not in August.**

The result is not weaker for being a validity map. It is more defensible, it is
quantitative where the literature is qualitative, and it comes with a derived
criterion others can apply to their own CR models.

### 8.5 Then reconsider Q6 and Q7

Both were designed to characterise a breakdown that may not occur at divertor
conditions. Q6 (Hα artifact) still needs doing — the stale $-46\%$ number must
come out of Chapter 5 regardless. Q7 (the $S$ criterion) was to explain the
step-error scaling; if the step error is not realised, its role changes.
**Decide after item 4 above settles whether a breakdown regime exists anywhere
on the grid.**
