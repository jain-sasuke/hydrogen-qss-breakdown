# Defence Questions — Adversarial Examiner Pass (Gate 9)

**Written 10 September 2026.** Report only. No chapter, script, data file or
register entry was modified in producing this document. Four computations were
run read-only against `data/processed/cr_matrix/` and
`validation/divertor_map/divertor_map.csv`; each is named where it is used and
each is reproducible from the commands recorded in §5.

**Read before this:** `outputs/claim_hierarchy.md`, `outputs/pivot_decision.md`,
`outputs/findings_10_four_agent_review.md` incl. ADDENDA A–D. This document does
not repeat what those establish. Where an attack is already known, it is marked
**[known]** and only the *examiner-facing* form and the answer are given. Where
it is new, it is marked **[NEW]** and carries its own evidence.

**Standing:** the thesis under attack is the one that compiles today —
`thesis_main.tex` inputs chapters 1, 2, 3, 5, 6 only. Title on the title page:
*The Cost of Assuming Ionisation Balance — Transient Error in Balmer-Ratio
Diagnostics of Divertor Hydrogen Plasmas.*

---

# PART 0 — Ranking by damage

| # | Attack | Grade | Status | Is there a good answer? |
|---|---|---|---|---|
| **A1** | Neutral transport destroys the census **inside** the defensible $T_e\ge2$ eV scope, not only at the cold corner | **FATAL** | **[NEW]** | No. Scope or claim must change |
| **A2** | $\varepsilon$ is measured against *this model's own* CRE state. Gate D fails 400/400 and no ADAS PEC file exists in the repository. The thesis has never shown its CRE ratio agrees with any real lookup table | **FATAL** | **[NEW]** | Partial only. A comparison must be produced or the claim reworded |
| **A3** | The $n_e$-fixed dilemma. Both horns cannot be held, and the transport horn is *not* cured above 2 eV | **FATAL→MAJOR** | [known], sharpened | Yes for $T_e\ge2$ eV *only if* A1 is answered |
| **A4** | Molecules, and the fact that H$\alpha$ is the single worst Balmer line to have chosen for the regime invoked | **MAJOR** | [known] + **[NEW]** on the pair choice | Partial. D.4 bound exists; the pair choice does not |
| **A5** | The steady inversion is **non-unique**: $R(T_e)$ is two-valued on 5 of 8 density columns of the model's own grid | **MAJOR** | **[NEW]** | Yes, and it strengthens the thesis if stated first |
| **A6** | The thesis reports the error in $R$ and never converts it to the error in $T_e$, which is 2.3× larger; and 25 of the 45 breakdown pairs masquerade as legitimate steady states | **MAJOR** | **[NEW]** | Yes. This is a result, not a wound — if it is *in* the thesis |
| **A7** | Atomic-data uncertainty is never propagated, while the bound is claimed for "every atomic dataset" | **MAJOR** | [known] | Partial. The bound's data-independence is structural; the *maps* have no error bars |
| **A8** | The name. The registered title named QSS; the work shows QSS is exact | **MAJOR** | [known] | Yes — but it is a supervisor conversation, not an answer |
| **A9** | Why this temperature step, and does it survive 1 / 5 / 20 % | **MAJOR** | [known] | Yes, via $G$ — with one gap at 20 % |
| **A10** | Is $M$ sufficient or descriptive | MAJOR | [known] | Yes, and it is the thesis's best negative result |
| **A11** | What exactly is QSS breakdown if the excited levels stay QSS | MAJOR | [known] | Yes. This is the reversal |
| **A12** | Radiation trapping | MINOR above 2 eV, **MAJOR** below | [known] | Yes above 2 eV; no below |
| **A13** | Is it just a nonlinear equilibrium response | MINOR | [known] | Yes, cleanly |
| **A14** | Does the PE state correspond to a realisable plasma | MINOR | [known] | Yes — and A6 supplies the sharp version |
| **A15** | Shell populations vs $\ell$-resolution | MINOR | [known] | Yes |
| **A16** | Could two reservoir states give the same ratio | MINOR | **[NEW]** verified | Yes, provably no. Answerable in one line |
| **A17** | Why atomic hydrogen only | MINOR | [known] | Weak, and it collapses into A4 |
| **A18** | What experimental timescale would resolve the window | MINOR, but embarrassing | **[NEW]** on the gap | Numbers exist; no instrument is ever named |

**Structural, non-physics, and lethal in a different way** (§4): the abstract
compiles as `[TODO: write last]`; chapters 4 and 7 are not in the document;
all three appendices are empty; eight `\todo` and ten `[UNVERIFIED]` blocks
render in red inside the body text.

---

# PART 1 — The attacks nobody has made

This is where the value of this pass is. §2 handles the seventeen the supervisor
listed; four of them turn out to have sharper forms than the ones written down,
and those sharper forms are here.

---

## A1 — Neutral transport kills the result *inside* its own defensible scope

**[NEW]. Grade: FATAL as the claim is currently framed.**

### The attack

Chapter 6 §`sec:transport` treats neutral transport as a threat to the *cold
edge*. Its whole quantitative content is about $[0,4]$: $\tau_{\rm QSS} =
\SI{233}{ms}$ against a 6–10 µs transit, and "an effective ground-state renewal
time of about \SI{26}{\micro\second} would suffice" to drop that point below
10 %. `findings_10` ADDENDUM D.5 then *rescues* it: with charge exchange the
neutrals are trapped, the escape time is 72 µs rather than 6–10 µs, and "the
result survives, by a factor 2.8".

**That factor 2.8 was computed at one point, and it is the wrong point.** It is
the cold corner — which Chapter 5 §`sec:scope` and Chapter 6 both declare
outside the model's domain. Nobody has run the same calculation at the point the
thesis actually stands on.

I ran it. Scaling ADDENDUM D.5's own CX-diffusive model — its implied
$\sigma_{\rm cx} = 1.12\times10^{-14}$ cm², $\lambda = 1/(n_p\sigma_{\rm cx})$,
$D = v\lambda/3$, $\tau_{\rm esc} = L^2/\pi^2D$, reproducing D.5's 72 µs anchor
exactly — at $[15,3]$, $T_e = 2.02$ eV, $n_e = 1.93\times10^{13}$, the worst
point above 2 eV and the source of the headline 17.5 %:

| point | $\varepsilon_{\rm plateau}$ | $\tau_{\rm QSS}$ | renewal time needed to fall below 10 % | CX-diffusive escape, $L = 20$ cm | margin |
|---|---|---|---|---|---|
| $[0,4]$ cold corner | 0.3869 | 233 ms | 26.4 µs | 72.2 µs | **2.7×, survives** |
| **$[15,3]$ worst above 2 eV** | **0.1807** | **1.49 ms** | **75.3 µs** | **18.9 µs** | **0.25×, fails** |
| $[15,4]$ | 0.1562 | 444 µs | 103 µs | 50.7 µs | 0.49×, fails |
| $[15,5]$ | 0.1035 | 127 µs | 1425 µs | 136 µs | 0.10×, fails |

The cold corner survives *because* $\varepsilon_{\rm plateau}$ is enormous there
(0.387) and $\tau_{\rm QSS}$ is 233 ms. At the defensible point $\varepsilon$ is
half as large and $\tau_{\rm QSS}$ is 156× shorter, so the renewal time it can
tolerate is three times longer while the actual escape time is four times
shorter. The margin inverts.

Recounting the whole $T_e\ge2$ eV census with
$1/\tau_{\rm eff} = 1/\tau_{\rm QSS} + 1/\tau_{\rm esc}$:

| assumed slab | pairs | breakdown as published | breakdown with CX transport | worst |
|---|---|---|---|---|
| $L = 10$ cm | 448 | 45 | **0** | 0.0271 |
| $L = 20$ cm | 448 | 45 | **0** | 0.0632 |
| $L = 50$ cm | 448 | 45 | 7 | 0.1206 |

**The headline result of the thesis — "45 of 448 pairs above 2 eV exceed 10 %,
worst 17.5 %" — goes to zero for any slab thinner than about 30 cm.**

### Why this is worse than the version already in Chapter 6

Chapter 6 presents transport as the objection that "cannot be answered inside
this model", and it is right about that. But it then confines the damage to the
cold edge, and Chapter 5 §`sec:scope` builds the entire defence of the
$T_e\ge2$ eV boundary on two *other* processes — quasineutrality and Lyman
trapping — both of which do switch off above 2 eV. **Transport does not switch
off above 2 eV. It gets worse**, because $\tau_{\rm QSS}$ falls by two orders of
magnitude across that boundary while the neutral escape time falls by less than
one.

The thesis has therefore chosen its scope boundary using the two objections that
the boundary cures, and has quantified the third only where it happened to
survive.

### The answer the candidate should have ready

There is no answer that saves the claim as written. The candidate should give
this one, which is honest and still leaves a thesis:

> The zero-dimensional box has no transport, and the result is conditional on
> the ground-state neutral population being frozen for the duration of the
> event. That conditional was stated for the cold corner and I did not carry it
> to the defensible range, where it binds harder. Applying the charge-exchange
> diffusive estimate of the audit at $[15,3]$ gives an escape time of 19 µs
> against the 75 µs the point can tolerate, so the census of 45 does not survive
> the objection. What survives it is the part of the result that is not a
> census: the closure is exact to $10^{-8}$; the sensitivity $\bar S$ and the
> reservoir gain $G$ are properties of the operator and are unchanged by
> transport; and the bound $|d\ln R/d\ln b_1| < 1$ is a theorem. What transport
> changes is the *displacement* $\Delta\ln u$ — it replaces $\tau_{\rm QSS}$ with
> a shorter renewal time in exactly one place in Eq. (recipe), and a reader with
> a transport code can put their own number there. The map of where the
> coefficients are large stands; the claim that 45 specific points break down in
> a real divertor does not.

### What must change

**The evidence, or the claim — not the scope.** Widening the scope does not
help, because the problem is worst inside it. Three options, in order of cost:

1. **Reframe the census as conditional and put the conditional in the sentence,
   not in Chapter 6.** "45 of 448 pairs exceed 10 % *for a plasma parcel whose
   neutral population is renewed no faster than $\tau_{\rm QSS}$*" — and then
   state the renewal time each point can tolerate, which is a new column and
   costs one afternoon. That column is more useful than the census.
2. **Replace the census with the renewal-time map.** For each point report
   $\tau_{\rm renew}^{\rm crit}$, the ground-state renewal time at which the
   ELM-averaged error passes 10 %. It is a property of the operator, it is
   transport-agnostic, and it lets a SOLPS user answer the question directly.
   This is the same move the pivot already made from $\varepsilon$ to $G$.
3. Do the two-region calculation. Out of scope before September.

**A note on the estimate itself.** My $\tau_{\rm esc}$ is crude: it inherits
D.5's $\sigma_{\rm cx}$, assumes $T_n = T_e$, and at $[15,3]$ the mean free path
is 4.6 cm against a 20 cm slab, so the diffusive limit is marginal. The
free-streaming transit there is 7.2 µs — *ten times shorter* than the diffusive
estimate, which makes the attack stronger, not weaker. There is no assumption in
this calculation under which the point survives at $L\le20$ cm.

---

## A2 — The error is measured against the model's own CRE, never against any table

**[NEW]. Grade: FATAL to the framing; survivable by rewording.**

### The attack

The thesis claim is about a *lookup table*: "a table indexed only on $(T_e,n_e)$
silently assumes equilibrium ionisation balance, so during that window it
assigns the wrong physical meaning to a correctly measured ratio." Chapter 1
§`sec:inversion` names the table: ADAS, `\cite{Summers2006, ADAS}`.

But $\varepsilon$ is defined as the distance from the partial-equilibrium state
to **this model's own** collisional–radiative equilibrium state. So the quantity
computed is the distance between two states of one 43-state matrix. For that to
be the error of a diagnostic, this model's CRE line ratio must agree with the
table a practitioner uses. **That has never been shown, and three independent
facts say it may not.**

1. **`validation/gate_summary.txt` reads `Gate D: FAIL (0% of points)`.** The
   one external gate against ADAS gives
   $\eta = {\rm SCD_{model}}/{\rm SCD_{ADAS}} \in [5.73, 8346]$, and **0 of 400
   points agree within a factor 2**. `validate_gates.py` returns exit code zero
   regardless.
2. **There is no ADAS PEC file in the repository.** `find data -iname "*pec*" -o
   -iname "*adf15*"` returns nothing. The one place a Balmer-ratio comparison is
   claimed — `data/processed/adas/report_adas.md`, "H$\alpha$/H$\beta$ = 4.1
   against ADAS PEC 2.8–4.5" — rests on a file that does not exist and has no
   producing script (`claim_hierarchy` §A.8). **`chapter3.tex:1377-1387` promises
   this comparison to the reader.**
3. **Chapter 2 §`sec:atoms_scope` concedes the point itself:** "the effective
   coefficients this model produces *cannot be compared directly with those
   tabulated by ADAS*. No ADAS dataset enters the rate matrix at any point."

The examiner's sentence is: *"You have quantified the error of your own model
against your own model. Show me that your equilibrium ratio is the one in the
table you say is wrong."*

### The answer the candidate should have ready

It is a good one, and it needs to be *in the thesis*, not improvised:

> Two of those three are answerable and the third is a real gap. Gate D's
> failure is diagnosed, not mysterious: `SCD_model` sums ionisation over the
> full steady state, while ADAS SCD is the *ionizing* coefficient, and the
> `acd_adas` array is loaded and never used, so the recombination half was never
> implemented. It is reported as a failure and not made to pass, which is the
> correct handling of a definitional mismatch. Chapter 2 is right that the
> *effective coefficients* of an open model are not ADAS-comparable, because
> ADAS closes the system and this one does not.
>
> But $\varepsilon$ does not depend on the effective coefficients. It is a
> ratio of ratios, $|R^{\rm PE}/R^{\rm CRE} - 1|$, and the normalisation of the
> population vector cancels exactly — that is verified: $\Delta$ and the tanh
> bound are invariant under rescaling the $n_g/n_i$ normalisation by factors of
> $10^{10}$ and $10^{-7}$. What would have to agree with ADAS is the *shape* of
> $R^{\rm CRE}(T_e,n_e)$, not its scale, and the honest position is that this
> has not been checked against a PEC file.

**And then the candidate must say what he will do about it**, because the
examiner will ask. The cheapest sufficient answer is one figure: this model's
$R^{\rm CRE}(T_e)$ at $n_e = 10^{14}$ overlaid on the same curve built from
an ADAS adf15 H PEC set. If the shapes agree to within the 10 % threshold the
thesis uses, the framing survives intact and the thesis gains its only genuine
comparison to a diagnostic table. If they do not, $\varepsilon$ is a statement
about this model and the abstract must say so.

### What must change

**The evidence.** One `adf15` file and one figure. Until then:

- **Delete `chapter3.tex:1377-1387`**, which promises a comparison that does not
  exist. This is already on the `claim_hierarchy` PART 6 deletion list and it is
  now the *most* dangerous of the four, not the least.
- **Delete the 4.1 vs 2.8–4.5 sentence** wherever it survives; it is
  unsupported by any file in `data/`.
- Report Gate D's failure in the body with its diagnosis, and state in one
  sentence why $\varepsilon$ is insensitive to it.

---

## A5 — The inversion the thesis criticises is already non-unique

**[NEW]. Grade: MAJOR. Handled first, it is a gift; handled second, it is a wound.**

The supervisor's list asks *"Is the inversion unique?"* and the brief records
that it "has never been examined at all". It has now. There are two separate
questions inside it and they have opposite answers.

### (a) Could two different reservoir states produce the same line ratio? **No, provably.**

Within the two-channel split $n_p = a_p u + c_p$, the ratio is a Möbius function
of the reservoir variable,
$$R(u) = \frac{a_3 u + c_3}{a_4 u + c_4},$$
which is strictly monotone in $u$ unless $a_3c_4 = a_4c_3$, i.e. unless
$f_3 = f_4$. Measured over all 784 (point, direction) pairs of
`divertor_map.csv`: $f_3 - f_4$ is **positive at every one of them**, minimum
$0.0402$ at $[42,7]$. So at fixed $(T_e, n_e)$ the map $u \mapsto R$ is a
bijection and the reservoir state is uniquely recoverable from the ratio.

**This is a positive result and it belongs in Chapter 5 next to the bound.** It
says the *forward* problem this thesis solves is well-posed, and it is one line
of algebra plus one grep.

### (b) Is the steady $(T_e)$ inversion unique? **No — and it fails on the model's own grid.**

Solving $L\mathbf n + \mathbf S = 0$ at every grid point and forming the shell
ratio $R = n_3/n_4$, $R$ is **not monotone in $T_e$**. It has an interior
minimum on five of the eight density columns:

| $n_e$ (cm⁻³) | $R$ minimum | at $T_e$ | $R$ at 10 eV | degenerate band in $R$ | example: same $R$ at |
|---|---|---|---|---|---|
| $1.00\times10^{12}$ | 0.84621 | 7.91 eV | 0.84759 | 0.16 % | 6.53 eV **and** 10.00 eV |
| $2.68\times10^{12}$ | 0.90957 | 5.96 eV | 0.92013 | 1.16 % | 4.02 eV **and** 10.00 eV |
| $7.20\times10^{12}$ | 0.93905 | 5.43 eV | 0.95976 | **2.21 %** | **3.31 eV and 10.00 eV** |
| $1.93\times10^{13}$ | 0.89079 | 5.96 eV | 0.90818 | 1.95 % | 3.69 eV **and** 10.00 eV |
| $5.18\times10^{13}$ | 0.79345 | 7.20 eV | 0.80006 | 0.83 % | 5.11 eV **and** 10.00 eV |
| $1.39\times10^{14}$ | 0.70774 | 9.10 eV | 0.70813 | 0.06 % | 8.27 eV **and** 10.00 eV |

At $n_e = 7.2\times10^{12}$ the ratio $R = 0.9598$ is produced by a plasma at
**3.31 eV and by a plasma at 10.0 eV** — a factor 3.0 in temperature — and
nothing in the ratio distinguishes them.

The thesis already owns half of this. Chapter 7 §`sec:implications` records that
$\varepsilon_{\rm step}$ "falls to $4.9\times10^{-5}$ near \SI{6.9}{\electronvolt}"
and concludes that "a bounded error in the observable produces an *unbounded*
error in the inferred temperature", and admits "its locus across the rest of the
grid has not been mapped". That is the *same* turning point, seen through its
absolute value. The map is above; it takes one solve loop.

**But Chapter 7 stops one step short of the stronger and simpler statement:**
a stationary point of $R(T_e)$ is not merely an amplifier, it is a **fold**. The
ratio is two-valued. The table cannot assign a unique meaning to a measured
ratio in that band *at all*, transient or not, and the degeneracy band is 0.06 %
to 2.2 % wide in $R$ — smaller than the 10 % threshold the thesis uses and
comparable to the ~5 % combined ADAS-and-atomic-data uncertainty that threshold
was matched to.

### The answer the candidate should have ready

> The reservoir inversion is unique and I can prove it: $f_3 - f_4 > 0$ at all
> 784 pairs, minimum 0.040, so $R(u)$ is a strictly monotone Möbius map. The
> temperature inversion is not unique, and that is a property of the steady
> table, not of my transient. $R(T_e)$ has a turning point between 5.4 and
> 9.1 eV on five of eight density columns; at $7.2\times10^{12}$ the same ratio
> is returned by 3.3 eV and by 10 eV. Chapter 7 reports the amplification at
> 6.9 eV and I did not recognise it as a fold. It sharpens the thesis rather
> than weakening it: the table is ambiguous in a band narrower than its own
> quoted uncertainty *before* any transient is applied, and my result is the
> additional error on top of that.

### What must change

**The claim, and it gets stronger.** Add a subsection to Chapter 5 or 7:

- State (a) as a well-posedness result for the forward problem.
- Map the locus of $\partial\ln R/\partial\ln T_e = 0$ across the grid — one
  solve loop over the existing `L_grid` — and plot it on the same axes as the
  ridge. Chapter 7 already says this "has not been mapped" and it is cheap.
- State the fold, with the degeneracy band width, and say plainly that in that
  band a single Balmer ratio does not determine a temperature. Chapter 1
  §`sec:inversion` already tells the reader "one ratio is one constraint on two
  unknowns"; this is the sharper version of the same honesty.

---

## A6 — The error is reported in the wrong variable, and most of it is invisible

**[NEW]. Grade: MAJOR. This is a result the thesis is currently throwing away.**

### The attack

Every headline number in the thesis is an error in $R$: 17.5 %, 38.7 %, 7.2 %,
the 10 % threshold. **Nobody measures $R$ in order to know $R$.** The physicist
of Chapter 1 measures $R$ in order to write down a temperature. The quantity she
cares about is the error in $T_e$, which is
$\varepsilon_R / |\partial\ln R/\partial\ln T_e|$ — and since
$|\partial\ln R/\partial\ln T_e| \le 0.64$ everywhere on the grid and passes
through zero at the fold of A5, the error is *always* amplified and sometimes
unboundedly.

Inverting the emitted partial-equilibrium ratio against the model's own CRE
table at the same density:

| point | true $T_e$ after step | $\varepsilon$ in $R$ | $T_e$ the table returns | apparent $T_e$ error |
|---|---|---|---|---|
| $[15,3]$ worst above 2 eV | 2.121 eV | 18.1 % | **1.273 eV** | **−40.0 %** |
| benchmark $[23,5]$ | 3.089 eV | 6.4 % | 2.319 eV | −24.9 % |
| $[0,4]$ cold corner | 1.048 eV | 38.7 % | *no solution* | — |

Over the 45 breakdown pairs above 2 eV:

- **25 of 45** produce a ratio that lies **on** the CRE table. The diagnostic
  returns a single, unambiguous, entirely plausible temperature. There is no
  residual, no failed fit, no warning of any kind. Median apparent $T_e$ error
  **−40.0 %**, range −43.2 % to +117.3 %.
- **20 of 45** produce a ratio **off** the table — no solution exists at that
  density. These are detectable, and a practitioner would notice.

So the thesis's own worst case, stated in the variable the user works in, is a
factor 1.7 in temperature, not 18 %; and *more than half of it is silent*.

### Why this is an attack and not just an omission

Because the examiner will read "17.5 %" next to the sentence about combined
ADAS-and-atomic-data uncertainty of ~5 %, conclude the effect is three times the
noise floor, and move on — and the candidate will have undersold his own result
by a factor 2.3 while leaving open the reply *"17 % on a line ratio is within
the error bar of the atomic data you just told me about."* Stated in $T_e$, that
reply is unavailable: nobody claims a 40 % temperature uncertainty from atomic
data.

### The answer the candidate should have ready

> Reporting in $R$ was the conservative choice and it costs a factor 2.3. The
> ratio error propagates to temperature through $\partial\ln R/\partial\ln T_e$,
> which never exceeds 0.64 in magnitude on this grid, so the temperature error
> is always the larger number: at the worst defensible point an 18 % error in
> the ratio is a 40 % error in the inferred temperature, from 2.12 eV to
> 1.27 eV. And 25 of the 45 breakdown pairs land on a valid point of the table,
> so the inversion succeeds and returns a wrong answer with no diagnostic
> signature. That is the practically important half of the result and I report
> it in the wrong variable.

### What must change

**The evidence, added; then the claim, strengthened.** One extra column in the
census table and one paragraph. The computation is fifteen lines against the
existing `L_grid` and `divertor_map.csv`. It should carry the on-table /
off-table split, because "the failure is undetectable in 56 % of cases" is the
sentence that makes a diagnostician care.

---

## A4b — H$\alpha$/H$\beta$ is the pair that detachment spectroscopy avoids

**[NEW] as an attack; the molecular problem itself is [known]. Grade: MAJOR.**

Chapter 1 justifies the choice: they are "the two brightest", the visible is
"convenient for windows, fibres and detectors", and "both end on $n = 2$, so the
ratio does not depend on anything about $n = 2$". Every one of those is a reason
of *convenience*, and none is a diagnostic argument.

The diagnostic argument runs the other way. The literature the thesis itself
cites — Verhaegh, Wijkamp — attributes **60–70 % of the $D_\alpha$ light** at
detachment onset to molecular channels, against 10–20 % of $D_\gamma$. That is
precisely why the practitioners working in the regime this thesis invokes use
the **high-$n$ Balmer lines**, $n = 5$ to $9$, whose emission is
recombination-dominated and molecule-poor. The thesis has picked the one Balmer
line in the series that those practitioners deliberately discard, and then
reported that its model contains no molecules.

The two facts compound. `findings_10` ADDENDUM D.4 constructs the bound —
$\phi_3 = 0.60$, $\phi_4 = 0.30$ reduces the cold-corner $f_3 - f_4$ by a factor
2.7 to 5.2 — and it is largest *because* $\phi_3 \gg \phi_4$, i.e. because the
chosen pair straddles the steepest part of the molecular-contamination gradient.
A pair further up the series would have a smaller correction *and* a smaller
uncertainty on the correction.

### The answer the candidate should have ready

> H$\alpha$/H$\beta$ was chosen for brightness and for the shared lower level,
> which makes the ratio independent of $n = 2$ and of the absolute calibration.
> The molecular objection is real and I have bounded it rather than only
> admitting it: taking the published emissivity fractions, the sensitivity falls
> by a factor 2.7 to 5.2 at the cold corner. What I have not done, and should,
> is repeat the map for a higher pair. The framework is pair-agnostic — $\bar S$
> and $G$ are computed for any $(p,q)$ — and §7.3 already reports that the
> $(3,5)$ pair moves the worst density by a factor 2.68, so the machinery
> exists. A $(6,7)$ or $(7,8)$ map would be both more relevant to detachment
> spectroscopy and less exposed to the molecular correction.

### What must change

**The scope, or one more run.** Either the thesis states plainly in Chapter 1
that it treats the $(3,4)$ pair *because* it is the textbook case and not
because it is the one used in detachment analysis — which is honest and cheap —
or it produces the $(p,q)$ map for one high-$n$ pair, which is the same script
with two indices changed and would answer the objection outright. The second is
worth more than any remaining item on the `claim_hierarchy` "what X still
needs" lists.

---

## A18 — No instrument is ever named

**[NEW] as a gap. Grade: MINOR, but it is the last question of a viva and it
should not be the one that goes badly.**

Chapter 7 has the timescales: a half-life of **12.95 µs** at the benchmark and
**1.10 ms** at $[15,3]$, against $\tau_{\rm relax} = 2.28$ ns, and the
conclusion "a \SI{100}{\micro\second} exposure does not integrate the error
away". Those are the right numbers.

What is absent, everywhere in the document: any named spectrometer, camera,
machine, campaign, or achievable exposure. There is no statement of what
integration time a divertor Balmer system actually runs at, no comparison to the
ELM duration on any existing device, and no proposed measurement. Chapter 7's
"further work" proposes four *calculations* and zero experiments. An examiner
asking "what would falsify this?" gets, from the document as written, "another
calculation".

### The answer the candidate should have ready

> The window to resolve is set by $\tau_{\rm QSS}$, which is 22.7 µs at the
> benchmark and 1.5 ms at the worst defensible point, and by the transient that
> drives it, which for an ELM is 100 µs. So the requirement is an exposure short
> compared with $\tau_{\rm QSS}$ and a repetition able to follow the ELM — of
> order 1–10 µs at the benchmark, and comfortably 100 µs at $[15,3]$. Divertor
> Balmer systems on existing devices run at exposures in that range, so the
> prediction is testable now rather than in principle: after an ELM the inferred
> temperature from a Balmer ratio should undershoot by tens of percent and
> recover on $\tau_{\rm QSS}$, with the recovery time itself scaling as
> $1/K_{\rm ion}(1s)n_e$ — that scaling is the falsifiable part, because it is a
> prediction about the *shape* of the recovery and not about its size.

**That last clause is the important one and it is not in the thesis.**
$\lambda_0 = 2.286 \times K_{\rm ion}(1S)\,n_e$ is already established
(`thesis_ready.md` A1). It converts the result from a bound into a signed,
scaling, time-resolved prediction — the only genuinely falsifiable statement the
thesis is in a position to make about a real machine.

### What must change

**The claim, extended.** One paragraph in Chapter 7 stating the predicted
recovery signature and its $n_e^{-1}$ scaling, and two sentences with a citation
naming the exposure times divertor Balmer systems achieve. That is the whole
repair, and it is the difference between a thesis that ends in a calculation and
one that ends in a measurement someone could go and make.

---

## Smaller new ones, recorded so they are not found by the examiner first

| | Attack | Grade |
|---|---|---|
| **A19** | **Two-photon $2s$ decay is not in the matrix.** Chapter 2: "Two-photon $2s$ decay, of order \SI{10}{\per\second}, is not included." But ADDENDUM D.2 rests its *ℓ-mixing-independent* bound on it — "the two-photon rate 8.229 s⁻¹ alone is 553× faster than $|\lambda_0|$". That bound is a statement about a physical atom, not about the model, and if quoted as validation of the partition it is quoting a rate the matrix does not contain. Say which. | MINOR |
| **A20** | **The model is hydrogen; the transport numbers are deuterium.** Chapter 6 computes transit for "a deuterium atom at 1–3 eV" while every rate in the matrix is $^1$H with infinite nuclear mass. The transit time differs by $\sqrt2$ and the $\ell$-mixing rate by the reduced mass. Immaterial numerically, sloppy in a viva, and trivially fixed by one sentence. | MINOR |
| **A21** | **Line-of-sight integration is never raised.** A spectrometer views a chord through steep gradients; the thesis compares a 0-D point value to a chord-integrated measurement. Since $\varepsilon$ varies by a factor 8 across one decade in $n_e$, chord-averaging is not a small correction and it is not in the limitations chapter at all. | MINOR–MAJOR |
| **A22** | **Impurities appear nowhere** in Chapters 6 or 7 — no C, Be, W, N or Ne, no impurity radiative cooling, no impurity-driven recombination. In a seeded detached divertor this is not a minor omission and it is the one limitation that is not even admitted. | MINOR |
| **A23** | **The "10 % threshold matched to ~5 % ADAS + ~5 % atomic-data uncertainty" is now inconsistent with Chapter 2's own benchmark.** Chapter 2 measures the CCC data against Anderson RMPS at **mean 29.0 % absolute error, 42.4 % within 20 %**. A threshold justified by a 5 % data uncertainty cannot survive next to a table reporting 29 %. See A7. | MAJOR |

---

# PART 2 — The seventeen

Each: the strongest form of the attack, the verdict, the answer, and — where the
answer is not enough — what must change.

---

## Q1. Why is the chosen temperature step physically meaningful?

**Grade: MAJOR. [known] — `pivot_decision` §2 item 1 already retires it.**

**The attack.** It is not physically meaningful, and the thesis knows it. The
+4.81 % step exists because someone chose 50 logarithmically spaced points
between 1 and 10 eV. $\varepsilon_{\rm plateau}$ is linear in it. "38.7 %" means
"38.7 % per 4.81 % step" and is meaningless quoted bare. The examiner's follow-up
is immediate and there is no third answer: *what if the ELM is 10 %?* — 87.4 %.
*Twenty percent?* — 190.4 %.

**The answer.**

> The step size is a grid artifact and the percentage derived from it is not a
> property of the plasma. That is why the chapter does not headline it. The
> step-independent object is the reservoir gain $G = d\ln u/d\ln T_e$, and the
> error is $\varepsilon = |\exp(\bar S G \Delta\ln T_e) - 1|$ for any step the
> reader cares about. Over 736 (direction, point) triples carrying $k = 1, 2$ and
> 4, the median spread in $|G|$ across a fourfold change of step is 5.6 % and
> the maximum is **6.74 %**, while $\varepsilon$ at the same points changes by
> up to a factor **8.44**. Every $\varepsilon$ in the thesis carries its step
> size; the maps that carry the physics are $\bar S$ and $G$.

**What must change:** nothing further — provided the chapters actually execute
the pivot. **Verify before submission** that no bare percentage survives.
`pivot_decision` §4 requires it and `claim_hierarchy` PART 6 item 4 makes it the
one architectural change that repairs a broken chain.

---

## Q2. Does the result survive 1, 5 and 20 percent perturbations?

**Grade: MAJOR. [known], with one live gap.**

**The attack.** The candidate will want to answer "yes, because $G$ is stable".
Press on the arithmetic. $G$'s 6.74 % stability was measured over $k = 1, 2, 4$,
i.e. steps of **4.81 %, 9.85 % and 20.68 %** — so 5 % and 20 % are inside the
measured range and **1 % is not**. Worse, the linearisation
$\varepsilon \approx |f_3-f_4||\ln x|$ was validated only over
$|\ln x| \in [0.124, 0.682]$, and the four-interval numbers sit **3.8× beyond**
that ceiling. So the thesis has a validated answer at 5 %, an exact-solve answer
at 20 % that sits outside its own linearisation window, and no measurement at
all at 1 %.

**The answer.**

> At 5 % and 20 % the answer is measured: $|G|$ moves by at most 6.74 % across
> that fourfold range while $\varepsilon$ moves by up to a factor 8.44, which is
> exactly why $G$ and not $\varepsilon$ is what the chapter reports. At 1 % the
> answer is the small-step limit itself — $d\varepsilon/d\ln T_e = |\bar S G|$,
> which runs from 0.362 to 3.535 over the $T_e\ge2$ eV set, so a 1 % excursion
> costs at most about 3.5 % in the ratio and about 1.3 % at the benchmark. The
> exact form must be used at 20 %, not the linearisation: the four-interval
> numbers are exact linear solves and any sentence explaining them through
> $|f_3-f_4|\,|\ln x|$ is outside where that decomposition was checked.

**What must change:** the *evidence*, cheaply. Run $k = 1$ down to a fractional
step of 1 % — a sub-grid-interval step is a one-line change to the existing
script — so the 1 % end is measured rather than extrapolated. And carry the
"do not use the linearised form" instruction physically next to every
four-interval number, per `findings_10` §8.5.

---

## Q3. Why is $n_e$ held fixed?

**Grade: FATAL as posed; MAJOR after the $T_e\ge2$ eV restriction; and see A1.**

**The attack, in its strongest form.** This is the question the brief flags as
having a quantitative answer *against* the thesis, and it does. Quasineutrality
ties $n_e$ to $n_g$: every ground-state atom that ionises delivers one electron.
At $T_e = 1$ eV the plasma is 95–98 % neutral, so the reservoir collapse that
*produces* the error is the same collapse that *invalidates the operator it was
computed with*. **68 of 392 one-step operators (17 %) require
$|\Delta n_e/n_e| > 10$ %; 36 (9 %) require more than 100 %.** At the cold corner
the required correction is +20.0. The model evaluates the post-step operator at
the old $n_e$ at every one of them.

And the dilemma is genuine, not a technicality:

- **If the parcel is closed**, quasineutrality forces $n_e$ up by 10–20× at the
  cold end and every rate in the post-step operator is wrong by that factor.
- **If the parcel is transport-fed** so $n_e$ genuinely stays fixed, then the
  ground state is being resupplied from outside and is *not stale* — which
  removes the mechanism that produces the error.

**Both cannot hold.** Chapter 5 §`sec:quasineutrality` states this correctly and
in those words.

**The answer.**

> Holding $n_e$ fixed is self-consistent only where the ionisation degree is
> high, and I measured where that is rather than assuming it. Above 2 eV the
> required correction falls below $10^{-3}$ and the question does not arise;
> below it, 68 of 392 operators need more than 10 % and 36 need more than 100 %.
> That is the reason for the $T_e\ge2$ eV restriction, and it is not a hedge: it
> is the region where the model's own assumptions are mutually consistent. It
> coincides, from entirely independent physics, with the boundary at which Lyman
> trapping stops mattering. Below it the cold-corner numbers are reported as the
> asymptotic behaviour of a model outside its own domain, because suppressing
> them would be worse.

**Where the answer runs out — and it does.** The examiner's follow-up is:
*"You escaped the first horn above 2 eV. Show me you escaped the second."*
Above 2 eV, $\Delta n_e/n_e < 10^{-3}$ because the neutrals are a small
perturbation on the **electron** budget. That says nothing about whether the
**neutral** population is resupplied by transport — and A1 shows it is, on
19 µs against a 1.5 ms $\tau_{\rm QSS}$. The two horns are not symmetric and
the $T_e\ge2$ eV restriction only cures one of them.

**What must change:** merge §`sec:transport` and §`sec:open_system` into the
single dilemma, as ADDENDUM D.1 already instructs — and then extend the dilemma
above 2 eV rather than closing it there. See A1 for the repair.

---

## Q4. Why atomic hydrogen only? / Q5. What happens with molecules?

**Grade: MAJOR. [known], with two live problems.**

**The attack.** At $[0,4]$ the model's own equilibrium is **96.6 % neutral**.
Those are conditions where the cited literature attributes 60–70 % of the
$D_\alpha$ light and 10–20 % of $D_\gamma$ to molecular channels. A grep of the
rate assembly for `H2|molecul|MAR|dissociat` returns nothing. Chapter 6 states:
"this work offers no bound on how much, and **none can be constructed from the
data in this repository**."

Two things make this worse than a normal admitted limitation:

1. **The structural point (ADDENDUM B §B.4).** A molecular channel is a *third*
   channel. It does not perturb the two-channel split — it **destroys its
   functional form**. $n_p = a_p n_g + c_p n_i + m_p n_{\rm H_2}$ is not a
   one-parameter family, the logistic is not a logistic, and the $\tanh$ bound is
   a statement about a one-dimensional family of curves that has no reason to
   survive. Chapter 7 concedes exactly this. **So molecules do not reduce the
   result by a factor; they invalidate the derivation.**
2. **Chapter 6 and ADDENDUM D.4 contradict each other in the document.**
   Chapter 6 says no bound can be constructed; D.4 constructs one from the two
   references cited in that same paragraph. Chapter 7 flags the contradiction in
   an `[UNVERIFIED]` block and quotes the bound anyway. An examiner reading both
   chapters finds a thesis arguing with itself.

**The answer.**

> Atomic hydrogen only is a data limit and a scope limit, and it is the reason
> the grid stops at 1 eV. The bound Chapter 6 says cannot be constructed can in
> fact be constructed from the two papers cited in that paragraph: for a given
> upper level the molecular emissivity fraction is the population fraction, and
> molecular-assisted recombination bypasses the ground state, so
> $f_m \to f_m(1-\phi_m)$. Taking $\phi_3 = 0.60$, $\phi_4 = 0.30$ reduces the
> cold-corner sensitivity by a factor 2.7, and $\phi_3 = 0.70$, $\phi_4 = 0.45$
> by 5.2. That sentence in Chapter 6 is wrong and I will correct it.
>
> The honest limit is structural rather than numerical: a molecular channel is a
> third supply channel, and the two-channel split is exact only for two. The
> framework is two-channel **by construction**. Above about 3 eV the molecular
> density is small and the description holds; where it is not small, the
> functional form and not just the coefficient is at risk, and I say so.

**What must change:** **the evidence** — replace "no bound can be constructed"
with D.4's bound, and reconcile Chapter 6 with Chapter 7 so the document stops
contradicting itself. Then **the claim**: state the two-channel-by-construction
limit *where the logistic is derived*, in Chapter 5, not only in Chapter 6.
ADDENDUM B §B.4 calls this "the most defence-dangerous item" and it is still
only in the limitations chapter. And see **A4b** — the pair choice makes this
worse than it needs to be.

---

## Q6. What does radiation trapping change?

**Grade: MINOR above 2 eV — genuinely well answered. MAJOR below.**

**The attack.** At $T_e = 1$ eV, $\tau_{\rm Ly\alpha} = 114$ per cm and the
escape factor over 5 cm is $2.7\times10^{-5}$. The Einstein coefficients in
`L_grid.npy` are wrong, as effective decay rates, **by four to five orders of
magnitude** at the point carrying the headline number.

**The answer, and it is one of the strongest passages in the project.**

> It was computed rather than argued. All 14 Lyman channels, escape factors
> applied self-consistently — $\Theta_P$ depends on $n(1s)$ which depends on
> $\Theta_P$ — converged at all 400 points, swept over slab thickness rather
> than fixed at one, with three validation gates passed before any result was
> read, including one that failed at 1156 % on first run and caught a $4\pi$
> CGS/SI error. The untrapped rebuild reproduces the canonical matrix exactly,
> $\max|{\rm rebuilt} - {\rm canonical}| = 0$, so every trapped number is a
> difference against the real matrix.
>
> Above 2 eV nothing moves: the breakdown count stays at 45 of 448 for $D = 1$,
> 5 and 20 cm and the worst case runs 0.1748, 0.1746, 0.1740 — a change of 0.5 %
> across a twentyfold range in a parameter this zero-dimensional model does not
> contain. That is what makes the $T_e\ge2$ eV boundary a measurement rather
> than a disclaimer. Below it the headline falls by a factor 2.5 to 3.3, to
> 11.6–15.7 %, and which value it takes inside that range is set by an assumed
> slab thickness — so the point still breaks down but the magnitude is not
> quotable. Below 1.15 eV the density column carrying the maximum wanders across
> a factor 7 as a function of $D$, so the two coldest rows cannot support a
> claim about where the maximum is at all.

**The two follow-ups the examiner will have.**

1. *"Your escape factor is the wrong geometry."* Correct, and it must be
   conceded before it is asked: ADAS214 eq. 3.14.14 is the isotropic /
   sphere-centre case (ADAS g1), not a slab; a true slab is ≈2.1× smaller at
   large $\tau_c$, and `escape_factor.py` documents it as a slab. It affects only
   the $T_e<2$ eV numbers, because $\Theta_P\approx1$ above. **Decide before the
   viva whether to relabel it isotropic or redo it as g2**; arriving without a
   decision is worse than either choice.
2. *"Trapping breaks your derivation, not just your numbers."* This is the
   sharper one and it is ADDENDUM B §B.4: trapping makes $A = A(n_g)$, which
   breaks the linearity of the excited populations in $n_g$, which makes the
   Hill coefficient $\ne 1$, which makes the width $\ne 1$, which turns the
   bound into $\tanh(m\Delta/4)$ with $m$ unknown. The exactness of the
   two-channel split *requires* $L_{EE}$, $L_{Eg}$ and $\mathbf S_E$ to contain
   no $n_g$. **This belongs next to the derivation in Chapter 5, and it is
   currently only in Chapter 6.**

---

## Q7. Is the effect just a nonlinear equilibrium response?

**Grade: MINOR. Cleanly answerable, and the answer is a good one.**

**The attack.** You have applied a temperature step and observed that a ratio
changes. Any nonlinear function changes when you change its argument. What makes
this a *transient* result rather than a restatement that $R$ depends on $T_e$?

**The answer.**

> It is the opposite of a nonlinear response — the mechanism is exactly linear
> and that is what makes it a theorem. The excited populations are **affine** in
> the ground-state population, $n_p = a_p n_g + c_p n_i$, verified by
> superposition to $3.075\times10^{-14}$ over all 784 grid-point/direction
> pairs. Because the response is affine, the ground-fed fraction is a **unit-width**
> logistic in $\ln u$ — a Hill function of coefficient exactly 1 — and the whole
> 42-state network enters only through the location $x_p = \ln(c_p/a_p)$. Width
> 1 is equivalent to exact linearity in the reservoir.
>
> And the comparison is not between two temperatures. It is between two states
> **at the same final temperature**: the partial-equilibrium state, which is the
> excited manifold equilibrated with the *new* operator and the *old* reservoir,
> and the collisional–radiative equilibrium state, which is the new operator
> with its own reservoir. Both are evaluated at the identical $(T_e, n_e)$. What
> differs is one number, $u = n_g/n_i$, and the entire error is
> $\varepsilon = |\exp(\bar S\,\Delta\ln u) - 1|$ with $\bar S$ the mean of
> $f_3-f_4$ along the path. A nonlinear equilibrium response would not
> factorise like that, and would not have a bound.

**Do not offer the $\tanh$ gate as evidence here.** It is a theorem given
$a, c \ge 0$; fed a deliberate $3\leftrightarrow4$ shell swap it still passes.
The severe checks are the superposition residual and the reduced-vs-full $R$
test, which *do* catch that swap. `make_ch3_figures.py` prints "tanh bound
honoured at all 248 points", which reads as validation and is not.

---

## Q8. Why call it non-Markovian?

**Grade: MAJOR — and the version in the brief is now half-stale, which is itself
a trap.**

**The state of play.** The word "non-Markovian" appears **nowhere** in any
chapter — `grep -rn -i "non-markov\|markov" thesis_tex/*.tex` returns nothing.
The title page already reads *The Cost of Assuming Ionisation Balance*. The only
surviving traces are the repository name `non_markovian_cr`, a Mori–Zwanzig
module in `src/rates/`, and one line in `thesis_main.tex:233` listing "the
Mori–Zwanzig memory kernel" under further work.

**So the attack has two halves and the candidate must be ready for both.**

*Half one, the registered title.* It named quasi-steady-state validity — the
approximation this work shows to be exact to $6.73\times10^{-9}$ at the worst
point on the grid. The examiner's line: *"Your title names an approximation you
have proved correct. What is the thesis about?"*

*Half two, the abandoned framing.* If the examiner has seen the project under
its old name, or opens the repository: *"You set out to find memory effects.
Did you find any, and if not, why is that not a null result?"*

**The answer.**

> The registered title named quasi-steady state because that is the
> approximation the field worries about, and testing it was the point of the
> work. The result is a reversal: the closure is exact to one part in $10^8$
> along the trajectory at the worst grid point, four orders of magnitude better
> than the error actually observed. Naming a breakdown that does not occur would
> be an own goal, so the title now names what does fail — the equilibrium
> ionisation-balance assumption hidden inside the lookup table.
>
> On memory: the system *is* non-Markovian in the formal sense, and that is
> precisely why the closure works. Eliminating the fast manifold leaves a memory
> kernel, and the kernel's correlation time is $\tau_{\rm relax} = 2.28$ ns
> against a reservoir time of 22.7 µs — a separation of $10^4$, with a spectral
> gap of $3.7\times10^7$ between $\lambda_0$ and $\lambda_1$ and no intermediate
> mode anywhere in the spectrum. A memory kernel that narrow is a delta function
> on the timescale of the dynamics, so the Markovian reduction is not merely
> adequate, it is exact to $10^{-8}$. The interesting physics is not memory in
> the excited manifold. It is that the *reservoir* the manifold is slaved to is
> the stale one, and no amount of memory-kernel machinery finds that, because it
> is not a dynamical effect at all — it is a mislabelling of a state.

**What must change:** **the claim's name, which has already changed.** Two
things remain. **(i)** The title change is a supervisor conversation and it must
happen before the examiners are appointed, not at the viva —
`claim_hierarchy` PART 6 item 7 flags it. **(ii)** The candidate should say the
memory-kernel sentence *in Chapter 3*, once, where the two-timescale structure
is established. It costs one paragraph, it disarms the question completely, and
it converts an abandoned line of work into a stated and quantified reason for
abandoning it.

---

## Q9. What exactly is QSS breakdown if the excited levels remain QSS?

**Grade: MAJOR as a trap, and it is the thesis's single best moment if handled
right.**

**The attack.** The examiner has read a project whose central metric was once
called a QSS error, and now reads that QSS is exact. *"Either your closure fails
or it does not. Which is it, and what did you spend two years measuring?"*

**The answer — and the candidate must lead with the distinction, not arrive at it.**

$$\text{QSS} \equiv \dot{\mathbf n}_E = 0 \qquad \textbf{not} \qquad \dot n_g = 0$$

> There are two different assumptions and the field's habit of calling both
> "quasi-steady state" is what made this hard to see. The first is that the
> excited manifold is algebraically slaved to the reservoir. That is the
> quasi-steady-state closure and it is what the eigenvalue separation licenses.
> Measured directly along a full 43-state trajectory, its residual is
> $8.66\times10^{-6}$ at the benchmark and $6.73\times10^{-9}$ at the worst grid
> point. It does not fail anywhere I tested.
>
> The second is that the reservoir itself has stopped moving — that the plasma
> has *finished changing*. That is a separate and much stronger assumption, it
> is the one a two-parameter lookup table imposes when it indexes on
> $(T_e, n_e)$ alone, and it is the one that fails. The error a table makes was
> $10^{-2}$ to $10^{-1}$ at the same points where the closure residual was
> $10^{-8}$. **Four orders of magnitude between them. That gap is the thesis.**
>
> The metric I once called a QSS error is the distance from the
> partial-equilibrium state to the collisional-radiative-equilibrium state. It
> was misnamed, I found it by integrating the full system and measuring both
> quantities along the same trajectory, and finding it is part of the result
> rather than an embarrassment: it identifies which of two conflated
> approximations actually costs anything.

**What must change:** nothing in the physics. But `chapter4.tex`
§`sec:qss_ratio` still contains the sentence that originated the whole retracted
framing — *"these ratios depend on $(T_e,n_e)$ but not on the absolute
normalisation of $\mathbf n$"* — and everything downstream inherits from it.
`claim_hierarchy` PART 6 item 1 makes patching it the first job. **It is
currently moot only because chapter 4 is not compiled** (§4 below), which is not
a solution.

---

## Q10. Is $M$ mathematically sufficient or merely descriptive?

**Grade: MAJOR. [known], and the answer is the thesis's best negative result —
but it has a citation problem.**

**The attack.** $M = \tau_{\rm QSS}/\tau_{\rm relax}$ runs from 86.8 to
$1.73\times10^9$ across the grid. If it is the criterion, it should locate the
failure. Does it?

**The answer.**

> $M$ is necessary and it is not sufficient, and I can say how badly. The
> maximum of $M$ and the maximum of the diagnostic error are **52× apart in
> density** — $M_{\max}$ at $n_e = 10^{12}$, $\varepsilon_{\max}$ at
> $5.18\times10^{13}$ — and at the point of largest $M$ the plateau error is
> 12 %, the 75th percentile rather than an extremum. The raw correlation
> $\mathrm{corr}(\log M, \log\varepsilon) = +0.757$ is spurious: it drops to
> $+0.27$ to $+0.33$ controlling linearly for $(T_e, n_e)$ and **flips sign** to
> between $-0.16$ and $-0.44$ under quadratic control. A bare $e^{13.6/T_e}$,
> containing no dynamics whatsoever, correlates at $+0.708$, and $\log M$ is
> 93 % explained by $(\log T_e, \log n_e)$ alone.
>
> What $M$ certifies is that the excited states have equilibrated *with the
> reservoir*. It says nothing about whether the reservoir is in the right place.
> Those are different questions and only the second one costs a diagnostic
> anything.

**Quote the sign, not the magnitude.** Over eight scope-and-basis combinations
the quadratic partial correlation ran $-0.22$ to $-0.70$. The robust statement
is "the partial correlation is negative"; the number is not robust.

**The citation problem, and it must be pre-empted, not conceded under pressure.**
Greenland, *J. Nucl. Mater.* **290–293**, 615 (2001), already concluded that CR
validity criteria "are not related to the equilibrium time-scales" and that "the
eigenvalues have secondary importance" — in general form, for arbitrary CR
systems, twenty-five years ago. And Sawada & Fujimoto (1994) carries *"Validity
range of the quasi-steady-state solution of coupled rate equations"* **in its
title**. The candidate's position:

> Greenland stated the negative result in general form and I do not improve on
> it. What this work adds is the quantification for a specific diagnostic and
> the map of where it bites — turning a general criterion into a number, which
> is a smaller contribution than stating the criterion and is the one available
> here.

Chapter 7 says exactly this, in those words. **It must be said in Chapter 1
too**, where the examiner forms their view of what is being claimed. And the
`\todo` in Chapter 1 demanding "the precise statement of what Sawada & Fujimoto
established" is, per ADDENDUM B §B.5, *"the single largest unresolved
publication risk in Chapter 1"* — the paper must be read before the viva.

---

## Q11. How sensitive is the result to atomic data?

**Grade: MAJOR. [known] and under-defended. This is the question with the
weakest available answer.**

**The attack.** The thesis is a claim about the accuracy of a diagnostic, built
on a rate matrix. What is the uncertainty on the rate matrix, and how does it
propagate? The answer, from the document itself:

- **No propagation exists.** Chapter 2, verbatim: *"A factor of three in a weak
  cascade channel is not a factor of three in the observable. **That is an
  argument, not a measurement; it would be strengthened by scaling the $n=5$
  block and recomputing the observable, and that test was not performed.**"*
- **The one external cross-section benchmark is poor.** Against Anderson RMPS,
  340 comparisons: **42.4 % within 20 %, mean absolute error 29.0 %**. For
  $n_{\rm up} = 5$ alone: **14.5 % within 20 %, mean 40.4 %**. Chapter 2: *"Taken
  at face value the first row is a failure."*
- **Lotz is used outside anything checked**, for $n = 10$–15, with the model's
  own metadata recording *"overestimates CCC by factor ~4–8"*, and because the
  three-body coefficients are built *from* it by detailed balance, that
  overestimate propagates into exactly the shells whose feed is 99.96 %
  three-body.
- **The one atomic-data discrepancy that touches the mechanism directly is
  open.** Against Fujimoto Table 4.1(b), $r_1$ is low by **8.3× at $p=3$** and
  **4.4× at $p=4$** — the two shells whose difference *is* the mechanism, at a
  density one grid interval from the ridge. Chapter 6 §`sec:r1_deficit`
  publishes this as a model failure and states *"No alternative suspect has been
  identified."*
- **And Chapter 7 nonetheless claims the bound holds "for every shell pair,
  every $(T_e,n_e)$ and every atomic dataset."**

**The answer — and it must separate two things the examiner will conflate.**

> There are two claims and they have different exposure to the atomic data.
>
> The **bound** is structural and genuinely data-independent.
> $\max|f_3-f_4| = \tanh(|\Delta|/4) < 1$ follows from each $f_m$ being a
> unit-width logistic, which follows from the excited populations being affine
> in $n_g$, which follows from the linearity of the rate equation. No rate
> coefficient enters that chain. Changing the data moves $\Delta$; it cannot
> move the form or the ceiling. That is why "for every atomic dataset" is
> defensible for the inequality.
>
> The **maps** — $\bar S$, $G$, the ridge location — depend on the data, and
> what I have is bounds on a few axes rather than a propagated uncertainty.
> $\ell$-mixing is saturated: scaling every $\ell$-mixing rate over a range of
> $\times0.1$ to $\times10$ moves $f_3-f_4$ by less than 0.3 % at the benchmark
> and at the ridge, and the $F(U_m)$ error, which was a factor 3–7 on those
> rates, moved $\tau_{\rm relax}$ by under 0.85 % anywhere on the grid.
> Truncation at $n_{\max} = 15$ extrapolates to +0.9 % at the benchmark, −1.4 %
> at the cold corner. $\Delta$ varies 0.07 % across five different
> $\ell$-weightings. What I have **not** done is scale the excitation block
> itself and recompute, and that is the test the RMPS comparison demands.

**Where the answer runs out.** Two places, and both must be conceded before
being extracted:

1. **The threshold's justification is now inconsistent.** The 10 % breakdown
   threshold is justified as matching combined ADAS-PEC (~5 %) and atomic-data
   (~5 %) uncertainty. Chapter 2 measures the atomic data at **29 % mean
   absolute error**. Those two numbers cannot both stand in one document.
2. **The $r_1$ deficit is published as a model failure that the project's own
   audit believes is a misread of the target.** ADDENDUM D.3 shows Fujimoto's
   quoted coronal asymptote requires $C(1s\to n{=}2) = 1.44\times10^{-7}$
   cm³/s — **fifteen times the accepted value** — and that a single-row density
   offset reconciles the whole table to ±10 % against a factor-50 spread as
   published. The script's own docstring says *"Re-check against the book before
   anything enters the thesis."* **It was not done, and it has entered the
   thesis.**

**What must change: the evidence, and it is the highest-value remaining run.**

1. **Scale the $n=5$ excitation block by ×2 and ×0.5 and recompute $f_3-f_4$.**
   Chapter 2 names this test and says it was not performed. It is one
   afternoon and it converts the weakest answer in the viva into a measured one.
2. **Report $\partial(\text{ridge location})/\partial(r_1\ \text{scaling})$.**
   `claim_hierarchy` F.4 sets the criterion already: if a factor-3 change in
   $r_1$ moves the ridge by more than one grid interval, the ridge *location*
   must be withdrawn and only the mechanism retained.
3. **Read Fujimoto Table 4.1(b) from the book** and check the density-row
   labels. One sentence settles it. Until then, per Chapter 7's own words,
   *"the benchmark adjudicates nothing"* — and §`sec:r1_deficit` must be
   withdrawn or heavily qualified rather than left publishing a self-indictment
   the project no longer believes.
4. **Reconcile the 10 % threshold's stated rationale with the 29 % RMPS
   number**, or re-derive the threshold from something else.

---

## Q12. Why H$\alpha$/H$\beta$?

**Grade: MAJOR. See A4b for the full attack; the summary is here.**

**The attack.** The stated reasons are brightness, visible wavelength, and the
shared lower level. All three are conveniences. The diagnostic reason runs the
other way: H$\alpha$ carries 60–70 % molecular contamination at detachment
onset, which is why detachment spectroscopy uses the high-$n$ Balmer lines. And
`findings_10` §7.3 already establishes that the result belongs to the $(3,4)$
pair and not to "the Balmer diagnostic" — an H$\alpha$/H$\gamma$ diagnostic has
its worst density a factor 2.68 lower.

**The answer.** As A4b. In short: the shared lower level makes the ratio
independent of $n=2$ and of absolute calibration; the pair is the textbook case;
the framework is pair-agnostic; and the honest sentence is *"the H$\alpha$/H$\beta$
ratio is least reliable at…"*, never *"the Balmer diagnostic is"*.

**What must change:** run one high-$n$ pair, or say plainly in Chapter 1 that
the pair was chosen as the canonical case rather than as the one used in
detachment analysis.

---

## Q13. Are shell populations enough when the model is $\ell$-resolved?

**Grade: MINOR. Well answered, with one number that must not be over-quoted.**

**The attack.** The observable in the map is $n_3/n_4$, a shell ratio. The real
observable is an $A$-weighted line ratio, and $4F$ cannot decay to $n=2$ at all
($\Delta\ell = 2$, E1-forbidden), so the $n=4$ shell sum contains a state
H$\beta$ cannot see. Why is the shortcut safe?

**The answer.**

> It is measured, not assumed. The $\ell$-populations move as a rigid body:
> $f(4S) = 0.0608$ against $f(4F) = 0.0606$, and the $4F$ fraction of the $n=4$
> shell runs 0.4361 to 0.4375 against the statistical $14/32 = 0.4375$, so
> proton-impact $\ell$-mixing drives $n=4$ statistical to better than 0.3 %
> everywhere on the grid. The populations come from the solve — grep for
> `statistical`, `stat_weight` or `(2l+1)` in the source returns zero hits, so
> nothing is imposed. The consequence of $4F$ being invisible to H$\beta$ is
> measured directly: $\varepsilon(\text{Balmer})/\varepsilon(\text{shell})$ runs
> 0.978 to 0.9999, worst case 2.2 % at the lowest density. Photon-weighting
> against energy-weighting differs by $6.7\times10^{-16}$.

**The trap the candidate must not walk into.** The often-quoted "0.06 %"
(0.386683 vs 0.386903) is a **single-point** figure. Chapter 4 warns against
quoting it grid-wide; `thesis_main.tex:190` still headlines it. **The grid-wide
number is 2.2 %.** Quote 2.2 %, mention 0.06 % as the benchmark value if at all.

**The unclosed corner.** The bundled block $n = 9$–15 carries **no $\ell$-mixing
of its own** — it is *assumed* statistical rather than driven to it. Chapter 2
concedes: *"that assumption has not been tested directly."*
`verify_bundling_psm20.py` exists and **has never been executed**. Two defects
sit in it (it silently synthesises grids if files are missing, violating the
project's own fail-loudly rule; and it may read zeros for the bundled indices
and return a false "bundling INVALID" from missing data), so "it is one command"
is wrong. **Read those two blocks and run it, or declare it a scope limitation
in Chapter 6 — not both and not silence.**

---

## Q14. Does the partial-equilibrium state correspond to any physically realisable plasma state?

**Grade: MINOR as posed; see A6 for the sharp version, which is where the value is.**

**The attack.** Is the PE state a real state of a hydrogen plasma, or an
artifact of holding one number fixed while changing others?

**The answer.**

> It is a real state and it is directly computable in one linear solve:
> $$\mathbf n_E^{\rm PE} = -L_{EE}^{-1}\left(L_{Eg}n_g^{\rm old} + \mathbf S_E\right)$$
> with $L_{EE}$, $L_{Eg}$ and $\mathbf S_E$ evaluated at the **new**
> $(T_e, n_e)$. It is not constructed — it is what the system occupies. The
> analytic expression matches an independent stiff LSODA integration of the full
> 43-state system to **six digits** at the cold corner, 0.240751 against
> 0.240747; the 1.5 % gap at the benchmark is window sampling, and
> back-extrapolation recovers 0.329045 for a ratio of 1.000047, with the fitted
> decay time coming out at $1.079\,\tau_{\rm QSS}$ *without the fit being told
> $\tau_{\rm QSS}$* and $R^2 = 0.9999999$. Every component is non-negative:
> $-L_{EE}$ is a Z-matrix with strictly positive column sums, hence a
> non-singular M-matrix, so $(-L_{EE})^{-1}\ge0$ elementwise, and the measured
> minimum entry over all 400 points is $+6.0\times10^{-13}$ with zero negatives.

**The sharp version of the question, which the examiner may reach.** *"Not
whether it is realisable — whether it is **distinguishable**."* That is A6, and
the answer is: at 25 of the 45 breakdown pairs above 2 eV the emitted ratio lies
on the CRE table, so the state is indistinguishable from a legitimate steady
plasma at a different temperature and the inversion returns a wrong answer with
no signature. At the other 20 it lies off the table and no solution exists.
**That is a better answer than "yes it is realisable", and it is not currently
in the thesis.**

---

## Q15. Is the inversion unique? / Q16. Could two different reservoir states produce the same line ratio?

**Grade: MAJOR. See A5 for the full treatment and the numbers.**

Short form, since these two are the ones the brief records as never examined:

- **Q16 — two reservoir states, same ratio: provably no.** $R(u)$ is a Möbius
  function, strictly monotone because $f_3-f_4 > 0$ at all 784 pairs with a
  minimum of 0.0402. The reservoir is uniquely recoverable from the ratio at
  fixed $(T_e,n_e)$. **The criticism does not cut both ways** — the forward
  problem is well-posed. State it and it disarms the question in one line.
- **Q15 — is the inversion unique: no, and it fails on the model's own grid.**
  $R(T_e)$ at fixed $n_e$ has an interior turning point on five of eight density
  columns, between 5.4 and 9.1 eV. At $7.2\times10^{12}$ the same ratio is
  produced at **3.31 eV and 10.0 eV**. The degeneracy band is 0.06 % to 2.2 %
  wide in $R$ — narrower than the thesis's own 10 % threshold. Chapter 7 has the
  amplification at 6.9 eV and does not recognise it as a fold, and admits its
  locus "has not been mapped".

**What must change: the claim, and it gets stronger.** Map the locus of
$\partial\ln R/\partial\ln T_e = 0$ — one solve loop — and state the fold
explicitly. It costs an afternoon and it converts an unexamined weakness into a
second structural result about the same table.

---

## Q17. What experimental timescale would actually resolve the predicted window?

**Grade: MINOR technically, MAJOR rhetorically. See A18.**

**The numbers exist:** $\tau_{\rm QSS}$ is 22.7 µs at the benchmark and 1.49 ms
at $[15,3]$; the half-life of the error is 12.95 µs and 1.10 ms respectively,
against $\tau_{\rm relax} = 2.28$ ns; and a 100 µs exposure does not integrate
the error away.

**What is missing** is any named instrument, exposure, machine or measurement —
and the falsifiable prediction that would come free. See A18 for the answer to
give and the one paragraph that repairs it.

---

# PART 3 — Two known attacks the examiner will make that are not on the list

## The ridge location

`claim_hierarchy` F.4 is candid and the candidate should be too, unprompted:
one decade in the assumed $n(1s)$ moves the ridge roughly one decade in $n_e$;
the grid resolves 0.43 decades per column and a parabolic fit shifts the vertex
15 %; the correct statement is a **range, $7\times10^{12}$ to $5\times10^{13}$
cm⁻³, resolved to about a factor 3**. The *mechanism* — three independent
routes, Griem's $n^{-17/2}$ shell-pair prediction of 6.7 against a measured 7.2,
the absolute density within 6 % of Griem's LTE criterion, and
`boundary_descent.csv` putting $\bar n = 3.1$ at that density at every
temperature — is the strongest evidenced result in the thesis and survives the
location being a range.

**Also:** `verify_ridge_mechanism.py` applies no $M > 900$ mask while
`verify_plateau_gridmap.py` does, and **54 of 400 points have $M \le 900$**, all
at $j\ge4$. The ridge script's $j = 6, 7$ statistics include points where the
plateau state does not physically exist. Either re-run masked or state it.

## "Its height falls with temperature"

85 % a step-convention artifact. At the ridge column $\varepsilon$ falls 8.4×
while $|\ln x|$ falls 5.5× and the intrinsic sensitivity $|f_3-f_4|$ is **flat
to ±13 % and non-monotonic**, peaking at 1.6 eV. The physically interesting
statement is the opposite of the one written: **at the ridge density the
diagnostic's sensitivity to ground-state lag is essentially
temperature-independent across the whole decade.**

---

# PART 4 — The attack that has nothing to do with physics

An examiner opens the PDF before reading it. What they see:

| | |
|---|---|
| **The abstract** | compiles as `[TODO: write last]` in red. The intended abstract exists only as a comment, and two of its seven beats — "at least 39 % at 1 eV" and "largest at the ionizing–recombining crossover, i.e. detachment" — are claims the body explicitly retracts |
| **Chapter 4** | not `\input`. One page in the ToC. The 47 KB `chapter4.tex` containing every validation number is not in the document. `\ref{sec:fujimoto}` and `\ref{sec:gate_d}` resolve to an empty skeleton — and `sec:gate_d` does not exist in `chapter4.tex` either |
| **Chapter 7** | not `\input`. One page. Every conclusion, every novelty statement, every retraction, and the non-invertibility result live in a file the reader never sees. The skeleton titles it *Conclusions* while the file titles it *What It Means*, and both carry `\label{ch:conclusions}` |
| **Appendices A, B, C** | empty `\chapter` stubs. Appendix B is the atomic-data provenance table that `claim_hierarchy` calls "the strongest rung in the project" |
| **Body text** | eight `\todo{}` blocks in Chapter 6, seven `[UNVERIFIED]` in Chapter 5, three in Chapter 7, a 31-line "COORDINATOR NOTES (delete before submission)" header in `chapter4.tex` |
| **Front matter** | no declaration, no certificate, no nomenclature list despite ~30 macros defined in the preamble |

**This is not a physics attack and it is more likely than any of them to
determine the examiner's opening mood.** It also interacts with the physics in
one specific way: **the thesis's answers to Q9, Q11 and Q17 live in chapters 4
and 7.** A candidate who answers "that is addressed in my conclusions chapter"
about a chapter the examiner cannot find is in a worse position than one who
never wrote it.

---

# PART 5 — Provenance of the new computations

All read-only, all against the canonical artifacts. Interpreter
`/opt/anaconda3/envs/cr/bin/python`.

| Result | Inputs | Method |
|---|---|---|
| $f_3-f_4$ sign census (784/784 positive, min 0.0402 at $[42,7]$) | `validation/divertor_map/divertor_map.csv` | direct column read |
| $R^{\rm CRE}(T_e,n_e)$, 50×8 | `data/processed/cr_matrix/{L_grid,S_grid}.npy` | $\mathbf n = -L^{-1}\mathbf S$; $R = \sum_{i\in\{3,4,5\}}n_i / \sum_{i\in\{6..9\}}n_i$, state indices per `divertor_map.txt` header |
| Turning points and degeneracy bands | as above | $\mathrm{sign}(\Delta\ln R/\Delta\ln T_e)$ per column; linear interpolation for the second branch |
| Apparent $T_e$ error and on/off-table census | as above + `divertor_map.csv` | $R^{\rm PE} = R^{\rm CRE}(i{+}1,j)(1\pm\varepsilon_{\rm plateau})$, inverted against column $j$ of $R^{\rm CRE}$ |
| CX-diffusive escape times | `findings_10` ADDENDUM D.5 anchor, reproduced to 72.2 µs | $\sigma_{\rm cx}=1.122\times10^{-14}$ cm² implied by D.5; $\lambda = 1/(n_e\sigma)$, $v = 9.79\times10^5\sqrt{T_e/1\,\mathrm{eV}}$ cm/s, $D = v\lambda/3$, $\tau = L^2/\pi^2 D$ |
| Transport recount | `divertor_map.csv` | $1/\tau_{\rm eff} = 1/\tau_{\rm QSS} + 1/\tau_{\rm esc}$ into the map's own lower-bound formula, $\tau_d = 10^{-4}$ s |

**Caveats on my own numbers, stated so they are not used beyond their scope.**
The transport estimate is a scaling of one anchor in one addendum; it assumes
$T_n = T_e$, a homogeneous slab, and at $[15,3]$ the mean free path is 4.6 cm
against a 20 cm slab so the diffusive limit is marginal — though the
free-streaming alternative there is 7.2 µs, which is *shorter*, so no assumption
inside this construction saves the point. The apparent-$T_e$ census uses the
model's own CRE table as the inversion target, which is exactly what A2 says has
never been validated against a real one; it is therefore an internally
consistent statement about this model and not a claim about ADAS. The two-valued
bands are computed on the 50-point grid with linear interpolation between nodes.

---

# PART 6 — What must change, in order of the cost of not doing it

1. **Compile chapters 4 and 7, write the abstract, fill or delete the
   appendices, strip every `\todo` and `[UNVERIFIED]`.** Nothing else on this
   list matters if the examiner opens a document with a red `[TODO]` where the
   abstract should be.
2. **Carry the transport conditional above 2 eV (A1).** Replace the census, or
   attach the tolerable-renewal-time column to it. This is the only fatal
   physics item with a cheap repair.
3. **Delete `chapter3.tex:1377-1387`** (the ADAS comparison that does not
   exist) and produce one $R^{\rm CRE}$-vs-ADAS-PEC figure, or reword the
   framing so $\varepsilon$ is stated as a distance between two states of one
   model (A2).
4. **Read Fujimoto Table 4.1(b) from the book** and either withdraw
   §`sec:r1_deficit` or rewrite it. Chapter 7 already concedes that until this
   is done the benchmark adjudicates nothing (Q11).
5. **Scale the $n=5$ excitation block and recompute** — the one test Chapter 2
   names and says was not performed (Q11). And reconcile the 10 % threshold's
   stated rationale with the 29 % RMPS mean error.
6. **Add the temperature-error column and the on/off-table split** (A6). Fifteen
   lines, and it is the sentence a diagnostician will remember.
7. **Map $\partial\ln R/\partial\ln T_e = 0$ and state the fold** (A5). One solve
   loop; Chapter 7 already says it has not been mapped.
8. **Settle the escape-factor geometry** — isotropic or slab (Q6). Arriving
   without a decision is worse than either decision.
9. **Run `verify_bundling_psm20.py` with its two defects understood, or declare
   the bundled-$\ell$ assumption a scope limitation** (Q13). Not both, not
   silence.
10. **Put the memory-kernel paragraph in Chapter 3** (Q8), the Greenland and
    Sawada–Fujimoto citations in Chapter 1 (Q10), the opacity chain next to the
    logistic derivation in Chapter 5 (Q6), and the falsifiable recovery
    signature in Chapter 7 (Q17).
11. **The title.** Already changed on the title page. Confirm with the
    supervisor that the registered title changes with it, before examiners are
    appointed.
12. **Final grep against the PDF**, per `claim_hierarchy` PART 6:
    `25 ns` · `M = 611` · `−46%` · `1.18 µs` · `ne^-1.00` · `38.7%` ·
    `detachment` · `ITER reference` · `QSS breakdown`.

---

## The single hardest question, if the examiner only asks one

> *"Your title says divertor. Your Chapter 6 says it is not a divertor result
> below 2 eV, it is not detachment, it cannot say what a real divertor does, and
> it cannot name the density its own ridge sits at as a region of a tokamak.
> Above 2 eV, where you say the model is self-consistent, a charge-exchanged
> neutral crosses the plasma in twenty microseconds against your
> one-and-a-half-millisecond ionisation time. So what, exactly, is the divertor
> result?"*

**The answer that survives it** is not a defence of the census. It is:

> The divertor is where the question comes from, not where the answer is
> quantified. What this thesis establishes is a property of a class of
> collisional–radiative model: the quasi-steady-state closure is exact to
> $10^{-8}$, the failure is in the ionisation-balance assumption underneath it,
> that failure has a closed form $\varepsilon = |\exp(\bar S G \Delta\ln T_e)-1|$
> with two tabulated structural maps, and it has a ceiling
> $|d\ln R/d\ln b_1| < 1$ that no atomic dataset can raise. Every one of those is
> a statement about the operator and none of them depends on transport,
> molecules, or the neutral density. Applying them to a real divertor requires a
> ground-state renewal time that a zero-dimensional model cannot supply, and I
> report what renewal time each point would tolerate rather than asserting one.

**If that is the answer, the title should say so.** The word "Divertor" in a
subtitle is currently doing work that Chapters 6 and 7 spend twenty pages taking
back.
