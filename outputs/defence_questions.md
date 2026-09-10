# Defence Questions — Adversarial Examiner Pass (Gate 9)

**Written 10 September 2026. Report only.** No chapter, script, data file or
register entry was modified in producing this document.

**Every new number in this document was computed here and validated against a
recorded project value before it was written down.** The five validation checks
are in PART 5. Two figures in an earlier draft of this pass did not survive
those checks and have been corrected; one attack was withdrawn as unfair and
replaced by the narrower version that survives. Where a number is quoted from a
chapter it carries its file and line.

**Prior work assumed, not repeated:** `outputs/claim_hierarchy.md`,
`outputs/pivot_decision.md`, `outputs/findings_10_four_agent_review.md` incl.
ADDENDA A–D. Attacks already known there are marked **[known]** and given only
in examiner-facing form. New attacks are marked **[NEW]** and carry their own
evidence.

**The document under attack** is the one that compiles today: `thesis_main.tex`
inputs chapters 1, 2, 3, 5 and 6 only. Title page: *The Cost of Assuming
Ionisation Balance — Transient Error in Balmer-Ratio Diagnostics of Divertor
Hydrogen Plasmas.*

---

# PART 0 — Ranking by damage

| # | Attack | Grade | Status |
|---|---|---|---|
| **A1** | **The closed-parcel assumption fails at 100 % of the breakdown points and holds at 37 % of the rest.** The assumption that generates the result fails at exactly the points that generate it | **FATAL** | **[NEW]** |
| **A2** | $\varepsilon$ is measured against *this model's own* CRE state. Gate D fails 400/400, no ADAS PEC file exists, and Chapter 3 promises a comparison that was never made | **FATAL** | **[NEW]** |
| **A3** | The $n_e$-fixed dilemma. Both horns cannot be held; the $T_e\ge2$ eV restriction cures only one of them | **MAJOR** | [known], sharpened by A1 |
| **A4** | Molecules destroy the functional form, not just the coefficient — and H$\alpha$/H$\beta$ is the pair detachment spectroscopy deliberately avoids | **MAJOR** | [known] + **[NEW]** on the pair |
| **A5** | The steady inversion is **non-unique**: $R(T_e)$ is two-valued on 5 of 8 density columns of the model's own grid | **MAJOR** | **[NEW]** |
| **A6** | The error is reported in $R$, never in $T_e$. It is 2.3× larger in $T_e$, and 25 of the 45 breakdown pairs are undetectable | **MAJOR** | **[NEW]** |
| **A7** | The $n=5$ excitation block: 40.4 % mean error over 200 comparisons, and the test Chapter 2 names to bound its effect was not performed | MAJOR | [known], **narrowed** |
| **A8** | The name. The registered title named the approximation shown to be exact | MAJOR | [known] |
| **A9** | Why this step, and does it survive 1 / 5 / 20 % | MAJOR | [known] |
| **A10** | Is $M$ sufficient or descriptive — and is it Greenland 2001 | MAJOR | [known] |
| **A11** | What is QSS breakdown if the excited levels stay QSS | MAJOR | [known] |
| **A12** | Radiation trapping — minor above 2 eV, and it breaks the *derivation* below | MINOR / MAJOR | [known] |
| **A13** | Is it just a nonlinear equilibrium response | MINOR | [known] |
| **A14** | Does the PE state correspond to a realisable plasma | MINOR | [known] |
| **A15** | Shell populations vs $\ell$-resolution | MINOR | [known] |
| **A16** | Could two reservoir states give the same ratio | MINOR | **[NEW]**, answerable |
| **A17** | Why atomic hydrogen only | MINOR | [known] |
| **A18** | What experimental timescale would resolve the window | MINOR | **[NEW]** gap |

**Separately, and more likely than any of these to set the examiner's mood:**
the abstract compiles as `[TODO: write last]`, chapters 4 and 7 are not in the
document, and all three appendices are empty stubs. PART 4.

---

# PART 1 — The attacks nobody has made

## A1 — The closed-parcel assumption fails at exactly the points that produce the result

**[NEW]. FATAL.**

### The attack

Chapter 6 treats neutral transport as a threat to the **cold edge**. Its
quantitative content is about $[0,4]$: $\tau_{\rm QSS} = 233$ ms against a
6–10 µs transit, and a renewal time of $\approx26$ µs would drop that point
below 10 %. `findings_10` ADDENDUM D.5 then rescues it — with charge exchange
the neutrals are trapped, the escape time is 72 µs not 6–10 µs, and *"the result
survives, by a factor 2.8"*.

**That factor was computed at one point, and it is a point the thesis has
already declared outside its own domain.** The same calculation has never been
run inside the defensible range.

I ran it, using ADDENDUM D.5's own CX-diffusive model and reproducing its 72 µs
anchor exactly (CHECK 4). The result is not that the numbers get somewhat worse.
It is a clean structural failure:

| slab | breakdown pairs (45) | the rest (403) |
|---|---|---|
| $L = 10$ cm | median $\tau_{\rm esc}/\tau_{\rm QSS} = 0.0023$, **0 % satisfy** $\tau_{\rm esc}>\tau_{\rm QSS}$ | median 0.050, **28 % satisfy** |
| $L = 20$ cm | median **0.0091**, **0 % satisfy** | median 0.200, **37 % satisfy** |
| $L = 50$ cm | median 0.057, **4 % satisfy** | median 1.25, **51 % satisfy** |

**Not one of the 45 breakdown points satisfies the assumption the model uses to
compute them**, at any slab thickness up to 20 cm, while more than a third of
the non-breakdown points do.

### Why the selection is systematic and not bad luck

A point appears in the census *because* $\tau_{\rm QSS}$ is long — that is what
makes the error persist through the ELM and survive the time-average. And a long
$\tau_{\rm QSS}$ is precisely the condition under which neutral transport
dominates local ionisation, because $\tau_{\rm esc}$ depends on the plasma
density and the geometry and not at all on $\tau_{\rm QSS}$. **The breakdown set
is, by construction, the subset of the grid where the model's transport
assumption is worst.** The census is not a list of points where the diagnostic
fails; it is a list of points where the model is least entitled to say.

The two boundaries the thesis *does* defend — quasineutrality and Lyman trapping
— both switch off above 2 eV. Transport does not. It gets worse across that
boundary, because $\tau_{\rm QSS}$ falls by two orders of magnitude while
$\tau_{\rm esc}$ falls by less than one.

### What transport actually does to the error — and why "it goes away" is the wrong reply

The obvious defence is that resupplying the ground state removes the staleness
and therefore the error. That is only one of two limits, and they run in
opposite directions:

- **If the influx drives $n_g$ toward its post-step CRE value**, the error
  decays on $\tau_{\rm esc}$ instead of $\tau_{\rm QSS}$ and the ELM-averaged
  bound collapses. Recomputing the census this way gives **0 of 448 above 10 %
  at $L\le20$ cm** and 7 at $L = 50$ cm.
- **If the influx pins $n_g$ at a recycling-determined value unrelated to local
  CRE** — which is the physically realistic case in a divertor, where the
  neutral density is set at the target — then $n_g$ never reaches the post-step
  CRE value at all, the error **does not decay**, and the relevant number is
  $\varepsilon_{\rm plateau}$ itself, the *upper* bound. That is **larger** than
  what the thesis reports.

So the honest statement is not that transport removes the effect. It is that
transport **brackets it from both sides and the model computes neither
bracket**. The census assumes a third thing — a closed parcel relaxing on
$\tau_{\rm QSS}$ — which CHECK 5 shows holds at none of the 45 points.

### The answer the candidate should give

There is no answer that saves the census. This one is honest and leaves a thesis:

> The box has no transport and the result is conditional on the ground-state
> population being renewed no faster than $\tau_{\rm QSS}$. I stated that
> conditional for the cold corner and did not carry it into the defensible
> range, where it binds harder — and worse, it binds selectively, because a long
> $\tau_{\rm QSS}$ is both what puts a point in the census and what makes
> transport dominate there. Applying the charge-exchange estimate from my own
> audit, not one of the 45 points satisfies the assumption at a 20 cm scale
> length.
>
> What survives is everything that is a property of the operator rather than of
> the parcel: the closure is exact to $10^{-8}$; the sensitivity $\bar S$ and
> the reservoir gain $G$ are computed from $L$ and are untouched by transport;
> and $|d\ln R/d\ln b_1| < 1$ is a theorem. Transport enters in exactly one
> place — it replaces $\tau_{\rm QSS}$ as the reservoir renewal time in
> $\Delta\ln u$. So the defensible object is not a census but a map of the
> renewal time each point can tolerate before the error falls below threshold,
> which a reader with a transport code can use directly and which I can compute
> from what I already have.

### What must change

**The evidence, or the claim — not the scope.** Widening the scope does not help
because the failure is worst inside it. In order of cost:

1. **Replace the census with the tolerable-renewal-time map.** For each point
   report $\tau_{\rm renew}^{\rm crit}$, the renewal time at which the
   ELM-averaged error crosses 10 %. It is a property of the operator,
   transport-agnostic, and answers a SOLPS user's question directly. At $[15,3]$
   it is 75 µs; at $[0,4]$, 26 µs. This is the same move the pivot already made
   from $\varepsilon$ to $G$, and it is the single highest-value repair left in
   the project.
2. If the census is kept, **put the conditional in the sentence** — "45 of 448
   pairs exceed 10 % *for a parcel whose neutral population is renewed no faster
   than $\tau_{\rm QSS}$*" — and state, in the same paragraph, that this
   condition is not met at any of the 45 under a charge-exchange estimate.
3. Merge §`sec:transport` and §`sec:open_system` into the single dilemma, as
   ADDENDUM D.1 already instructs, and extend it above 2 eV rather than closing
   it there.

**On the robustness of my own estimate.** $\tau_{\rm esc}$ inherits D.5's
$\sigma_{\rm cx} = 1.12\times10^{-14}$ cm², assumes $T_n = T_e$, and treats
escape as diffusive. At $[15,3]$ the mean free path is 4.6 cm against a 20 cm
slab, so the diffusive limit is marginal there — but the free-streaming
alternative is 7.2 µs, *shorter* than the 18.9 µs diffusive value, so the
attack strengthens under the alternative assumption. The ionisation fraction
over the whole $T_e\ge2$ eV set is $\ge0.983$ (CHECK 4), so $n_p\approx n_e$ is
safe. There is no choice inside this construction under which the 45 survive at
$L\le20$ cm.

---

## A2 — The error is measured against the model's own CRE, never against any table

**[NEW]. FATAL to the framing; survivable by rewording.**

### The attack

The thesis claim is about a **lookup table**: a table indexed only on
$(T_e,n_e)$ assumes equilibrium ionisation balance and therefore assigns the
wrong meaning to a correctly measured ratio. Chapter 1 §`sec:inversion` names
the table: ADAS.

But $\varepsilon$ is the distance from the partial-equilibrium state to **this
model's own** CRE state. For that to be a diagnostic error, this model's
equilibrium line ratio must be the one in the table. **That has never been
shown, and three facts say it may not be.**

1. **`validation/gate_summary.txt` reads `Gate D: FAIL (0% of points)`** —
   $\eta = {\rm SCD_{model}}/{\rm SCD_{ADAS}} \in [5.73, 8346]$, **0 of 400
   points within a factor 2**.
2. **No ADAS PEC file exists in the repository.** `find data -iname "*pec*" -o
   -iname "*adf15*"` returns nothing. The one claimed Balmer comparison —
   H$\alpha$/H$\beta$ = 4.1 against "ADAS PEC 2.8–4.5" in
   `data/processed/adas/report_adas.md` — rests on a file that is not there and
   has no producing script. **`chapter3.tex:1377-1387` promises this comparison
   to the reader.**
3. **Chapter 2 concedes the point:** the effective coefficients this model
   produces *cannot be compared directly with those tabulated by ADAS*, and no
   ADAS dataset enters the rate matrix at any point.

The examiner's sentence: *"You have quantified the error of your model against
your model. Show me your equilibrium ratio is the one in the table you say is
wrong."*

### The answer the candidate should give

> Gate D's failure is diagnosed, not mysterious: `SCD_model` sums ionisation
> over the full steady state while ADAS SCD is the *ionizing* coefficient, and
> the recombination half was never implemented — `acd_adas` is loaded and never
> used. It is reported as a failure and not made to pass. Chapter 2 is right
> that the effective coefficients of an open model are not ADAS-comparable,
> because ADAS closes the system and this one does not.
>
> But $\varepsilon$ does not depend on the effective coefficients. It is a ratio
> of ratios, $|R^{\rm PE}/R^{\rm CRE}-1|$, and the normalisation of the
> population vector cancels exactly — verified: $\Delta$ and the $\tanh$ bound
> are invariant under rescaling the $n_g/n_i$ normalisation by $10^{10}$ and
> $10^{-7}$. What would have to match ADAS is the *shape* of
> $R^{\rm CRE}(T_e,n_e)$, not its scale, and I have not checked that against a
> PEC file.

**And then he must say what he will do**, because the examiner will ask. One
figure suffices: this model's $R^{\rm CRE}(T_e)$ at $n_e = 10^{14}$ overlaid on
the same curve from an ADAS adf15 H PEC set. If the shapes agree within the
10 % threshold the thesis uses, the framing survives intact and the thesis gains
its only real comparison to a diagnostic table. If they do not, $\varepsilon$ is
a statement about this model and the abstract must say so.

### What must change

**The evidence.** One `adf15` file, one figure. Until then **delete
`chapter3.tex:1377-1387`** — it is the most dangerous of the four deletions on
the `claim_hierarchy` PART 6 list, not the least — and delete the 4.1 vs 2.8–4.5
sentence wherever it survives. Report Gate D's failure in the body with its
diagnosis and one sentence on why $\varepsilon$ is insensitive to it.

---

## A5 — The inversion the thesis criticises is already non-unique

**[NEW]. MAJOR. Handled first it is a gift; handled second it is a wound.**

The brief records that uniqueness "has never been examined at all". It has now.
There are two questions inside it, with opposite answers.

### (a) Two reservoir states, same ratio? **Provably no.**

The ratio is a Möbius function of the reservoir variable,
$$R(u)=\frac{a_3u+c_3}{a_4u+c_4},$$
strictly monotone unless $f_3=f_4$. Measured over all 784 (point, direction)
pairs: $f_3-f_4$ is **positive at every one**, minimum $0.0402$ at $[42,7]$. At
fixed $(T_e,n_e)$ the map $u\mapsto R$ is a bijection and the reservoir state is
uniquely recoverable.

**The diagnostic criticism does not cut both ways.** The forward problem this
thesis solves is well-posed, and saying so costs one line of algebra and one
grep. It belongs in Chapter 5 beside the bound.

### (b) Is the temperature inversion unique? **No — and it fails on the model's own grid.**

Solving $L\mathbf n+\mathbf S=0$ at every grid point (CHECK 1 validates the
solve against ADDENDUM B's $R = 0.777768$ to 6 parts in $10^7$), $R = n_3/n_4$
is **not monotone in $T_e$**. It has an interior minimum on five of eight
density columns:

| $n_e$ (cm⁻³) | $R$ min | at $T_e$ | $R$(10 eV) | degenerate band | same $R$ at |
|---|---|---|---|---|---|
| $1.00\times10^{12}$ | 0.84621 | 7.91 eV | 0.84759 | 0.16 % | 6.53 and 10.00 eV |
| $2.68\times10^{12}$ | 0.90957 | 5.96 eV | 0.92013 | 1.16 % | 4.02 and 10.00 eV |
| $7.20\times10^{12}$ | 0.93905 | 5.43 eV | 0.95976 | **2.21 %** | **3.31 and 10.00 eV** |
| $1.93\times10^{13}$ | 0.89079 | 5.96 eV | 0.90818 | 1.95 % | 3.69 and 10.00 eV |
| $5.18\times10^{13}$ | 0.79345 | 7.20 eV | 0.80006 | 0.83 % | 5.11 and 10.00 eV |
| $1.39\times10^{14}$ | 0.70774 | 9.10 eV | 0.70813 | 0.06 % | 8.27 and 10.00 eV |

At $n_e = 7.2\times10^{12}$ the ratio $R = 0.9598$ is produced by a plasma at
**3.31 eV and by one at 10.0 eV** — a factor 3.0 in temperature, with nothing in
the ratio to separate them.

**The thesis already owns half of this and stops one step short.**
`chapter7.tex:218-229` records that $\varepsilon_{\rm step}$ *"falls to
$4.9\times10^{-5}$ near \SI{6.9}{\electronvolt} … the $n=3/n=4$ ratio is locally
non-invertible for temperature at that point, so a bounded error in the
observable produces an unbounded error in the inferred temperature"*, and adds
*"Its locus across the rest of the grid has not been mapped."* That is the same
turning point seen through an absolute value. The map above is one solve loop.

But a stationary point of $R(T_e)$ is not merely an amplifier — it is a **fold**.
The ratio is two-valued, and the degeneracy band is 0.06 % to 2.2 % wide in $R$,
**narrower than the 10 % threshold the thesis uses** and comparable to the ~5 %
uncertainty that threshold was matched to.

### The answer

> The reservoir inversion is unique and provably so: $f_3-f_4>0$ at all 784
> pairs, minimum 0.040, so $R(u)$ is a strictly monotone Möbius map. The
> temperature inversion is not, and that is a property of the steady table, not
> of my transient. $R(T_e)$ turns over between 5.4 and 9.1 eV on five of eight
> density columns; at $7.2\times10^{12}$ the same ratio is returned by 3.3 eV
> and by 10 eV. Chapter 7 reports the amplification and I did not recognise it
> as a fold. It sharpens the thesis: the table is ambiguous in a band narrower
> than its own quoted uncertainty *before* any transient, and my result is what
> is added on top.

### What must change

**The claim, and it gets stronger.** Map the locus of
$\partial\ln R/\partial\ln T_e=0$ — Chapter 7 already says it has not been
mapped — plot it against the ridge, and state the fold with its band width.

---

## A6 — The error is reported in the wrong variable, and most of it is invisible

**[NEW]. MAJOR. A result the thesis is currently discarding.**

Every headline number is an error in $R$. **Nobody measures $R$ to know $R$.**
The physicist of Chapter 1 measures it to write down a temperature, and that
error is $\varepsilon_R/|\partial\ln R/\partial\ln T_e|$ — always larger,
because $|\partial\ln R/\partial\ln T_e|\le0.64$ everywhere on the grid, and
unbounded at the fold of A5.

Computing the partial-equilibrium state directly from
$\mathbf n_E^{\rm PE}=-L_{EE}^{-1}(L_{Eg}n_g^{\rm old}+\mathbf S_E)$ and
inverting its ratio against the CRE table at the same density (CHECK 2 validates
this state against the map's recorded $\varepsilon_{\rm plateau}$ to
$2\times10^{-15}$):

| point | true $T_e$ after step | $\varepsilon$ in $R$ | table returns | apparent $T_e$ error |
|---|---|---|---|---|
| $[15,3]$ worst above 2 eV | 2.121 eV | 18.1 % | **1.273 eV** | **−40.0 %** |
| benchmark $[23,5]$ | 3.089 eV | 6.4 % | 2.319 eV | −24.9 % |
| worst $T_e$ error in the set | 2.121 eV | **11.4 %** | **4.609 eV** | **+117.3 %** |
| $[0,4]$ cold corner | 1.048 eV | 38.7 % | *no solution* | — |

Over the 45 breakdown pairs above 2 eV:

- **25 of 45** produce a ratio lying **on** the CRE table. The inversion
  succeeds and returns a single, plausible, wrong temperature with no residual
  and no warning. Median apparent error **−40.0 %**, range −43.2 % to +117.3 %.
- **20 of 45** produce a ratio **off** the table — no solution exists at that
  density. These are detectable.

So the worst case, stated in the variable the user works in, is a **factor 2.2
in temperature from an 11 % error in the ratio**, and more than half of it is
silent.

### Why this is an attack and not merely an omission

The examiner will read "17.5 %" beside a threshold justified by ~5 % combined
uncertainty, conclude the effect is three times the noise, and reach for the
reply *"17 % on a line ratio is within the error bar of the atomic data you just
described."* Stated in $T_e$ that reply is unavailable: nobody claims a 40 %
temperature uncertainty from atomic data. **The candidate has undersold his own
result by a factor 2.3 and left the easiest counter open.**

### The answer

> Reporting in $R$ was conservative and it costs a factor 2.3. The ratio error
> propagates through $\partial\ln R/\partial\ln T_e$, which never exceeds 0.64
> in magnitude on this grid, so the temperature error is always the larger
> number: at the worst defensible point an 18 % ratio error is a 40 %
> temperature error, 2.12 eV read as 1.27 eV, and the worst temperature error in
> the set comes from an 11 % ratio error. And 25 of the 45 land on a valid point
> of the table, so the inversion succeeds and returns a wrong answer with no
> diagnostic signature.

### What must change

**The evidence, added; then the claim, strengthened.** One column in the census
table and one paragraph. It is fifteen lines against files that already exist,
and "the failure is undetectable in 56 % of cases" is the sentence a
diagnostician will remember.

---

## A4b — H$\alpha$/H$\beta$ is the pair detachment spectroscopy avoids

**[NEW] as an attack. MAJOR.**

Chapter 1 justifies the choice: the two brightest, visible wavelengths
convenient for windows and fibres, and a shared lower level so the ratio is
independent of $n=2$. All three are conveniences; none is a diagnostic argument.

The diagnostic argument runs the other way. The literature the thesis itself
cites attributes **60–70 % of the $D_\alpha$ light** at detachment onset to
molecular channels, against 10–20 % of $D_\gamma$ — which is exactly why
practitioners in this regime work with the **high-$n$ Balmer lines**, $n=5$–9,
whose emission is recombination-dominated and molecule-poor. The thesis picked
the one line in the series those practitioners discard, in a model with no
molecules.

The two compound. ADDENDUM D.4's molecular bound is a factor 2.7–5.2 on
$f_3-f_4$ *because* $\phi_3\gg\phi_4$ — the chosen pair straddles the steepest
part of the contamination gradient. A pair further up the series would have both
a smaller correction and a smaller uncertainty on it.

### The answer

> The pair was chosen for brightness and for the shared lower level, which makes
> the ratio independent of $n=2$ and of absolute calibration. The molecular
> objection is real and I have bounded it rather than only admitting it: with
> the published emissivity fractions the sensitivity falls by 2.7 to 5.2 at the
> cold corner. What I have not done is repeat the map for a higher pair. The
> framework is pair-agnostic — $\bar S$ and $G$ are defined for any $(p,q)$ —
> and §7.3 already reports that the $(3,5)$ pair moves the worst density by
> 2.68, so the machinery exists.

**What must change:** either run one high-$n$ pair — the same script with two
indices changed, and it would answer the objection outright — or state plainly
in Chapter 1 that $(3,4)$ is treated as the canonical textbook case and not as
the pair used in detachment analysis.

---

## A18 — No instrument is ever named

**[NEW] as a gap. MINOR technically, and it is the last question of a viva.**

Chapter 7 has the timescales: a half-life of 12.95 µs at the benchmark and
1.10 ms at $[15,3]$ against $\tau_{\rm relax}=2.28$ ns, and a 100 µs exposure
does not integrate the error away. Those are right.

Absent everywhere: any named spectrometer, camera, machine, campaign or
achievable exposure. Chapter 7's further work proposes four calculations and
zero experiments. *"What would falsify this?"* currently answers "another
calculation".

### The answer

> The window is set by $\tau_{\rm QSS}$ — 22.7 µs at the benchmark, 1.5 ms at
> the worst defensible point — and by the transient driving it, 100 µs for an
> ELM. So the requirement is an exposure short compared with $\tau_{\rm QSS}$:
> of order 1–10 µs at the benchmark and comfortably 100 µs at $[15,3]$, which
> divertor Balmer systems already achieve. The prediction is that after an ELM
> the inferred temperature undershoots by tens of percent and recovers on
> $\tau_{\rm QSS}$ — and since $\lambda_0 = 2.286\,K_{\rm ion}(1s)\,n_e$, the
> recovery time should scale inversely with density. **That scaling is the
> falsifiable part**, because it is a prediction about the shape of the recovery
> rather than its size, and it is insensitive to everything transport does to
> the magnitude.

**What must change:** one paragraph in Chapter 7 with the recovery signature and
its $n_e^{-1}$ scaling, and two sentences citing achievable exposures. It is the
difference between a thesis that ends in a calculation and one that ends in a
measurement someone could make. Note that this prediction survives A1 — it is a
statement about the operator's eigenvalue, not about the parcel.

---

## Smaller new ones, recorded so the examiner does not find them first

| | Attack | Grade |
|---|---|---|
| **A19** | **Two-photon $2s$ decay is not in the matrix** — Chapter 2 says so. Yet ADDENDUM D.2 rests its $\ell$-mixing-independent bound on it ("8.229 s⁻¹ alone is 553× faster than $\lambda_0$"). That bound describes a physical atom, not the model. If it is quoted to defend the partition, say which. | MINOR |
| **A20** | **The model is $^1$H with infinite nuclear mass; the transport numbers are deuterium.** Immaterial numerically ($\sqrt2$ on transit), sloppy in a viva, fixed by one sentence. | MINOR |
| **A21** | **Line-of-sight integration is never raised.** A spectrometer views a chord through steep gradients; the thesis compares a 0-D point value to a chord-integrated measurement. Since $\varepsilon$ varies 8× across one decade in $n_e$, chord-averaging is not a small correction, and it is not in the limitations chapter at all. | MINOR–MAJOR |
| **A22** | **Impurities appear nowhere** in Chapters 6 or 7 — no C, Be, W, N, Ne, no impurity radiative cooling or recombination. In a seeded detached divertor this is the one limitation not even admitted. | MINOR |
| **A23** | **`chapter7.tex:149` carries an `[UNVERIFIED]`** stating that no script in the repository performs the two aggregations behind the reservoir-gain claim — the 6.3 % spread and the 8.44 factor. Those two numbers are the headline of the pivot. | MAJOR |

---

# PART 2 — The seventeen

## Q1. Why is the chosen temperature step physically meaningful?

**MAJOR. [known] — `pivot_decision` §2 already retires it.**

**The attack.** It is not. The +4.81 % step exists because someone chose 50
log-spaced points between 1 and 10 eV, and $\varepsilon_{\rm plateau}$ is linear
in it. "38.7 %" means "38.7 % per 4.81 % step". The follow-up is immediate:
*what if the ELM is 10 %?* — 87.4 %. *Twenty?* — 190.4 %.

**The answer.**

> The step is a grid artifact and the percentage from it is not a property of
> the plasma, which is why the chapter does not headline it. The
> step-independent object is the reservoir gain $G = d\ln u/d\ln T_e$, and
> $\varepsilon = |\exp(\bar S G\,\Delta\ln T_e)-1|$ for whatever step the reader
> cares about. Across the 736 point-and-direction triples for which one, two and
> four grid intervals were all computed, $|G|$ moves by at most **6.3 %** while
> $\varepsilon_{\rm plateau}$ at the same points changes by up to a factor
> **8.44**. Every $\varepsilon$ in the thesis carries its step size.

**What must change:** nothing further, *provided* the chapters execute the
pivot — and provided A23 is closed, since no script currently produces those two
numbers.

---

## Q2. Does the result survive 1, 5 and 20 percent perturbations?

**MAJOR. [known], with one live gap.**

**The attack.** Press the arithmetic behind "yes, because $G$ is stable". The
6.3 % figure was measured over $k=1,2,4$ — steps of 4.81 %, 9.85 % and 20.68 %.
So 5 % and 20 % are inside the measured range and **1 % is not**. And the
linearisation $\varepsilon\approx|f_3-f_4||\ln x|$ was validated only over
$|\ln x|\in[0.124,0.682]$, while the four-interval numbers sit **3.8× beyond**
that ceiling.

**The answer.**

> At 5 % and 20 % it is measured: $|G|$ moves at most 6.3 % across that fourfold
> range while $\varepsilon$ moves by up to 8.44. At 1 % the answer is the
> small-step limit itself, $d\varepsilon/d\ln T_e = |\bar S G|$, which over the
> defended range runs from 0.362 to 3.535 — so a 1 % excursion costs at most
> about 3.5 % in the ratio and about 1.3 % at the benchmark. At 20 % the exact
> form must be used, not the linearisation: those numbers are exact linear
> solves and any sentence explaining them through $|f_3-f_4||\ln x|$ is outside
> where that decomposition was checked.

**What must change:** the evidence, cheaply — run a sub-grid-interval step so
the 1 % end is measured rather than extrapolated, and keep the "do not use the
linearised form" instruction physically attached to every four-interval number.

---

## Q3. Why is $n_e$ held fixed?

**FATAL as posed; MAJOR after the $T_e\ge2$ eV restriction; and then see A1.**

**The attack.** Quasineutrality ties $n_e$ to $n_g$. At 1 eV the plasma is
95–98 % neutral, so the reservoir collapse that *produces* the error is the same
collapse that *invalidates the operator it was computed with*. **68 of 392
one-step operators (17 %) require $|\Delta n_e/n_e|>10$ %; 36 (9 %) require more
than 100 %.** At the cold corner it is +20.0. The model evaluates the post-step
operator at the old $n_e$ at every one.

The dilemma is genuine:

- **Closed parcel** → quasineutrality forces $n_e$ up 10–20× at the cold end and
  every rate in the post-step operator is wrong by that factor.
- **Transport-fed** so $n_e$ stays fixed → the ground state is resupplied and is
  not stale, removing the mechanism.

Chapter 5 §`sec:quasineutrality` states this correctly and in those words.

**The answer.**

> Holding $n_e$ fixed is self-consistent only where the ionisation degree is
> high, and I measured where rather than assuming. Above 2 eV the required
> correction falls below $10^{-3}$; below it, 68 of 392 operators need more than
> 10 %. That is the reason for the restriction, and it is not a hedge — it is
> the region where the model's own assumptions are mutually consistent, and it
> coincides, from independent physics, with where Lyman trapping stops
> mattering. Below it the numbers are reported as the asymptotic behaviour of a
> model outside its domain, because suppressing them would be worse.

**Where the answer runs out.** *"You escaped the first horn above 2 eV. Show me
you escaped the second."* Above 2 eV $\Delta n_e/n_e<10^{-3}$ because the
neutrals are a small perturbation on the **electron** budget. That says nothing
about whether the **neutral** population is transport-renewed — and A1 shows it
is, at every breakdown point, by 40–1000×. **The two horns are not symmetric and
the restriction cures only one.**

---

## Q4. Why atomic hydrogen only? / Q5. What happens with molecules?

**MAJOR. [known], with two live problems.**

**The attack.** At $[0,4]$ the model's own equilibrium is **96.6 % neutral** —
conditions where the cited literature attributes 60–70 % of $D_\alpha$ to
molecular channels. `grep -riE "H2|molecul|MAR|dissociat" src/rates/` returns
nothing. Chapter 6 states: *"this work offers no bound on how much, and none can
be constructed from the data in this repository."*

Two things make it worse than an ordinary admitted limitation:

1. **Structural (ADDENDUM B §B.4).** A molecular channel is a *third* channel.
   It does not perturb the two-channel split, it **destroys its functional
   form**: $n_p = a_pn_g+c_pn_i+m_pn_{\rm H_2}$ is not a one-parameter family,
   the logistic is not a logistic, and the $\tanh$ bound is a statement about a
   one-dimensional family with no reason to survive. **Molecules do not scale the
   result; they invalidate the derivation.**
2. **The document argues with itself.** Chapter 6 says no bound can be
   constructed; ADDENDUM D.4 constructs one from the two papers cited in that
   same paragraph; Chapter 7 flags the contradiction in an `[UNVERIFIED]` block
   and quotes the bound anyway.

**The answer.**

> Atomic hydrogen only is a data limit and a scope limit, and it is why the grid
> stops at 1 eV. The bound Chapter 6 says cannot be constructed can be, from the
> two references in that paragraph: the molecular emissivity fraction is the
> population fraction and MAR bypasses the ground state, so
> $f_m\to f_m(1-\phi_m)$. With $\phi_3=0.60$, $\phi_4=0.30$ the cold-corner
> sensitivity falls by 2.7; with 0.70 and 0.45, by 5.2. That sentence is wrong
> and I will correct it.
>
> The honest limit is structural rather than numerical: the two-channel split is
> exact only for two channels, so the framework is two-channel **by
> construction**. Above about 3 eV molecular densities are small and the
> description holds; where they are not, the functional form and not just the
> coefficient is at risk.

**What must change:** replace "no bound can be constructed" with D.4's bound;
reconcile Chapters 6 and 7 so the document stops contradicting itself; and state
the two-channel-by-construction limit **where the logistic is derived**, in
Chapter 5. ADDENDUM B calls this "the most defence-dangerous item" and it is
still only in the limitations chapter. See also **A4b**.

---

## Q6. What does radiation trapping change?

**MINOR above 2 eV — genuinely well answered. MAJOR below.**

**The attack.** At 1 eV, $\tau_{\rm Ly\alpha}=114$ per cm and the escape factor
over 5 cm is $2.7\times10^{-5}$: the Einstein coefficients in `L_grid.npy` are
wrong as effective decay rates by **four to five orders of magnitude** at the
point carrying the headline.

**The answer, one of the strongest passages in the project.**

> It was computed, not argued. All 14 Lyman channels, escape factors applied
> self-consistently — $\Theta_P$ depends on $n(1s)$ which depends on $\Theta_P$
> — converged at all 400 points, swept over slab thickness rather than fixed,
> with three gates passed before any result was read, one of which failed at
> 1156 % on first run and caught a $4\pi$ CGS/SI error. The untrapped rebuild
> reproduces the canonical matrix exactly, $\max|{\rm rebuilt-canonical}|=0$.
>
> Above 2 eV nothing moves: the count stays at 45 of 448 for $D=1$, 5 and 20 cm
> and the worst case runs 0.1748, 0.1746, 0.1740 — 0.5 % across a twentyfold
> range in a parameter this model does not contain. That is what makes the
> boundary a measurement rather than a disclaimer. Below it the headline falls
> by 2.5–3.3× and which value it takes is set by an assumed slab thickness, so
> the point still breaks down but the magnitude is not quotable. Below 1.15 eV
> the column carrying the maximum wanders across a factor 7 with $D$, so the two
> coldest rows cannot support a ridge-location claim at all.

**Two follow-ups, both of which must be conceded before they are asked.**

1. **Wrong geometry.** ADAS214 eq. 3.14.14 is the isotropic / sphere-centre case
   (ADAS g1), not a slab; a true slab is ≈2.1× smaller at large $\tau_c$, and
   `escape_factor.py` documents it as a slab. Affects only $T_e<2$ eV. **Decide
   before the viva** whether to relabel or redo as g2; arriving without a
   decision is worse than either.
2. **Trapping breaks the derivation, not just the numbers.** ADDENDUM B §B.4:
   trapping makes $A=A(n_g)$ → linearity in $n_g$ broken → Hill coefficient
   $\ne1$ → width $\ne1$ → the bound becomes $\tanh(m\Delta/4)$ with $m$
   unknown. The exactness of the split *requires* $L_{EE}$, $L_{Eg}$ and
   $\mathbf S_E$ to contain no $n_g$. **This belongs next to the derivation and
   is currently only in Chapter 6.**

---

## Q7. Is the effect just a nonlinear equilibrium response?

**MINOR. Cleanly answerable.**

**The attack.** You changed $T_e$ and a ratio changed. Any nonlinear function
does that. What makes this transient rather than a restatement that $R$ depends
on $T_e$?

**The answer.**

> It is the opposite of a nonlinear response, and that is what makes it a
> theorem. The excited populations are **affine** in the ground-state
> population, $n_p = a_pn_g+c_pn_i$, verified by superposition to
> $3.075\times10^{-14}$ across all 784 pairs. Because the response is affine the
> ground-fed fraction is a **unit-width** logistic in $\ln u$ — a Hill function
> of coefficient exactly 1 — and the whole 42-state network enters only through
> the location $x_p=\ln(c_p/a_p)$.
>
> And the comparison is not between two temperatures. Both states are evaluated
> at the **same** final $(T_e,n_e)$: the partial-equilibrium state is the
> excited manifold equilibrated with the new operator and the old reservoir, the
> CRE state is the new operator with its own reservoir. One number differs, $u$,
> and the whole error is $\varepsilon=|\exp(\bar S\,\Delta\ln u)-1|$. A
> nonlinear equilibrium response would not factorise like that and would not
> have a bound.

**Do not offer the $\tanh$ gate as evidence.** It is a theorem given
$a,c\ge0$ and passes a deliberate $3\leftrightarrow4$ shell swap. The severe
checks are the superposition residual and the reduced-vs-full $R$ test, which do
catch that swap.

---

## Q8. Why call it non-Markovian?

**MAJOR — and the premise is now half-stale, which is its own trap.**

**State of play.** "Non-Markovian" appears **nowhere** in any chapter
(`grep -rn -i "markov" thesis_tex/*.tex` → nothing). The title page already
reads *The Cost of Assuming Ionisation Balance*. Surviving traces: the
repository name, a Mori–Zwanzig module in `src/rates/`, and one line in
`thesis_main.tex:233` listing "the Mori–Zwanzig memory kernel" under further
work.

**Two halves, both of which need an answer.** *The registered title* named
quasi-steady-state validity — the approximation shown exact to
$6.73\times10^{-9}$. *The abandoned framing* — an examiner who has seen the
project under its old name will ask whether memory effects were found.

**The answer.**

> The registered title named quasi-steady state because that is the
> approximation the field worries about, and testing it was the point. The
> result is a reversal: the closure is exact to one part in $10^8$ at the worst
> grid point, four orders better than the error actually observed. Naming a
> breakdown that does not occur would be an own goal, so the title now names
> what does fail.
>
> On memory: the system **is** non-Markovian in the formal sense, and that is
> exactly why the closure works. Eliminating the fast manifold leaves a memory
> kernel whose correlation time is $\tau_{\rm relax}=2.28$ ns against a
> reservoir time of 22.7 µs — a separation of $10^4$, with a spectral gap of
> $3.7\times10^7$ and no intermediate mode anywhere in the spectrum. A kernel
> that narrow is a delta function on the timescale of the dynamics, so the
> Markovian reduction is not merely adequate, it is exact to $10^{-8}$. The
> interesting physics is not memory in the excited manifold; it is that the
> reservoir the manifold is slaved to is the stale one — and no memory-kernel
> machinery finds that, because it is not a dynamical effect at all. It is a
> mislabelling of a state.

**What must change.** The title change must be settled with the supervisor
**before examiners are appointed**, not at the viva. And the memory-kernel
paragraph should appear **in Chapter 3**, once, where the two-timescale
structure is established — it costs one paragraph, disarms the question
completely, and converts an abandoned line of work into a quantified reason for
abandoning it.

---

## Q9. What exactly is QSS breakdown if the excited levels remain QSS?

**MAJOR as a trap, and the thesis's best moment if led with rather than reached.**

$$\text{QSS}\equiv\dot{\mathbf n}_E=0\qquad\textbf{not}\qquad\dot n_g=0$$

**The answer.**

> There are two assumptions and the field's habit of calling both
> "quasi-steady state" is what made this hard to see. The first is that the
> excited manifold is algebraically slaved to the reservoir — that is the
> closure, and it is what the eigenvalue separation licenses. Measured along a
> full 43-state trajectory its residual is $8.66\times10^{-6}$ at the benchmark
> and $6.73\times10^{-9}$ at the worst grid point. It does not fail anywhere I
> tested.
>
> The second is that the reservoir has stopped moving — that the plasma has
> *finished changing*. That is separate, much stronger, is what a two-parameter
> table imposes, and it is the one that fails. The error a table makes was
> $10^{-2}$ to $10^{-1}$ at the same points where the closure residual was
> $10^{-8}$. **Four orders of magnitude between them. That gap is the thesis.**
>
> The metric I once called a QSS error is the distance from the
> partial-equilibrium state to the CRE state. It was misnamed; I found it by
> integrating the full system and measuring both along the same trajectory; and
> finding it is part of the result, because it identifies which of two conflated
> approximations actually costs anything.

**What must change:** nothing in the physics. `chapter4.tex` §`sec:qss_ratio`
still contains the sentence that originated the retracted framing, and
everything downstream inherits from it. It is currently moot only because
chapter 4 is not compiled, which is not a solution.

---

## Q10. Is $M$ mathematically sufficient or merely descriptive?

**MAJOR. [known]. The best negative result in the thesis, with a citation problem.**

**The answer.**

> $M$ is necessary and not sufficient, and I can say how badly. $M_{\max}$ and
> $\varepsilon_{\max}$ are **52× apart in density**, and at the $M$ maximum the
> plateau error is 12 %, the 75th percentile. The raw correlation
> $\mathrm{corr}(\log M,\log\varepsilon)=+0.757$ is spurious: it falls to
> $+0.27$–$+0.33$ controlling linearly for $(T_e,n_e)$ and **flips sign** to
> between $-0.16$ and $-0.44$ under quadratic control. A bare $e^{13.6/T_e}$,
> containing no dynamics, correlates at $+0.708$, and $\log M$ is 93 % explained
> by $(\log T_e,\log n_e)$ alone.
>
> What $M$ certifies is that the excited states have equilibrated *with the
> reservoir*. It says nothing about whether the reservoir is in the right place.
> Only the second question costs a diagnostic anything.

**Quote the sign, not the magnitude** — over eight scope/basis combinations the
quadratic partial ran $-0.22$ to $-0.70$.

**The citation problem, which must be pre-empted rather than conceded under
pressure.** Greenland (2001) already concluded that CR validity criteria "are
not related to the equilibrium time-scales" and that "the eigenvalues have
secondary importance", in general form, 25 years ago. Sawada & Fujimoto (1994)
carries *"Validity range of the quasi-steady-state solution of coupled rate
equations"* **in its title**. The position:

> Greenland stated the negative result in general form and I do not improve on
> it. What this adds is quantification for a specific diagnostic and a map of
> where it bites — a smaller contribution than stating the criterion, and the
> one available here.

Chapter 7 says this. **It must also be said in Chapter 1**, where the examiner
forms their view of what is claimed. And the Chapter 1 `\todo` demanding the
precise statement of what Sawada & Fujimoto established is, per ADDENDUM B §B.5,
*"the single largest unresolved publication risk in Chapter 1"* — read the paper
before the viva.

---

## Q11. How sensitive is the result to atomic data?

**MAJOR, but narrower than it first appears. [known].**

**The attack must be aimed carefully, because the obvious version fails.** The
aggregate benchmark row looks damning — against Anderson RMPS, 340 comparisons,
**42.4 % within 20 %, mean absolute error 29.0 %** — and Chapter 2 says of it
*"Taken at face value the first row is a failure."* But Chapter 2 then
decomposes it correctly and the decomposition defends the model:

| subset | comparisons | within 20 % | mean \|error\| |
|---|---|---|---|
| all transitions | 340 | 42.4 % | 29.0 % |
| $n_{\rm up}\le4$ | 140 | **82.1 %** | **12.7 %** |
| excited-state, $n_{\rm up}\le4$ | 104 | 86.5 % | 10.7 % |
| $n_{\rm up}=5$ only | 200 | 14.5 % | 40.4 % |

and the two transitions the mechanism leans on hardest agree at −14.5 % / −2.2 %
($1s\to2p$) and −6.7 % / −0.7 % ($2p\to3d$). **An examiner who quotes the 29 %
without the decomposition will be corrected in one sentence.** What survives:

1. **The $n=5$ block carries 40.4 % mean error over 200 comparisons, and the
   test that would bound its effect was not run.** Chapter 2 concedes it in
   terms an examiner will read aloud: *"A factor of three in a weak cascade
   channel is not a factor of three in the observable. **That is an argument,
   not a measurement; it would be strengthened by scaling the $n=5$ block and
   recomputing the observable, and that test was not performed.**"*
2. **Lotz is used outside anything checked** for $n=10$–15, with the metadata
   recording *"overestimates CCC by factor ~4–8"* — and because the three-body
   coefficients are built *from* it by detailed balance, that propagates into
   exactly the shells whose feed is essentially all three-body.
3. **The threshold's stated rationale is inconsistent.** The 10 % threshold is
   justified as matching ~5 % ADAS-PEC and ~5 % atomic-data uncertainty. The
   measured atomic-data figure on the transitions that matter is **12.7 %**, and
   29.0 % in aggregate. Those numbers cannot both stand in one document.
4. **The $r_1$ deficit is published as a model failure the project's own audit
   no longer believes.** Chapter 6 §`sec:r1_deficit` reports $r_1$ low by 8.3×
   at $p=3$ — on the two shells the mechanism is built from — while ADDENDUM D.3
   shows Fujimoto's quoted asymptote requires $C(1s\to n{=}2)$ **fifteen times
   the accepted value**, and that a single-row density offset reconciles the
   table to ±10 % against a factor-50 spread as published. The script's docstring
   says *"Re-check against the book before anything enters the thesis."*
   **It was not done, and it has entered the thesis.**

**The answer.**

> Two claims with different exposure. The **bound** is structural and genuinely
> data-independent: $\max|f_3-f_4|=\tanh(|\Delta|/4)<1$ follows from each $f_m$
> being a unit-width logistic, which follows from the populations being affine
> in $n_g$, which follows from linearity of the rate equation. No rate
> coefficient enters that chain — changing the data moves $\Delta$, it cannot
> move the form or the ceiling.
>
> The **maps** depend on the data, and what I have is bounds on several axes
> rather than a propagated uncertainty. $\ell$-mixing is saturated: scaling
> every $\ell$-mixing rate over $\times0.1$ to $\times10$ moves $f_3-f_4$ by
> under 0.3 % at the benchmark and the ridge, and the $F(U_m)$ error — a factor
> 3–7 on those rates — moved $\tau_{\rm relax}$ by under 0.85 % anywhere.
> Truncation extrapolates to +0.9 % at the benchmark and −1.4 % at the cold
> corner. $\Delta$ varies 0.07 % across five $\ell$-weightings. The excitation
> data are good where the kinetics live: 82.1 % within 20 % and 12.7 % mean
> error for $n_{\rm up}\le4$, with the disagreement confined to $n=5$, which
> feeds cascades rather than the supply of the emitting levels. What I have not
> done is scale the $n=5$ block and recompute, and that is the one test the
> benchmark demands.

**What must change: the evidence, and it is the highest-value remaining run.**

1. **Scale the $n=5$ excitation block ×2 and ×0.5 and recompute $f_3-f_4$.**
   Chapter 2 names this test and says it was not performed. One afternoon, and
   it converts the weakest answer in the viva into a measured one.
2. **Report $\partial(\text{ridge location})/\partial(r_1\text{ scaling})$.**
   `claim_hierarchy` F.4 sets the criterion: if a factor-3 change moves the
   ridge more than one grid interval, the *location* is withdrawn and only the
   mechanism kept.
3. **Read Fujimoto Table 4.1(b) from the book** and check the density-row
   labels. Until then §`sec:r1_deficit` must be withdrawn or heavily qualified
   rather than left publishing a self-indictment the project disbelieves.
4. **Reconcile the 10 % threshold's rationale with 12.7 %**, or re-derive it.

---

## Q12. Why H$\alpha$/H$\beta$?

**MAJOR. Full attack at A4b.** Summary: the stated reasons are brightness,
visible wavelength and the shared lower level — all conveniences. H$\alpha$
carries 60–70 % molecular contamination at detachment onset, which is why
detachment spectroscopy uses high-$n$ Balmer lines. And §7.3 already establishes
the result belongs to the $(3,4)$ pair, not to "the Balmer diagnostic": an
H$\alpha$/H$\gamma$ diagnostic has its worst density a factor 2.68 lower. Write
*"the H$\alpha$/H$\beta$ ratio is least reliable at…"*, never *"the Balmer
diagnostic is"*.

---

## Q13. Are shell populations enough when the model is $\ell$-resolved?

**MINOR. Well answered, with one number that must not be over-quoted.**

**The attack.** The map's observable is a shell ratio; the real observable is
$A$-weighted, and $4F$ cannot decay to $n=2$ at all ($\Delta\ell=2$,
E1-forbidden), so the $n=4$ shell sum contains a state H$\beta$ cannot see.

**The answer.**

> Measured, not assumed. The $\ell$-populations move as a rigid body:
> $f(4S)=0.0608$ against $f(4F)=0.0606$, and the $4F$ fraction of the $n=4$
> shell runs 0.4361–0.4375 against the statistical $14/32=0.4375$, so
> proton-impact $\ell$-mixing drives $n=4$ statistical to better than 0.3 %
> everywhere. The populations come from the solve — grep for `statistical`,
> `stat_weight` or `(2l+1)` returns zero hits, so nothing is imposed. The
> consequence of $4F$ being invisible is measured directly:
> $\varepsilon(\text{Balmer})/\varepsilon(\text{shell})$ runs 0.978–0.9999,
> worst case 2.2 % at the lowest density. Photon- against energy-weighting
> differs by $6.7\times10^{-16}$.

**The trap.** The often-quoted "0.06 %" is a **single-point** figure;
`thesis_main.tex:190` still headlines it. **The grid-wide number is 2.2 %.**

**The unclosed corner.** The bundled block $n=9$–15 carries no $\ell$-mixing of
its own — it is *assumed* statistical rather than driven to it, and Chapter 2
concedes the assumption "has not been tested directly".
`verify_bundling_psm20.py` exists and **has never been executed**, and two
defects sit in it (it silently synthesises grids if files are missing, violating
the project's own fail-loudly rule; and it may read zeros for the bundled
indices and return a false "INVALID" from missing data). **Read those two blocks
and run it, or declare it a scope limitation — not both, and not silence.**

---

## Q14. Does the partial-equilibrium state correspond to any physically realisable plasma state?

**MINOR as posed. The sharp version is A6.**

**The answer.**

> It is a real state, computable in one linear solve:
> $\mathbf n_E^{\rm PE}=-L_{EE}^{-1}(L_{Eg}n_g^{\rm old}+\mathbf S_E)$ with
> $L_{EE}$, $L_{Eg}$, $\mathbf S_E$ at the **new** $(T_e,n_e)$. It is not
> constructed — it is what the system occupies. The analytic expression matches
> an independent stiff LSODA integration of the full 43-state system to **six
> digits** at the cold corner, 0.240751 against 0.240747; the 1.5 % gap at the
> benchmark is window sampling, and back-extrapolation recovers 0.329045 for a
> ratio of 1.000047, with the fitted decay time emerging as
> $1.079\,\tau_{\rm QSS}$ *without the fit being told $\tau_{\rm QSS}$* and
> $R^2=0.9999999$. Every component is non-negative: $-L_{EE}$ is a Z-matrix with
> strictly positive column sums, hence a non-singular M-matrix, so
> $(-L_{EE})^{-1}\ge0$ elementwise — measured minimum entry $+6.0\times10^{-13}$
> over all 400 points, zero negatives.

**The sharper question the examiner may reach:** *not whether it is realisable
— whether it is **distinguishable**.* That is A6: at 25 of the 45 breakdown
pairs the emitted ratio lies on the CRE table, so the state is indistinguishable
from a legitimate steady plasma at another temperature and the inversion returns
a wrong answer with no signature. At the other 20 no solution exists. **That is
a better answer than "yes, it is realisable", and it is not in the thesis.**

---

## Q15. Is the inversion unique? / Q16. Could two different reservoir states produce the same line ratio?

**MAJOR. Full treatment and numbers at A5.**

- **Q16 — two reservoir states, same ratio: provably no.** $R(u)$ is Möbius and
  strictly monotone because $f_3-f_4>0$ at all 784 pairs, minimum 0.0402. **The
  criticism does not cut both ways**; the forward problem is well-posed, and
  saying so disarms the question in one line.
- **Q15 — is the inversion unique: no, and it fails on the model's own grid.**
  $R(T_e)$ turns over between 5.4 and 9.1 eV on five of eight density columns;
  at $7.2\times10^{12}$ the same ratio is produced at 3.31 eV and 10.0 eV. The
  degeneracy band is 0.06 %–2.2 % wide in $R$, narrower than the thesis's own
  threshold. Chapter 7 has the amplification and does not name it a fold, and
  admits its locus "has not been mapped".

---

## Q17. What experimental timescale would actually resolve the predicted window?

**MINOR technically, MAJOR rhetorically. See A18.** The numbers exist —
$\tau_{\rm QSS}$ of 22.7 µs at the benchmark and 1.49 ms at $[15,3]$, error
half-lives of 12.95 µs and 1.10 ms, and a 100 µs exposure that does not
integrate the error away. What is missing is any named instrument, and the
falsifiable $n_e^{-1}$ recovery-time prediction that comes free.

---

# PART 3 — Two known attacks not on the list

**The ridge location.** Be candid unprompted: one decade in the assumed $n(1s)$
moves the ridge roughly one decade in $n_e$; the grid resolves 0.43 decades per
column; a parabolic fit shifts the vertex 15 %. The correct statement is a
**range, $7\times10^{12}$–$5\times10^{13}$ cm⁻³, resolved to about a factor 3**.
The *mechanism* survives that entirely — Griem's $n^{-17/2}$ shell-pair
prediction of 6.7 against a measured 7.2, the absolute density within 6 % of
Griem's LTE criterion, and `boundary_descent.csv` putting $\bar n = 3.1$ at that
density at every temperature. Separately, `verify_ridge_mechanism.py` applies no
$M>900$ mask while `verify_plateau_gridmap.py` does, and **54 of 400 points have
$M\le900$**, all at $j\ge4$: the ridge script's $j=6,7$ statistics include points
where the plateau state does not physically exist.

**"Its height falls with temperature"** is 85 % a step-convention artifact. At
the ridge column $\varepsilon$ falls 8.4× while $|\ln x|$ falls 5.5× and
$|f_3-f_4|$ is **flat to ±13 % and non-monotonic**, peaking at 1.6 eV. The
physically interesting statement is the opposite of the written one: **at the
ridge density the sensitivity to ground-state lag is essentially
temperature-independent across the whole decade.**

---

# PART 4 — The attack that has nothing to do with physics

| | |
|---|---|
| **Abstract** | compiles as `[TODO: write last]` in red. The intended text exists only as a comment, and two of its seven beats — "at least 39 % at 1 eV" and "largest at the ionizing–recombining crossover, i.e. detachment" — are claims the body explicitly retracts |
| **Chapter 4** | not `\input`. One page in the ToC. The 47 KB file holding every validation number is not in the document; `\ref{sec:fujimoto}` resolves to an empty skeleton |
| **Chapter 7** | not `\input`. One page. Every conclusion, novelty statement, retraction and the non-invertibility result live in a file the reader never sees. The skeleton titles it *Conclusions*, the file *What It Means*, and both carry `\label{ch:conclusions}` |
| **Appendices A, B, C** | empty `\chapter` stubs. Appendix B is the atomic-data provenance table `claim_hierarchy` calls "the strongest rung in the project" |
| **Body text** | eight `\todo{}` in Chapter 6, seven `[UNVERIFIED]` in Chapter 5, three in Chapter 7, and a 31-line "COORDINATOR NOTES (delete before submission)" header in `chapter4.tex` |
| **Front matter** | no declaration, no certificate, no nomenclature list despite ~30 preamble macros |

**This is more likely than any physics attack to set the examiner's opening
mood, and it interacts with the physics in one specific way: the thesis's
answers to Q9, Q11 and Q17 live in chapters 4 and 7.** A candidate who says
"that is addressed in my conclusions chapter" about a chapter the examiner
cannot find is worse off than one who never wrote it.

---

# PART 5 — Verification of this document's own numbers

Run read-only against the canonical artifacts with
`/opt/anaconda3/envs/cr/bin/python`. Scripts at `/tmp/chk_pe.py`,
`/tmp/chk_tr.py`, `/tmp/chk_sel.py`.

| Check | What it tests | Result |
|---|---|---|
| **1** | My CRE solve and state indexing, against ADDENDUM B's recorded $R = 0.777768$ at $[23,5]$ | $0.7777675$ — **agrees to $5.9\times10^{-7}$** |
| **2** | My independently coded PE state, against the map's recorded $\varepsilon_{\rm plateau}$, at 10 point/direction cases | max relative difference **$1.2\times10^{-14}$**, 0 mismatches beyond $10^{-6}$. Also fixes the sign: heating → $R^{\rm PE}>R^{\rm CRE}$, cooling → below |
| **3** | Apparent-$T_e$ census rebuilt on the *direct* PE state rather than a reconstruction | 25 on-table / 20 off-table, median $-40.0$ %, range $-43.2$ % to $+117.3$ % — identical to the reconstruction |
| **4** | CX-diffusive escape model, against ADDENDUM D.5's 72 µs anchor; and ionisation fraction, to justify $n_p\approx n_e$ | anchor reproduced to 72.2 µs; ionisation fraction $\ge0.983$ over the whole $T_e\ge2$ eV set |
| **5** | Whether the closed-parcel assumption fails selectively at the breakdown points | **0 % of the 45 satisfy it at $L\le20$ cm, against 37 % of the other 403** |

**Two corrections this pass made to its own earlier draft**, recorded because
the project's standard requires it:

- The reservoir-gain stability was written as 6.74 %. `chapter7.tex:142` says
  **6.3 %**. Corrected.
- The atomic-data attack quoted the 29.0 % aggregate as though the thesis had
  hidden the decomposition. `chapter2.tex:1024-1032` gives the decomposition
  explicitly and it defends the model (82.1 % within 20 %, 12.7 % mean error for
  $n_{\rm up}\le4$). **That attack was withdrawn** and replaced by the narrower
  one that survives: the $n=5$ scaling test Chapter 2 names and did not run.

**Caveats on my own numbers.** The transport estimate inherits D.5's
$\sigma_{\rm cx}$, assumes $T_n=T_e$ and a homogeneous slab; at $[15,3]$ the mean
free path is 4.6 cm against a 20 cm slab so the diffusive limit is marginal,
though the free-streaming alternative there is shorter and strengthens the
attack. The apparent-$T_e$ census inverts against the model's own CRE table —
which is exactly what A2 says has never been validated against a real one — so
it is an internally consistent statement about this model, not a claim about
ADAS. The two-valued bands use linear interpolation between the 50 grid nodes.

---

# PART 6 — What must change, ordered by the cost of not doing it

1. **Compile chapters 4 and 7, write the abstract, fill or delete the
   appendices, strip every `\todo` and `[UNVERIFIED]`.** Nothing else matters if
   the examiner opens a document with a red `[TODO]` where the abstract belongs.
2. **Replace the census with the tolerable-renewal-time map, or attach the
   conditional to every sentence that quotes it (A1).** The only fatal physics
   item with a cheap repair.
3. **Delete `chapter3.tex:1377-1387`** and produce one $R^{\rm CRE}$-vs-ADAS-PEC
   figure, or reword the framing so $\varepsilon$ is stated as a distance
   between two states of one model (A2).
4. **Read Fujimoto Table 4.1(b) from the book**; withdraw or rewrite
   §`sec:r1_deficit` (Q11).
5. **Scale the $n=5$ excitation block and recompute** — the one test Chapter 2
   names and says was not performed — and reconcile the 10 % threshold's
   rationale with the 12.7 % measured error (Q11).
6. **Add the temperature-error column and the on/off-table split (A6).**
7. **Map $\partial\ln R/\partial\ln T_e=0$ and state the fold (A5).**
8. **Close A23** — write the reduction script behind the 6.3 % and 8.44 figures,
   which are the pivot's headline and currently carry an `[UNVERIFIED]`.
9. **Settle the escape-factor geometry** (Q6). Arriving without a decision is
   worse than either decision.
10. **Run `verify_bundling_psm20.py` with its two defects understood, or declare
    the bundled-$\ell$ assumption a scope limitation** (Q13). Not both.
11. **Placement:** the memory-kernel paragraph into Chapter 3 (Q8); Greenland and
    Sawada–Fujimoto into Chapter 1 (Q10); the opacity chain beside the logistic
    derivation in Chapter 5 (Q6); the falsifiable recovery signature into
    Chapter 7 (Q17).
12. **The title.** Already changed on the title page — confirm the registered
    title changes with it, before examiners are appointed.
13. **Final grep against the PDF:** `25 ns` · `M = 611` · `−46%` · `1.18 µs` ·
    `ne^-1.00` · `38.7%` · `detachment` · `ITER reference` · `QSS breakdown`.

---

## The single hardest question, if the examiner asks only one

> *"Your title says divertor. Your Chapter 6 says it is not a divertor result
> below 2 eV, it is not detachment, it cannot say what a real divertor does, and
> it cannot name the density its own ridge sits at as a region of a tokamak.
> Above 2 eV, where you say the model is self-consistent, not one of your
> forty-five breakdown points satisfies the transport assumption you computed
> them with. So what, exactly, is the divertor result?"*

**The answer that survives it** is not a defence of the census:

> The divertor is where the question comes from, not where the answer is
> quantified. What this thesis establishes is a property of a class of
> collisional–radiative model: the quasi-steady-state closure is exact to
> $10^{-8}$, the failure is in the ionisation-balance assumption underneath it,
> that failure has a closed form $\varepsilon=|\exp(\bar S G\Delta\ln T_e)-1|$
> with two tabulated structural maps, and it has a ceiling
> $|d\ln R/d\ln b_1|<1$ that no atomic dataset can raise. Every one of those is
> a statement about the operator, and none depends on transport, molecules or
> the neutral density. Applying them to a real divertor needs a ground-state
> renewal time a zero-dimensional model cannot supply — so I report the renewal
> time each point would tolerate, rather than asserting one.

**If that is the answer, the title should say so.** The word "Divertor" in the
subtitle is currently doing work that Chapters 6 and 7 spend twenty pages taking
back.
