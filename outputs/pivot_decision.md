# Pivot Decision — what the evidence supports, and the A-tier reframe

**Written 10 September 2026**, answering: do the results support the claim
hierarchy, and if not, what should the thesis claim instead?

**Answer: the hierarchy holds. The pivot needed is in the HEADLINE, not the
structure.** Claims B–F are all supported by verified evidence. What fails is
the *quantity currently headlined* — a step-size-dependent percentage that is
not a property of the plasma.

---

## 1. What the evidence supports

| Claim | Verdict | Strongest evidence |
|---|---|---|
| B — excited states relax fast | **holds** | τ_relax = 2.277 ns at [23,5], 4 sf, from a SHA-verified matrix; reproduced by three independent implementations |
| C — reservoir relaxes slowly | **holds, with a caveat** | τ_QSS = 22.73 µs, M = 9982. **C must not be allowed to imply M predicts failure** — see §2 |
| D — partial equilibrium exists | **holds, strongly** | two-channel split exact to 3.075×10⁻¹⁴ over all 400 points; analytic PE matches LSODA to 6 digits at [0,0] |
| E — spectroscopy sees the PE state | **holds — this IS the thesis** | QSS residual 6.73×10⁻⁹ at the worst point. The closure is exact; ionisation balance is what fails |
| F — the magnitude has a mechanism | **holds** | d ln R/d ln u = f₃−f₄ verified to 1.4×10⁻¹¹ against finite differences; max\|f₃−f₄\| = tanh(\|Δ\|/4) proved |
| A — the model is credible | **improving** | The Fujimoto objection is now dissolved (§3). Gate 3 audit of missing mechanisms is running |

---

## 2. What does NOT survive, and must leave the headline

1. **"38.7%".** ε_plateau is linear in the temperature step, and the step
   (+4.81%) exists only because someone chose 50 log-spaced points between
   1 and 10 eV. At +9.85% it is 87.4%; at +20.68%, 190.4%. Quoted bare it is
   meaningless, and an examiner asking "what if the ELM is 10%?" has no answer.
2. **"The ridge at 1.93×10¹³."** Two independent agents concluded the same
   thing: a real crest, but every row sits above 90% of its own maximum across
   roughly a decade, and the parabolic vertex is at 1.65×10¹³. **The location
   is resolved to about a factor of 3.**
3. **"It is detachment."** Retracted; 5.2× below the detached band.
4. **"M is largest where QSS is worst."** False. The maxima are 52× apart in
   density; at the M maximum ε is the 75th percentile.
5. **Divertor magnitude claims below 2 eV.** Optically thick by 5 orders,
   no molecules, no transport.

---

## 3. The Fujimoto objection is dissolved

The referee pass called the r₁ deficit (8.3× at p=3, 4.4× at p=4) the single
strongest objection to the thesis. It is not evidence against the model:

- **Normalisation refuted.** A Z(p)/Z(1) mismatch grows with p (1.44, 2.69,
  4.47, 6.77 … at 10 eV); the observed deficit *shrinks* with p (10.8, 8.3,
  4.4, 3.1 …). Wrong sign. And r₀, which shares the same Z(p), agrees to
  1.3% at p=2 and 12–14% at p=3,4.
- **Case A/B refuted.** Zeroing A(p→1s) can raise r₁ by at most 2.26× at p=3
  and 1.71× at p=4, against 8.29× and 4.39× needed. In the other direction,
  full case B would put r₁(2) ~70× ABOVE the tabulated value.
- **What it actually is: the ℓ-treatment.** Turning proton ℓ-mixing off moves
  r₁(3) by 3.35× and r₁(2) by **118×** at exactly the disputed density. Where
  the answer barely depends on ℓ-treatment (high density) the model agrees
  with Fujimoto within a factor 2. **A benchmark whose input assumption spans
  3–118× cannot adjudicate an 8× discrepancy.** It tests the ℓ-closure, not
  f₃−f₄.

**The honest statement for the thesis:** the benchmark tests the ℓ-closure and
neither variant implements Fujimoto's assumption. What must be looked up in the
book is one sentence — whether Table 4.1(b)'s level structure is bundled-n with
statistical ℓ, or ℓ-resolved.

---

## 4. The A-tier reframe: report structure and a bound, not a magnitude

**There is a step-independent invariant, and the step-dependent percentage is
derived from it.** Measured 10 Sep 2026 against the canonical matrix:

$$\varepsilon = \left|\exp\left(\bar S \cdot G \cdot \Delta\ln T_e\right) - 1\right|, \qquad G \equiv \frac{d\ln u}{d\ln T_e}$$

| point | G at k=1 | k=2 | k=4 | spread over a 4× step range |
|---|---|---|---|---|
| benchmark [23,5] | −5.736 | −5.634 | −5.439 | 5.5% |
| ridge [15,3] | −7.766 | −7.617 | −7.333 | 5.9% |
| worst [0,4] | −14.434 | −14.131 | −13.551 | 6.5% |

**Corrected and scoped, 10 Sep 2026**, against
`validation/reservoir_gain/reservoir_gain.csv` (2288 rows). An earlier version
of this document quoted the three points above and generalised from them, then
gave |S̄| and |G| ranges without saying what set they were extrema over. That
is the same defect this document criticises elsewhere, and it happened here.

Over all **736** (direction, point) triples carrying k = 1, 2 and 4:

| | median spread across k | maximum |
|---|---|---|
| \|G\| | 1.0555 | **1.0674**, i.e. **6.74%** |
| ε | 3.73 | **8.44** |

Ranges, each with its scope stated:

| quantity | over all 2288 rows | over window_ok heating at k=1 (338 rows) |
|---|---|---|
| \|S̄\| | **0.0355 to 0.4963** | 0.0649 to 0.4822 |
| \|G\| | **2.6396 to 14.5192** | same |

The earlier "0.065 to 0.482" for |S̄| was the restricted set quoted as though it
were global. |G|'s range does hold over all rows.

G is stable to **6.74%** while ε varies by up to a factor **8.44** over the same
step range. The small-step limit is **dε/d ln Te = |S̄ · G|**. Quote 6.74%, and
say G is *stable*, not invariant.

So the thesis reports **two structural maps** — the sensitivity S̄ = f₃ − f₄
(bounded above by tanh(|Δ|/4) < 1, data-independent) and the reservoir gain G
— and any reader computes the error for the transient they care about. That
converts the weakest claim into the strongest: "38.7% for a step nobody can
justify" becomes "here is the closed-form response, here is its universal
bound, compute your own case."

### The headline sentence

> The quasi-steady-state closure is exact to one part in 10⁸ — the
> approximation everyone worries about is not the problem. What fails is the
> ionisation-balance assumption hidden inside the lookup table. Its cost has a
> closed form, and a bound that no atomic dataset can exceed.

### What makes this A-tier rather than B-tier

1. **A reversal**, not an incremental measurement. The expected culprit is
   exonerated to 10⁻⁸ and the real one is identified.
2. **A closed-form bound that is data-independent.** |d ln R/d ln b₁| < 1 for
   every shell pair, every (Te, ne), every atomic dataset: a line ratio can
   never respond, in relative terms, faster than the reservoir driving it.
3. **A derived mechanism**, not a fitted correlation. The maximum is a
   *position* effect — it sits where ln(u_CRE/u_peak) changes sign, with Δ
   nearly flat across density.
4. **A negative result that is quantitative:** timescale separation does not
   predict validity. Greenland (2001) stated the general form; this quantifies
   it for a specific diagnostic and maps where it bites.
5. **Errors found, traced, bounded, and kept in.** The ℓ-mixing F(U_m) error,
   the eigenvalue filter, the misnamed central quantity, and now a step-size
   convention that inflated the headline.

### What must change in the written chapters

- Chapter 5 leads on the two structural maps (S̄ and G), not on a percentage.
  Every ε quoted carries its step size.
- The bound gets its own section, stated as data-independent.
- The mechanism is stated as a position effect (ADDENDUM B §B.2).
- The density maximum is quoted as a range, 7×10¹² – 5×10¹³ cm⁻³.
- Chapter 4 reports the Fujimoto benchmark as an ℓ-closure test, with the
  three refutations above, and the one sentence needed from the book.
- Abstract sentence 5 currently says "at least 39% at 1 eV" — that is exactly
  the claim being retired.
