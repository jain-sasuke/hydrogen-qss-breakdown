# Novelty classification — Gate 6

**Verified 10 September 2026** against Crossref (all DOIs re-queried), OpenAlex,
NASA ADS, and arXiv/ar5iv full text where extractable. Blocked: ScienceDirect,
royalsocietypublishing.org, Cambridge Core and IOP full text.

**Headline: two of the three items `findings_10` §9 called "genuinely new" must
be downgraded. The tanh identity is in print.**

Four categories, and the rule is to default to the weaker one. A false novelty
claim is far more damaging than an over-modest one.

---

## Claim 1 — the two-channel decomposition n_p = a_p·u + c_p

**Established. Attribution correct but insufficient.**

Bates, Kingston & McWhirter (1962) and Fujimoto & McWhirter (1990) are both
correctly cited and Crossref-verified. Two predecessors are missing:

- **McWhirter & Hearn (1963), Proc. Phys. Soc. 82, 641–654.** The paper that
  first *tabulated* the two coefficients. Nearest predecessor to the split, and
  it appeared nowhere in the repository.
- **Salzmann (1979), Phys. Rev. A 20(4), 1713.** Derives *analytical
  expressions* for r₀ and r₁. Any closed-form claim must clear it first.

Also: `chapter3.tex:1352-1353` credits the ionising/recombining classification
to Fujimoto (2004) alone. The primary sources are the JPSJ series, already in
the bibliography and literally titled for it.

Chapter 3 already says "Nor is it new". Nothing to defend.

## Claim 2 — the logistic form with unit width

**Independently derived**, and the previous judgement was if anything generous.

Structurally identical objects, all textbook: Michaelis–Menten fractional
saturation, the Rasch/1PL item characteristic curve, and Fermi–Dirac occupation,
which has unit width for the same structural reason.

**Nearest predecessor in this field, uncited for this: Verhaegh et al. (2017),
Nucl. Mater. Energy 12, 1112.** It defines F_exc(n) and F_rec(n), the fractions
of Balmer-n brightness from excitation and recombination, and infers them from a
Balmer line ratio. **F_exc(n) is the ground-fed fraction f_p under another
name.** `Verhaegh2019` is cited in Chapter 1, but only for the neutral fraction
as a free parameter, never for this. Most likely omission to be raised by a PPCF
referee.

`chapter7.tex:266-268` already says the right thing. Do not strengthen it; add
one clause naming F_exc(n).

## Claim 3 — the tanh bound. Three separable statements.

### 3(a) The identity — **independently derived. It is in print.**

**Foieri, Sánchez, Arrachea & Gopar (2006), Phys. Rev. B 74(16), 165313.**
Crossref-verified. Read from ar5iv full text, §II: they write the difference of
two Fermi functions displaced by eV and approximate it by a rectangle of
**height tanh(eV/4kT)** and width eV/tanh(eV/4kT).

A Fermi function is a unit-width logistic in E/kT; eV/kT is Δ. **Both** of the
thesis's statements are there: the peak height *and* the area Δ. They state it
without derivation or citation, which suggests it was already regarded as known
in that field.

Not found in item response theory, the logistic-distribution literature, or the
difference-of-sigmoids neural-network literature. That is absence at shallow
depth and nothing more.

### 3(b) The corollary |d ln R/d ln b₁| < 1 — **established**, in another vocabulary

Each f_p is the elasticity of a function affine in the reservoir, and that an
elasticity is bounded by the Hill coefficient, here unity, is standard metabolic
control analysis (Kacser & Burns 1973; **Heinrich & Rapoport 1974**). The tanh
sharpens the bound from 1; it does not establish it.

`claim_hierarchy.md` F.2 lists this corollary among what is new. **Downgrade.**

### 3(c) The application — **new**, and the only part of Claim 3 to claim

That the inequality bounds the diagnostic error of a hydrogen Balmer ratio with
respect to its ground-state reservoir, uniformly over every atomic dataset, and
is near-tight where the error is largest. No prior statement of a
data-independent ceiling on a CR line ratio's response to the neutral reservoir
was located.

## Claim 4 — the exact factorisation and the two-axis attribution

**4(a)** ln(R_PE/R_CRE) = S̄·Δln u — **established**. Fundamental theorem of
calculus plus a mean value. The thesis says so twice, in the right places.

**4(b)** ε = |exp(S̄·G·Δln Te) − 1| — **adapted**. Structurally the partitioned
response of metabolic control analysis: response = control × elasticity. Applied
to a new object.

**4(c)** The two-axis attribution — **new, weakly**. A measurement on this
matrix, not a theorem, and Chapter 5 labels it correctly.

## Claim 5 — timescale separation does not predict validity

**Established in general form (Greenland); the quantification is new.** Both
quotes verified and correctly split between the two 2001 papers.

**Wording caveat:** the published abstract reads "not related to equilibrium
time-scales", with no definite article. `chapter1.tex:674` renders it "not
related to *the* equilibrium timescales *of the system*" and
`findings_10.md:536` and `claim_hierarchy.md:447` put that version inside
quotation marks. Paraphrase is fine; quotation marks are not.

**The thesis undersells one point.** Greenland's criteria govern whether a
*reduced* model reproduces the *full* dynamics. This thesis's error is not a
model-reduction error at all: the reduction is exact to 6.73×10⁻⁹. So M fails to
predict a quantity Greenland was not discussing. Add that clause; it costs
nothing and is defensible.

## Claim 6 — the partial-equilibrium state as what spectroscopy observes

**Established in general form; the hydrogen instance is adapted.** The weakest
novelty position of the six, and the thesis makes no explicit claim for it,
which is correct.

The generic statement is the whole content of the **non-equilibrium ionisation**
literature in solar and astrophysical spectroscopy. Verified exemplar: **Olluri,
Gudiksen & Hansteen (2013), ApJ 767(1), 43**, reporting deduced densities up to
an order of magnitude wrong under NEI. **The thesis cites none of this
literature.** Within CR modelling, the GCR framework of Summers et al. (2006) is
built on the same separation and is cited, but never for this point.

What differs, in decreasing strength: the demonstration that the *excited-state*
closure is exact at the very point where the diagnostic is worst; the explicit
one-linear-solve form; and the Hα/Hβ map.

---

## The nearest predecessor, and the largest open risk

**Sawada & Fujimoto (1994), Phys. Rev. E 49, 5565.** Title verified verbatim.
The thesis's characterisation in `chapter1.tex:640-665` matches the abstract
word for word.

**But the same abstract also says "the overall response of excited level
populations to ionization and recombination rates was also examined."** The body
has not been read. If that section treats the response to a *changing*
ground-state balance, then `chapter7.tex:316-317`, "The two results are
compatible and neither contains the other", is at risk, and so is part of
Claim 6. **That sentence cannot be defended from the abstract alone.**

## Six citations to add, all verified

| # | reference | bears on |
|---|---|---|
| 1 | McWhirter & Hearn (1963), Proc. Phys. Soc. 82, 641 | Claim 1, first tabulation |
| 2 | **Foieri et al. (2006), Phys. Rev. B 74, 165313** | Claim 3(a), the tanh identity in print |
| 3 | Verhaegh et al. (2017), Nucl. Mater. Energy 12, 1112 | Claim 2, F_exc is f_p |
| 4 | Salzmann (1979), Phys. Rev. A 20, 1713 | Claim 1, analytic r₀, r₁ |
| 5 | Olluri et al. (2013), ApJ 767, 43 | Claim 6, NEI diagnostic error |
| 6 | Heinrich & Rapoport (1974), Eur. J. Biochem. 42, 89 | Claims 3(b), 4(b), elasticity |

**All six added to `references.bib` on 10 Sep 2026.** The `Verhaegh2017` author
list carries a note requiring re-check before use.

## Nine bibliography defects

**B1.** `chapter1.tex:124` cites `\cite{Stangeby2023, Stangeby2023}` — the same
key twice. Part A (Nucl. Fusion 63, 016016, doi ac9916) has a `% VERIFIED`
comment but **no entry**. Add it or delete the duplicate.
**B2.** `references.bib:369` — Kaveeva2020's verification comment names DOI
`10.1088/1361-6587/ab73c1`, which **does not exist**. The entry's own doi field
is correct. The data is right; the audit trail is wrong.
**B3.** Two orphaned `% VERIFIED` blocks with no entries beneath them, which is
how B1 happened.
**B4.** Anderson2002's title carries a "Corrigendum:" prefix not in the record.
**B5.** Krasheninnikov2017's page number is correct but is not a Crossref field,
while the comment claims Crossref verification.
**B6.** The Greenland quote inserts a definite article the abstract lacks.
**B7.** Four entries never cited: Bates1962b, Griem1964, Loarte2003, Kaveeva2020.
**B8.** Fujimoto2004 imprint: LC says Clarendon, Crossref says OUP. Not an error.
**B9.** The three references above were missing entirely.

Confirmed correct, so nobody re-audits: all five Fujimoto JPSJ parts, including
**Part IV at 49(4), 1569**; Bates II at volume **270**; Loarte2007 ending /S04;
Pigarov1998 ending /006; Stangeby2023 at volume 63; Wijkamp2023 as MAST-U. The
earlier IPPJ-AM-8 error is closed, with zero repo-wide hits.

## What remains open, and exactly what closes it

1. **Fujimoto (2004) Ch. 4.** Unread. If a closed form for the ratio of two
   population coefficients appears there, Claims 2 and 3(c) both weaken.
2. **Fujimoto JPSJ I–V.** All five verified bibliographically; none read. Part V
   is the likeliest home for the switching point c_m/a_m.
3. **Sawada & Fujimoto (1994), the body.** The single largest publication risk.
4. **Raju (1988), Psychometrika 53(4), 495.** Exact area formulae for the 1PL
   equal-discrimination case. If he also gives the maximum vertical distance,
   3(a) becomes *established* rather than independently derived.
5. **Balakrishnan (ed.), Handbook of the Logistic Distribution (1992).**
6. **Any tunnelling-spectroscopy monograph.** The Fermi-window height is folk
   knowledge there and is very likely older than 2006.

**Absence of evidence at this depth is not evidence of novelty, and for Claim
3(a) that absence has already been broken once.**

## Documents now superseded

- `findings_10` §9 items 1 and 2. Only item 3 survives, weakly.
- `findings_10` ADDENDUM B §B.5. Its caveat was correct and the outcome it
  warned of has occurred.
- `claim_hierarchy.md` F.2, which lists the corollary among what is new.
- `chapter7.tex:271`, "no prior statement of it was found". **Corrected
  10 Sep 2026.**

`chapter7.tex` §sec:novelty is otherwise the most accurate novelty statement in
the project, closer to right than either supporting document. Keep its
structure.
