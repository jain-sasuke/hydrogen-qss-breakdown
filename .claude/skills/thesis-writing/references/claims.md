# Claims, evidence, and quantitative discipline

## Claim types and what each requires

Classify every substantive claim before writing it, and never let one type
quietly become another.

| Type | Claim | Requires |
|---|---|---|
| A | Established physics | Authoritative literature, unless genuinely standard knowledge |
| B | Numerical or model fact | A traceable script, dataset, table, figure, calculation, or source |
| C | Result of this work | Evidence inside the thesis |
| D | Interpretation | Logical follow-through from evidence, with inference distinguished from observation |
| E | Novelty | Comparison against the relevant literature |
| F | Causal | A demonstrated mechanism — correlation does not promote itself |

The common silent promotions are B written as A (a model output stated as
established physics), D written as C (an interpretation stated as a result), and
F assumed from a co-varying trend.

## Verification, validation, analysis

Keep these logically separate, and never use a physically interesting result as
evidence that the model is correct.

- **Verification** — were the intended equations solved correctly? Unit tests,
  detailed balance, conservation, analytic limits, convergence, an independent
  code path, matrix identities.
- **Validation** — does the model reproduce trusted data or established
  behaviour?
- **Analysis** — what does the validated model reveal?

Chasing whether a specific number survives verification is `thesis-verification`,
not this skill. What this skill owes is prose that does not claim more standing
for a number than it has.

## Quantitative discipline

Replace evaluative adjectives with numbers wherever a number exists. Not *the
agreement is excellent* but *the two calculations agree within 1% over 98.7% of
the tested transitions*. Where relevant, give magnitude, units, range, reference
condition, uncertainty, comparison baseline, and the definition of the error
metric.

The opposite failure is real too: numbers that do not affect the argument make
the ones that do harder to find.

## Scope language

These are different claims and the wording must distinguish them:

- **this work shows** — established by evidence presented here
- **this result suggests** — consistent with, not established
- **this work does not test** — outside what was simulated
- **future work could determine** — open

Never stretch a computational result beyond the domain actually simulated. A
result for atomic hydrogen does not generalise to a divertor plasma without
naming the physics left out.

## Novelty

State the smallest defensible novelty claim. Not finding an implementation is not
evidence that none exists. Genuine contributions include a new result, a new
explanation, a new connection between known ideas, an existing method applied to
an untreated regime, better validation, quantifying an error previously assumed
negligible, identifying a failure regime, separating mechanisms previously
conflated, or a reproducible model that enables a new analysis.

The contribution should sharpen through the thesis — stated as a question in the
introduction, as what had to be built in the methods, as what was found in the
results, as why it happens in the discussion, and in the conclusion as exactly
what can now be claimed that could not be before. Copying the same sentence into
all five is a wasted opportunity and reads as padding.

## Negative results

A result that contradicted the expectation is not a problem to be hidden. Write
what was expected, what was observed, why the discrepancy matters, which
explanations were tested, what was ruled out, and what is unresolved. That
sequence is often stronger science than a confirmation.

## Figures

Every important figure answers a question. Identify it before writing the
discussion.

Where appropriate: what is plotted, what comparison matters, the dominant trend,
the quantitative feature, the exception, the physical explanation, the
implication. Do not narrate every visual feature and do not repeat the caption —
the caption says what is plotted, the text says what it means.

## Tables

Use one when exact values or structured comparison matter. Do not restate the
table in prose; extract only the values the argument uses.

## Methods must be reconstructable

A competent researcher should be able to rebuild the method without reverse-
engineering the code: governing equations, state definitions, processes included
*and excluded*, input data and provenance, interpolation and extrapolation,
units, detailed-balance relations, boundary and initial conditions, solver,
tolerances where consequential, convergence checks, validation tests, parameter
domain, approximations.

Implementation trivia belongs outside the main argument unless it affects
reproducibility or correctness. And an implementation choice is not a physical
principle — keep the chain visible: physics → mathematical model →
discretisation → numerical implementation → result. Code shows what was done, not
that it was justified.

## Assumptions ledger

Track per chapter: the assumption, why it is used, its validity regime, and what
changes if it is violated. Bring each into the prose where it affects
interpretation. A single late disclaimer section collecting assumptions the
reader needed twenty pages earlier satisfies nobody.

## Literature

Organise around the argument, not the authors. *Smith studied X, Jones studied Y,
Kumar studied Z* is a catalogue, and a catalogue makes no claim.

Build instead: what is established; how; where results agree; where they differ;
which assumptions differ; what remains uncertain; which unresolved issue
motivates this work. For each item distinguish consensus from an individual
result, a methodological choice, an assumption, an interpretation, and an
unresolved disagreement.

Citations support claims; they do not substitute for reasoning, and a reference
added to make a paragraph look scholarly is noise. Search for the specific claim
rather than the topic when checking constants, provenance of rates, historical
statements, accepted parameter ranges, claims that a method is standard, claims
of agreement or disagreement, and novelty.

Where sources disagree, compare definitions, assumptions, regimes and unit
conventions, prefer primary evidence, and state the disagreement honestly rather
than manufacturing consensus.

Never invent a reference. Mark `[SOURCE REQUIRED]` instead.
