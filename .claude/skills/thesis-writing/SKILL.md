---
name: thesis-writing
description: Write or rewrite scientific thesis prose — physics, plasma, chemical or general engineering — so that it develops ideas from first principles, stays readable by a first-year Master's student, and survives an examiner. Use whenever the user wants text produced or restructured rather than judged — "draft this section from my notes", "write up these results", "turn these plots into a results section", "rewrite this paragraph", "restructure this chapter", "make this readable", "write my abstract or introduction or conclusion", or when they hand over notes, code, figures or bullet points and want prose. Also use when acting on review findings the author already has. Do not use to judge whether a draft holds up — that is physics-thesis-review, which runs first when the text to be rewritten has not been reviewed. Do not use to establish whether a computed number is correct — that is thesis-verification.
---

# Thesis writing

Produce thesis prose in which the reader repeatedly knows why the question
arose, where the equation came from, what it assumes, what the result means, and
why the next section follows.

The failure mode this skill exists to prevent is fluent text that outruns its
evidence. A reviewer who invents a finding costs the author an afternoon. A
writer who invents a rate coefficient, a mechanism, or a convergence result puts
it in a submitted thesis under the author's name. Fluency is the easy part and
it is not the goal.

## Where this skill starts and stops

This skill writes. It does not certify.

- **Is the argument defensible?** → `physics-thesis-review`.
- **Is the number real?** → `thesis-verification`.
- **Would an examiner break this claim?** → `thesis-defense`.
- **Is this reviewer's criticism correct?** → `review-audit`.

**When asked to rewrite text that has not been reviewed, say so before
rewriting.** Polishing unreviewed prose is how a weak derivation acquires
authority: the sentences get better, the physics does not, and the improved
version is harder to challenge because it now reads as though someone checked
it. Offer the review first. If the author declines — deadline, or they only want
a paragraph tightened — proceed, and keep every scientific reservation visible in
the output rather than dissolving it into smoother wording.

Drafting new prose from the author's own notes and results does not require a
prior review. Revising existing scientific argument does.

## The reader

A first-year Master's student in the relevant discipline: comfortable with
calculus, differential equations, linear algebra, thermodynamics, transport,
kinetics and basic numerics; not familiar with this subfield's notation,
acronyms, databases, diagnostics, or conventions.

They lack context, not intelligence. Explaining what an eigenvalue is wastes
their time and signals that the rest of the writing has misjudged them.
`physics-thesis-review/references/writing.md` holds the shared prose standard —
read it before making stylistic choices, since the review skill will be checking
against it.

## Two entry paths

### A — Drafting from source material

Notes, code, results, figures, bullet points, or a supervisor's outline. Run the
five preparation passes below, then write.

### B — Revising existing prose

Check first whether review findings exist. If they do, work from them: they
define what must change and, just as usefully, what must not. If they do not,
apply the deferral above.

When revising, preserve verified technical meaning, equations, numerical
results, citations, and the author's genuine contribution. Do not preserve poor
ordering, duplicated explanation, artificial transitions, unsupported
interpretation, or bloat. Reorganisation is allowed and often the largest
available improvement — but never change a number to make a sentence flow.

## Before writing: five passes

Skipping these is what produces text that sounds finished and argues nothing.

**1. Inventory the material.** Sort what you have been given into: established
physics; this work's assumptions; definitions; equations; computed results;
interpretations; open questions. Mark every item you cannot trace to the
supplied material. Nothing carrying that mark may be written as fact.

**2. Order by dependency, not chronology.** Research order is rarely
explanatory order. Determine which idea must be understood before which.

**3. State the section's job in one sentence** — as an intellectual problem, not
a topic. "This chapter discusses the collisional-radiative model" names a
subject. "To determine whether a steady-state line-ratio inversion survives a
transient, the atomic populations must first be evolved without assuming
instantaneous equilibration" names a problem, and a problem is what generates
prose.

**4. Build the skeleton:** what the reader already knows → what is missing →
the reasoning or evidence that supplies it → what is now established → what
question this opens.

**5. Check prerequisites.** What would the reader above need that no earlier
section has given them? Add only what is actually required. Prerequisites added
defensively become the padding that later gets cut.

## Writing

Develop unfamiliar ideas up the ladder in `references/ladder.md`: physical
question → governing principle → simplest equation → define every object →
add processes → state assumptions where they enter → generalise → check →
interpret → connect. The ladder is the difference between a derivation the
reader can reconstruct and one they can only accept.

`references/architecture.md` covers thesis backbone, chapter entry and exit,
section shapes, transitions that carry logic, the repetition ledger, and
progressive compression.

`references/claims.md` covers claim types and the evidence each requires,
quantitative discipline, figures and tables, literature organisation, scope
language, and novelty.

Read the relevant file rather than working from memory of it — these hold the
specific patterns, and approximate recall of them produces approximate writing.

## Gates

Self-assessment scores are not a check. A model asked to rate its own draft out
of ten will report an eight and stop. These gates can fail, which is the only
property that makes a check worth running. Apply them to the drafted text and
report the result.

**Symbol gate.** Every symbol defined at or before first use, with units and
type (scalar, matrix, rate, rate coefficient, density, probability). List any
that are not.

**Equation gate.** Every displayed equation has a stated origin — which
conservation law, prior equation, or principle it descends from — the assumption
that admits it, and one stated consequence that is not a restatement of the
equation itself. "Eq. (3.8) describes how the population evolves" immediately
after Eq. (3.8) fails this gate.

**Number gate.** Every quantitative claim names the artifact it came from:
script, dataset, table, figure, cited source. Anything untraceable is written as
`[UNVERIFIED: what would have to be checked]` and left visible. Never smooth a
gap into confident phrasing.

**Assumption gate.** Each assumption appears where it is first used, not in a
disclaimer section later, and states what physical freedom it removes.

**Repetition gate.** Any fact stated twice must have changed function between
the two — defined, then derived, then interpreted, is legitimate progression.
Restating a definition is not. Cut or justify each recurrence.

**Claim gate.** Check the verb against the evidence. *This work shows* / *this
result suggests* / *this work does not test* / *future work could determine* are
different claims. Never let a correlation acquire a causal verb, or a
verification acquire the standing of a validation.

**Filler gate.** Delete-test the flagged phrases. If a sentence loses nothing
when a phrase is removed, the phrase was decoration.

**Reader gate.** Walk the text as the reader above and name the first sentence
where they would ask "where did that come from?" or "why are we calculating
this?". Repair that jump specifically. "Needs more explanation" is not a
diagnosis.

Then report honestly: which gates the draft failed, what was repaired, and what
remains open. A draft handed back with its open problems named is more useful
than one handed back clean.

## Never invent

Not physics, not mathematics, not citations, not numerical values, not
mechanisms, not agreement between sources.

Where evidence is missing, mark it — `[SOURCE REQUIRED]`, `[UNVERIFIED]`,
`[MECHANISM NOT ESTABLISHED]` — and continue. A marker is a work item the author
can close in a minute. A fabricated citation survives to the defense.

Where the source material appears to contain a scientific error, flag it and
stop. Do not write the error into authoritative prose, and do not quietly
correct it either: a silent repair hides a problem the author needs to know
about, and may be wrong.

Where sources genuinely disagree, compare their definitions, assumptions,
regimes and units, prefer primary evidence, and state the disagreement. Never
flatten it into a consensus that does not exist.

## Reference files

- `references/ladder.md` — first-principles development, mathematics, mechanism
  explanation, limiting cases.
- `references/architecture.md` — thesis and chapter structure, transitions,
  repetition, compression, detail allocation.
- `references/claims.md` — evidence standards, quantitative discipline, figures,
  literature, scope and novelty.
